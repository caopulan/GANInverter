import math
import os

from models.encoder import Encoder
from models.stylegan2.model import Generator
import torch

from utils.common import tensor2im
from utils.train_utils import load_train_checkpoint
from inference.inference import BaseInference


class RestyleInference(BaseInference):

    def __init__(self, opts, decoder=None):
        super(RestyleInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # resume from checkpoint
        checkpoint = load_train_checkpoint(opts)
        # initialize encoder and decoder
        latent_avg = None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = Generator(opts.resolution, 512, 8).to(self.device)
            self.decoder.eval()
            if checkpoint is not None:
                self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
            else:
                decoder_checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
                self.decoder.load_state_dict(decoder_checkpoint['g_ema'])
                latent_avg = decoder_checkpoint['latent_avg']
        if latent_avg is None:
            latent_avg = self.decoder.mean_latent(int(1e5))[0].detach() if checkpoint is None else checkpoint['encoder']['latent_avg'].unsqueeze(0).to(self.device)
        self.encoder = Encoder(opts, checkpoint, latent_avg, device=self.device).to(self.device).eval()
        self.encoder.set_progressive_stage(self.opts.n_styles)

        with torch.no_grad():
            self.avg_image, self.avg_latent = self.decoder([latent_avg],
                                                           input_is_latent=True,
                                                           randomize_noise=False,
                                                           return_latents=True)
            self.avg_image = self.avg_image.float().detach()

        # inv_result = tensor2im(self.avg_image[0])
        # inv_result.save(os.path.join(self.opts.output_dir, 'inversion', f'avg.jpg'))

    def inverse(self, images, images_resize, image_path, **kwargs):
        with torch.no_grad():
            for iter in range(self.opts.iteration):
                if iter == 0:
                    avg_image = torch.nn.AdaptiveAvgPool2d((256, 256))(self.avg_image)
                    avg_images = avg_image.repeat(images_resize.shape[0], 1, 1, 1)
                    x_input = torch.cat([images_resize, avg_images], dim=1)
                    result_latent = self.avg_latent.repeat(images_resize.shape[0], 1, 1)
                else:
                    images = torch.nn.AdaptiveAvgPool2d((256, 256))(images)
                    x_input = torch.cat([images_resize, images], dim=1)

                codes = self.encoder(x_input)
                codes = codes + result_latent
                images, result_latent = self.decoder([codes],
                                                     input_is_latent=True,
                                                     randomize_noise=False,
                                                     return_latents=True)
                for path, inv_img in zip(image_path, images):
                    basename = os.path.basename(path).split('.')[0] + '_' + str(iter)
                    inv_result = tensor2im(inv_img)
                    inv_result.save(os.path.join(self.opts.output_dir, 'inversion', f'{basename}.jpg'))

        return images, result_latent, None

    def edit(self, images, images_resize, image_path, editor):
        images, codes, _ = self.inverse(images, images_resize, image_path)
        edit_codes = editor.edit_code(codes)
        edit_images = self.generate(edit_codes)
        return images, edit_images, codes, edit_codes, None