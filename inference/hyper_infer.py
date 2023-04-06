from tqdm import tqdm
from criteria.lpips.lpips import LPIPS
import math

from models.hypernetworks.hypernetwork import SharedWeightsHyperNetResNet, SharedWeightsHyperNetResNetSeparable
from models.stylegan2.model import Generator
from models.encoders.psp_encoders import ResidualAligner, ResidualEncoder
from models.encoder import get_keys
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.train_utils import load_train_checkpoint, convert_weight
from inference.inference import BaseInference


class HyperstyleInference(BaseInference):

    def __init__(self, opts, decoder=None):
        super(HyperstyleInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # resume from checkpoint
        # TODO: hyperstyle ckpt load
        checkpoint = torch.load(opts.hypernet_checkpoint_path, map_location='cpu')
        checkpoint = convert_weight(checkpoint, opts)

        # initialize encoder and decoder
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

        if self.opts.hyperstyle_encoder_type == "SharedWeightsHyperNetResNet":
            self.hypernet = SharedWeightsHyperNetResNet(opts=self.opts).to(self.device)
        elif self.opts.hyperstyle_encoder_type == "SharedWeightsHyperNetResNetSeparable":
            self.hypernet = SharedWeightsHyperNetResNetSeparable(opts=self.opts).to(self.device)
        self.hypernet.eval()
        self.hypernet.load_state_dict(checkpoint['hypernet'], strict=True)

    def inverse(self, images, images_resize, image_paths, emb_codes, emb_images, emb_info):
        with torch.no_grad():
            weights_deltas = None
            for iter in range(self.opts.hyperstyle_iteration):
                emb_images = torch.nn.AdaptiveAvgPool2d((256, 256))(emb_images)
                x_input = torch.cat([images_resize, emb_images], dim=1)

                hypernet_outputs = self.hypernet(x_input)
                if weights_deltas is None:
                    weights_deltas = hypernet_outputs
                else:
                    weights_deltas = [weights_deltas[i] + hypernet_outputs[i] if weights_deltas[i] is not None else None
                                      for i in range(len(hypernet_outputs))]

                emb_images, result_latent = self.decoder([emb_codes],
                                                         weights_deltas=weights_deltas,
                                                         input_is_latent=True,
                                                         randomize_noise=False,
                                                         return_latents=True)
                # for path, inv_img in zip(image_path, images):
                #     basename = os.path.basename(path).split('.')[0] + '_' + str(iter)
                #     inv_result = tensor2im(inv_img)
                #     inv_result.save(os.path.join(self.opts.output_dir, 'inversion', f'{basename}.jpg'))

        return emb_images, result_latent, None

    def edit(self, images, images_resize, image_paths, emb_codes, emb_images, emb_info, editor):
        images, codes, refine_info = self.inverse(images, images_resize, image_paths, emb_codes, emb_images, emb_info)
        refine_info = refine_info[0]
        with torch.no_grad():
            decoder = Generator(self.opts.resolution, 512, 8).to(self.device)
            decoder.train()
            decoder.load_state_dict(refine_info['generator'], strict=True)
            edit_codes = editor.edit_code(codes)

            edit_images, edit_codes = decoder([edit_codes], input_is_latent=True, randomize_noise=False)
        return images, edit_images, codes, edit_codes, refine_info