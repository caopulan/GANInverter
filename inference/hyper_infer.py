from tqdm import tqdm
from criteria.lpips.lpips import LPIPS
import math
from models.stylegan2.model import Generator
from models.encoders.psp_encoders import ResidualAligner, ResidualEncoder
from models.encoder import get_keys
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.train_utils import load_train_checkpoint
from inference.inference import BaseInference


class HyperstyleInference(BaseInference):

    def __init__(self, opts, decoder=None):
        super(HyperstyleInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # resume from checkpoint
        checkpoint = load_train_checkpoint(opts)

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

        self.align = ResidualAligner().to(self.device).eval()
        self.align.load_state_dict(get_keys(checkpoint['encoder'], 'grid_align'), strict=False)
        self.residue = ResidualEncoder().to(self.device).eval()
        self.residue.load_state_dict(get_keys(checkpoint['encoder'], 'residue'), strict=False)

    def inverse(self, images, images_resize, image_paths, emb_codes, emb_images, emb_info):
        with torch.no_grad():
            emb_images_resize = torch.nn.functional.interpolate(torch.clamp(emb_images, -1., 1.), size=(256, 256), mode='bilinear')
            res = images_resize - emb_images_resize
            res_align = self.align(torch.cat((res, emb_images_resize), 1))
            conditions = self.residue(res_align)

            images, result_latent = self.decoder([emb_codes],
                                                 input_is_latent=True,
                                                 return_latents=True,
                                                 randomize_noise=False,
                                                 hfgi_conditions=conditions)

        return images, result_latent, None

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