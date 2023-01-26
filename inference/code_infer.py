import math
import os
from models.encoder import Encoder
from models.stylegan2.model import Generator
import torch
from utils.train_utils import load_train_checkpoint


class CodeInference:

    def __init__(self, opts, decoder=None):
        super(CodeInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2
        self.code_path = opts.code_path
        # resume from checkpoint
        checkpoint = load_train_checkpoint(opts)

        # initialize and decoder
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = Generator(opts.resolution, 512, 8).to(self.device)
            self.decoder.train()
            if checkpoint is not None:
                self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
            else:
                decoder_checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
                self.decoder.load_state_dict(decoder_checkpoint['g_ema'])

    def inverse(self, images, images_resize, image_name):
        codes = []
        for path in image_name:
            code_path = os.path.join(self.code_path, f'{os.path.basename(path[:-4])}.pt')
            codes.append(torch.load(code_path, map_location='cpu'))
        codes = torch.stack(codes, dim=0).to(images.device)
        with torch.no_grad():
            images, result_latent = self.decoder([codes], input_is_latent=True, return_latents=True)
        return images, result_latent
