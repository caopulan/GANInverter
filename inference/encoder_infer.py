import math
from models.encoder import Encoder
from models.stylegan2.model import Generator
import torch
from utils.train_utils import load_train_checkpoint
from inference.inference import BaseInference


class EncoderInference(BaseInference):

    def __init__(self, opts, decoder=None):
        super(EncoderInference, self).__init__()
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
            self.decoder.train()
            if checkpoint is not None:
                self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
            else:
                decoder_checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
                self.decoder.load_state_dict(decoder_checkpoint['g_ema'])
                latent_avg = decoder_checkpoint['latent_avg']
        if latent_avg is None:
            latent_avg = self.decoder.mean_latent(int(1e5))[0].detach() if checkpoint is None else None
        self.encoder = Encoder(opts, checkpoint, latent_avg, device=self.device).to(self.device)
        self.encoder.set_progressive_stage(self.opts.n_styles)

    def inverse(self, images, images_resize, image_path):
        with torch.no_grad():
            codes = self.encoder(images_resize)
            images, result_latent = self.decoder([codes], input_is_latent=True, return_latents=True)
        return images, result_latent
