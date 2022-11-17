from tqdm import tqdm

from criteria.lpips.lpips import LPIPS
from utils.common import tensor2im
from .inference import BaseInference
from .encoder_infer import EncoderInference
from .optim_infer import OptimizerInference
from .code_infer import CodeInference
import math
from models.stylegan2.model import Generator
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.train_utils import load_train_checkpoint, requires_grad


class PTIInference(BaseInference):

    def __init__(self, opts):
        super(PTIInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # initialize embedding
        if opts.embedding_mode == 'encoder':
            self.embedding_module = EncoderInference(opts)
        elif opts.embedding_mode == 'optim':
            self.embedding_module = OptimizerInference(opts)
        elif opts.embedding_mode == 'code':
            self.embedding_module = CodeInference(opts)

        # initial loss
        self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()

    def inverse(self, images, images_resize, image_name):
        embedding_images, embedding_latent = self.embedding_module.inverse(images_resize, image_name)

        # resume from checkpoint
        checkpoint = load_train_checkpoint(self.opts)

        # initialize decoder and regularization decoder
        latent_avg = None
        decoder = Generator(self.opts.resolution, 512, 8).to(self.device)
        decoder_r = Generator(self.opts.resolution, 512, 8).to(self.device)
        decoder.train()
        decoder_r.eval()
        if checkpoint is not None:
            decoder.load_state_dict(checkpoint['decoder'], strict=True)
            decoder_r.load_state_dict(checkpoint['decoder'], strict=True)
        else:
            decoder_checkpoint = torch.load(self.opts.stylegan_weights, map_location='cpu')
            decoder.load_state_dict(decoder_checkpoint['g_ema'])
            decoder_r.load_state_dict(decoder_checkpoint['g_ema'])
            latent_avg = decoder_checkpoint['latent_avg']
        if latent_avg is None:
            latent_avg = decoder.mean_latent(int(1e5))[0].detach() if checkpoint is None else None

        # initialize optimizer
        optimizer = optim.Adam(decoder.parameters(), lr=self.opts.pti_lr)

        pbar = tqdm(range(self.opts.pti_step))
        for i in pbar:
            gen_images, _ = decoder([embedding_latent], input_is_latent=True, randomize_noise=False)

            # calculate loss
            loss_lpips = self.lpips_loss(gen_images, images)
            loss_mse = F.mse_loss(gen_images, images)
            loss = self.opts.pti_lpips_lambda * loss_lpips + self.opts.pti_l2_lambda * loss_mse

            if self.opts.pti_use_regularization and i % self.opts.locality_regularization_interval == 0:
                w_samples = decoder_r.w_sample(images.shape[0])
                if embedding_latent.ndim < 3:
                    direction = w_samples - embedding_latent
                    direction_n = direction / torch.norm(direction, p=2, dim=-1, keepdim=True).repeat(1, decoder_r.style_dim)
                else:
                    direction = w_samples.unsqueeze(1).repeat(1, decoder_r.n_latent, 1) - embedding_latent
                    direction_n = direction / torch.norm(direction, p=2, dim=-1, keepdim=True).repeat(1, 1, decoder_r.style_dim)
                regularization_latent = embedding_latent + self.opts.alpha * direction_n

                with torch.no_grad():
                    images_r, _ = decoder_r([regularization_latent], input_is_latent=True, return_latents=True)
                    images_r = F.interpolate(torch.clamp(images_r, -1., 1.), size=(x.shape[2], x.shape[3]), mode='bilinear')

                r_p_loss = self.lpips_loss(gen_images, images_r)
                r_mse_loss = F.mse_loss(gen_images, images_r)
                r_loss = self.opts.r_lpips_lambda * r_p_loss + self.opts.r_l2_lambda * r_mse_loss
                loss += self.opts.r_lambda * r_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f"loss: {loss.item():.4f}; lr: {self.opts.pti_lr:.4f};"
                )
            )

        images, result_latent = decoder([embedding_latent], input_is_latent=True, randomize_noise=False)

        return images, result_latent, embedding_images
