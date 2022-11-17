import os

from tqdm import tqdm

from criteria.lpips.lpips import LPIPS
import math
from models.stylegan2.model import Generator
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.train_utils import load_train_checkpoint, requires_grad


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


class OptimizerInference:

    def __init__(self, opts):
        super(OptimizerInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # resume from checkpoint
        checkpoint = load_train_checkpoint(opts)

        # initialize encoder and decoder
        self.latent_avg = None
        self.decoder = Generator(opts.resolution, 512, 8).to(self.device)
        self.decoder.eval()
        if checkpoint is not None:
            self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
        else:
            decoder_checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
            self.decoder.load_state_dict(decoder_checkpoint['g_ema'])
            self.latent_avg = decoder_checkpoint['latent_avg'].to(self.device) if checkpoint is None else None
        self.latent_std = None

        # initial loss
        self.lpips_loss = LPIPS(net_type='vgg').to(self.device).eval()

    def inverse(self, images, images_resize, image_name):
        if self.latent_std is None:
            n_mean_latent = 10000
            with torch.no_grad():
                noise_sample = torch.randn(n_mean_latent, 512, device=self.device)
                latent_out = self.decoder.style(noise_sample)
                latent_mean = latent_out.mean(0)
                if self.latent_avg is None:
                    self.latent_avg = latent_mean
                self.latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        latent_std = self.latent_std.detach().clone()
        latent_mean = self.latent_avg.detach().clone()

        noises_single = self.decoder.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(images_resize.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.unsqueeze(0).repeat(images_resize.shape[0], 1)

        if self.opts.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, self.decoder.n_latent, 1)

        latent_in.requires_grad = True
        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=self.opts.lr)

        pbar = tqdm(range(self.opts.optim_step))
        for i in pbar:
            t = i / self.opts.optim_step
            lr = get_lr(t, self.opts.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * self.opts.noise * max(0, 1 - t / self.opts.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = self.decoder([latent_n], input_is_latent=True, noise=noises)
            img_gen = F.interpolate(torch.clamp(img_gen, -1., 1.), size=(images_resize.shape[2], images_resize.shape[3]), mode='bilinear')

            p_loss = self.lpips_loss(img_gen, images_resize)
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, images_resize)
            loss = self.opts.optim_lpips_lambda * p_loss + self.opts.noise_regularize * n_loss + \
                   self.opts.optim_l2_lambda * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        images, result_latent = self.decoder([latent_in.detach().clone()], input_is_latent=True, noise=noises,
                                             return_latents=True)

        return images, result_latent
