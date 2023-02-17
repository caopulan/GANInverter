from tqdm import tqdm
from criteria.lpips.lpips import LPIPS
import math
from models.stylegan2.model import Generator
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.train_utils import load_train_checkpoint
from inference.inference import BaseInference


class Space_Regulizer:
    def __init__(self, opts, original_G, lpips_net):
        self.opts = opts
        self.device = 'cuda'
        self.original_G = original_G
        self.morphing_regulizer_alpha = opts.pti_regulizer_alpha
        self.lpips_loss = lpips_net
        self.w_mean = original_G.mean_latent(100000).detach()

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = self.morphing_regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code

        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G(w_code, input_is_latent=True, noise=None, randomize_noise=False)[0] for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch):
        loss = 0.0

        w_samples = self.original_G.w_sample(num_of_sampled_latents)
        w_samples = 0.5 * w_samples + 0.5 * self.w_mean
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img, _ = new_G(w_code, input_is_latent=True, noise=None, randomize_noise=False)
            with torch.no_grad():
                old_img, _ = self.original_G(w_code, input_is_latent=True, noise=None, randomize_noise=False)

            if self.opts.pti_regulizer_l2_lambda > 0:
                l2_loss_val = torch.nn.MSELoss(reduction='mean')(old_img, new_img)
                loss += l2_loss_val * self.opts.pti_regulizer_l2_lambda

            if self.opts.pti_regulizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * self.opts.pti_regulizer_lpips_lambda

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch):
        ret_val = self.ball_holder_loss_lazy(new_G, self.opts.pti_latent_ball_num_of_samples, w_batch)
        return ret_val


class PTIInference(BaseInference):

    def __init__(self, opts):
        super(PTIInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # initial loss
        self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()

        self.checkpoint = load_train_checkpoint(self.opts)
        origin_decoder = Generator(self.opts.resolution, 512, 8).to(self.device)
        origin_decoder.load_state_dict(self.checkpoint['decoder'])
        if opts.pti_use_regularization:
            self.space_regulizer = Space_Regulizer(opts, origin_decoder, self.lpips_loss)

    def inverse(self, images, images_resize, image_paths, emb_codes, emb_images, emb_info):
        # initialize decoder and regularization decoder
        decoder = Generator(self.opts.resolution, 512, 8).to(self.device)
        decoder.train()
        if self.checkpoint is not None:
            decoder.load_state_dict(self.checkpoint['decoder'], strict=True)
        else:
            decoder_checkpoint = torch.load(self.opts.stylegan_weights, map_location='cpu')
            decoder.load_state_dict(decoder_checkpoint['g_ema'])

        # initialize optimizer
        optimizer = optim.Adam(decoder.parameters(), lr=self.opts.pti_lr)

        pbar = tqdm(range(self.opts.pti_step))
        for i in pbar:
            gen_images, _ = decoder([emb_codes], input_is_latent=True, randomize_noise=False)

            # calculate loss
            loss_lpips = self.lpips_loss(gen_images, images)
            loss_mse = F.mse_loss(gen_images, images)
            loss = self.opts.pti_lpips_lambda * loss_lpips + self.opts.pti_l2_lambda * loss_mse

            # TODO: use regularization may cause some erros
            if self.opts.pti_use_regularization and i % self.opts.pti_locality_regularization_interval == 0:
                ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(decoder, emb_codes)
                loss += self.opts.pti_regulizer_lambda * ball_holder_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f"loss: {loss.item():.4f}; lr: {self.opts.pti_lr:.4f};"
                )
            )

        images, result_latent = decoder([emb_codes], input_is_latent=True, randomize_noise=False)

        pti_info = [{'generator': decoder.state_dict()}]
        return images, emb_codes, pti_info
