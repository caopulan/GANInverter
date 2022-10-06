from torchvision.transforms import transforms

from configs.paths_config import model_paths

import torch
from torch import autograd
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from models.encoders import psp_encoders
from utils import train_utils
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from utils.ranger import Ranger
from training.encoder_trainer import EncoderTrainer
from loguru import logger


class HFGIEncoderTrainer(EncoderTrainer):
    def __init__(self, opts, prev_train_checkpoint=None):
        super(HFGIEncoderTrainer, self).__init__(opts)

        # HFGI special architectrue
        self.residue = psp_encoders.ResidualEncoder().to(self.device)   # Ec
        self.grid_transform = transforms.RandomPerspective(distortion_scale=opts.distortion_scale, p=opts.aug_rate)
        self.grid_align = psp_encoders.ResidualAligner().to(self.device)    # ADA

        # Load weights if needed
        self.load_weights()

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.latent_avg is None:
            self.latent_avg = self.decoder.mean_latent(int(1e5))[0].detach()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        if self.opts.use_wandb and opts.rank == 0:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts, 'hfgi')

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading hfgi from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            if self.opts.new_checkpont:
                encoder_load_status = self.encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
                print("encoder loading results: ", encoder_load_status)
                decoder_load_status = self.decoder.load_state_dict(ckpt['decoder_state_dict'], strict=False)
                print("decoder loading results: ", decoder_load_status)
                residue_load_status = self.residue.load_state_dict(ckpt['residue'], strict=True)
                print("residue loading results: ", residue_load_status)
                ada_load_status = self.grid_align.load_state_dict(ckpt['grid_align'], strict=True)
                print("ada loading results: ", ada_load_status)
            else:
                encoder_load_status = self.encoder.load_state_dict(self.get_keys(ckpt, 'encoder'), strict=False)
                print("encoder loading results: ", encoder_load_status)
                decoder_load_status = self.decoder.load_state_dict(self.get_keys(ckpt, 'decoder'), strict=False)
                print("decoder loading results: ", decoder_load_status)
                residue_load_status = self.residue.load_state_dict(self.get_keys(ckpt, 'residue'), strict=True)
                print("residue loading results: ", residue_load_status)
                ada_load_status = self.grid_align.load_state_dict(self.get_keys(ckpt, 'grid_align'), strict=True)
                print("ada loading results: ", ada_load_status)
            self.load_latent_avg(ckpt)
        else:
            if self.opts.layers == 50:
                print('Loading encoders weights from irse50!')
                encoder_ckpt = torch.load(model_paths['ir_se50'], map_location='cpu')
                # if input to encoder is not an RGB image, do not load the input layer weights
                self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights, map_location='cpu')
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            if self.opts.learn_in_w:
                self.load_latent_avg(ckpt, repeat=1)
            else:
                self.load_latent_avg(ckpt, repeat=self.opts.n_styles)

    def configure_optimizers(self):
        params = list(self.residue.parameters())
        params += list(self.grid_align.parameters())
        if self.opts.train_decoder:
            params += list(self.decoder.parameters())
        else:
            requires_grad(self.decoder, False)
            requires_grad(self.encoder, False)
        if self.opts.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def inverse(self, x, resize=True, input_code=False, return_latents=False, return_only_latent=False, return_aux=False):
        """if forward == 'decoder':
            return self.decoder"""
        codes = self.encoder(x)

        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w or codes.dim() == 2:
                codes = codes + self.latent_avg[0].repeat(codes.shape[0], 1)
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        if return_only_latent:
            return codes

        images, result_latent = self.decoder([codes], input_is_latent=True, randomize_noise=True, return_latents=True)

        imgs_ = torch.nn.functional.interpolate(torch.clamp(images, -1., 1.), size=(256, 256), mode='bilinear')
        res_gt = (x - imgs_).detach()
        res_unaligned = self.grid_transform(res_gt).detach()

        res_aligned = self.grid_align(torch.cat((res_unaligned, imgs_), 1))
        res = res_aligned.to(self.opts.device)

        delta = res - res_gt
        conditions = self.residue(res)
        if conditions is not None:
            images, result_latent = self.decoder([codes],
                                                 input_is_latent=True,
                                                 randomize_noise=True,
                                                 return_latents=True,
                                                 conditions=conditions)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent, delta
        else:
            return images, delta

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.face_pool.train()
        self.residue.train()
        self.grid_align.train()

        syn_iters = iter(self.synthesis_dataloader())
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                # self.optimizer.zero_grad()
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latent, delta = self.inverse(x, return_latents=True)

                if self.opts.w_loss != 0:
                    w_syn, y_syn = syn_iters.__next__()
                    if not self.opts.learn_in_w:
                        w_syn = w_syn.unsqueeze(dim=1).repeat([1, latent.shape[1], 1])
                    latent_syn = self.inverse(y_syn, return_only_latent=True)
                else:
                    w_syn, latent_syn = None, None

                loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, w_syn, latent_syn, delta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_dict['w_abs_mean'] = latent.abs().mean().item()
                loss_dict['w_var'] = latent.var().item()
                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                if self.opts.use_wandb and batch_idx == 0:
                    self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="train", step=self.global_step, opts=self.opts)

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.save_checkpoint(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.save_checkpoint(val_loss_dict, is_best=False)
                    else:
                        self.save_checkpoint(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    logger.info('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.encoder.eval()
        self.decoder.eval()
        self.face_pool.eval()
        self.residue.eval()
        self.grid_align.eval()

        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch

            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latent = self.inverse(x, return_latents=True)
                loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, y, y_hat,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))

            # Log images of first batch to wandb
            if self.opts.use_wandb and batch_idx == 0 and self.opts.rank == 0:
                self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="test", step=self.global_step, opts=self.opts)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.encoder.train()
                self.decoder.train()
                self.face_pool.train()
                self.residue.train()
                self.grid_align.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.encoder.train()
        self.decoder.train()
        self.face_pool.train()
        self.residue.train()
        self.grid_align.train()
        return loss_dict

    def calc_loss(self, x, y, y_hat, latent, w_syn=None, latent_syn=None, delta=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.w_loss > 0 and w_syn is not None:
            loss_w = F.mse_loss(w_syn, latent_syn)
            loss_dict['loss_w_l2'] = float(loss_w)
            loss += loss_w * self.opts.w_loss
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop
        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent, self.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda
        if self.opts.res_lambda > 0:
            target = torch.zeros_like(delta)
            loss_res = F.l1_loss(delta, target)
            loss_dict['loss_res'] = float(loss_res)
            loss += loss_res * self.opts.res_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def __get_save_dict(self):
        save_dict = super(HFGICoach, self).__get_save_dict()

        save_dict['residue_state_dict'] = self.residue.state_dict()
        save_dict['grid_align_dict'] = self.grid_align.state_dict()

        return save_dict
