import math
import os
import matplotlib
import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.stylegan2.model import Generator

matplotlib.use('Agg')
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils.ranger import Ranger
from torch import autograd
from utils.train_utils import get_train_progressive_stage, requires_grad
from utils import common, train_utils
from utils.train_utils import load_train_checkpoint
from criteria import id_loss, w_norm, moco_loss
from configs import transforms_config
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from loguru import logger


class EncoderTrainer:
    train_dataset = None
    test_dataset = None
    train_dataloader = None
    test_dataloader = None
    optimizer = None
    discriminator_optimizer = None
    mse_loss = None
    lpips_loss = None
    id_loss = None
    w_norm_loss = None
    moco_loss = None

    def __init__(self, opts):
        self.opts = opts
        self.global_step = opts.start_step
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # resume from checkpoint
        checkpoint = load_train_checkpoint(opts)

        # initialize encoder and decoder
        latent_avg = None
        self.decoder = Generator(opts.resolution, 512, 8).to(self.device)
        self.decoder.train()
        if checkpoint is not None:
            self.load_from_train_checkpoint(checkpoint)
        else:
            decoder_checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
            self.decoder.load_state_dict(decoder_checkpoint['g_ema'])
            latent_avg = decoder_checkpoint['latent_avg']
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256)).to(self.device)
        if latent_avg is None:
            latent_avg = self.decoder.mean_latent(int(1e5))[0].detach() if checkpoint is None else None
        self.encoder = Encoder(opts, checkpoint, latent_avg, device=self.device).to(self.device)

        # initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            dims = 512
            self.discriminator = LatentCodesDiscriminator(dims, 4).to(self.device)
            if opts.dist:
                self.discriminator = DistributedDataParallel(
                    self.discriminator,
                    device_ids=[torch.cuda.current_device()])
            self.real_w_pool = LatentCodesPool(opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(opts.w_pool_size)

        # initialize sncd
        if self.opts.sncd_lambda > 0:
            self.anchor_codes = []
            with torch.no_grad():
                w = self.decoder.w_sample(int(1e5))
                s_plus = self.decoder.get_style_space(w, split=True)
                for s in s_plus:
                    self.anchor_codes.append((s / s.norm(2, dim=-1, keepdim=True)).mean(dim=0, keepdim=True))

        self.configure_loss()

        self.configure_datasets()

        # Initialize logger
        self.log_dir = os.path.join(opts.exp_dir, 'logs')
        if opts.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            if self.opts.use_wandb:
                from utils.wandb_utils import WBLogger
                self.wb_logger = WBLogger(self.opts)

        # initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        if opts.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        self.configure_optimizers(checkpoint)
        self.progressive_stage = get_train_progressive_stage(self.opts.progressive_steps, self.global_step)

    def configure_datasets(self):
        transforms_dict = transforms_config.EncodeTransforms(self.opts).get_transforms()
        self.train_dataset = train_dataset = ImagesDataset(source_root=self.opts.train_dataset_path,
                                                           target_root=self.opts.train_dataset_path,
                                                           source_transform=transforms_dict['transform_source'],
                                                           target_transform=transforms_dict['transform_gt_train'],
                                                           opts=self.opts)
        self.test_dataset = test_dataset = ImagesDataset(source_root=self.opts.test_dataset_path,
                                                         target_root=self.opts.test_dataset_path,
                                                         source_transform=transforms_dict['transform_source'],
                                                         target_transform=transforms_dict['transform_test'],
                                                         opts=self.opts)

        # set dataloader
        train_batch_size = self.opts.batch_size // self.opts.gpu_num
        test_batch_size = self.opts.test_batch_size // self.opts.gpu_num
        assert self.opts.batch_size == train_batch_size * self.opts.gpu_num, 'Train batch size is not a multiple of gpu num.'
        assert self.opts.batch_size == test_batch_size * self.opts.gpu_num, 'Test batch size is not a multiple of gpu num.'
        if self.opts.dist:
            train_sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                drop_last=True,
                seed=self.opts.seed
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                batch_size=train_batch_size,
                num_workers=int(self.opts.workers // self.opts.gpu_num),
            )
            test_sampler = DistributedSampler(
                self.test_dataset,
                shuffle=False,
                drop_last=False,
                seed=self.opts.seed
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                sampler=test_sampler,
                batch_size=test_batch_size,
                num_workers=int(self.opts.test_workers // self.opts.gpu_num)
            )
        else:
            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=int(self.opts.workers),
                                               drop_last=True)
            self.test_dataloader = DataLoader(self.test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=int(self.opts.test_workers),
                                              drop_last=True)
        if self.opts.rank == 0:
            logger.info(f"Number of train samples: {len(train_dataset)}, train_batch_size per GPU: {train_batch_size}.")
            logger.info(f"Number of test samples: {len(test_dataset)}, test_batch_size per GPU: {test_batch_size}.")

    def configure_loss(self):
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

    def configure_optimizers(self, checkpoint):
        requires_grad(self.decoder, False)
        betas = (self.opts.optim_beta1, self.opts.optim_beta2)
        if self.opts.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.opts.learning_rate,
                                         weight_decay=self.opts.weight_decay, betas=betas)
        elif self.opts.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.opts.learning_rate,
                                          weight_decay=self.opts.weight_decay, betas=betas)
        elif self.opts.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.encoder.parameters(), lr=self.opts.learning_rate,
                                        weight_decay=self.opts.weight_decay)
        else:
            optimizer = Ranger(self.encoder.parameters(), lr=self.opts.learning_rate,
                               weight_decay=self.opts.weight_decay, betas=betas)
        if checkpoint is not None:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                logger.warning('Optimizer state dict is not in checkpoint!')

        if self.opts.w_discriminator_lambda > 0:
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=self.opts.discriminator_lr)
            if checkpoint is not None:
                if 'discriminator_optimizer_state_dict' in checkpoint:
                    self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
                else:
                    logger.warning('Discriminator optimizer state dict is not in checkpoint!')
        self.optimizer = optimizer

    def inverse(self, x):
        codes = self.encoder(x)
        images, result_latent = self.decoder([codes], input_is_latent=True, randomize_noise=True, return_latents=True)
        images = self.face_pool(images)
        return images, result_latent

    def train(self):
        self.encoder.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.encoder.set_progressive_stage(self.progressive_stage)
                loss_dict = {}
                if self.is_training_discriminator():
                    loss_dict = self.train_discriminator(batch)
                self.progressive_stage = get_train_progressive_stage(self.opts.progressive_steps, self.global_step)

                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latent = self.inverse(x)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                if self.opts.use_wandb and batch_idx == 0 and self.opts.rank == 0:
                    self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="train", step=self.global_step,
                                                       opts=self.opts)

                # Validation related
                val_loss_dict = None
                if ((
                            self.global_step % self.opts.val_interval == 0) and self.global_step != 0) or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.save_checkpoint(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.save_checkpoint(val_loss_dict, is_best=False)
                        self.save_checkpoint(val_loss_dict, is_best=False, is_last=True)
                    else:
                        self.save_checkpoint(loss_dict, is_best=False)
                        self.save_checkpoint(loss_dict, is_best=False, is_last=True)

                if self.global_step == self.opts.max_steps:
                    logger.info('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.encoder.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(batch)
            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latent = self.inverse(x)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, y, y_hat, title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))

            # Log images of first batch to wandb
            if self.opts.use_wandb and batch_idx == 0 and self.opts.rank == 0:
                self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="test", step=self.global_step,
                                                   opts=self.opts)

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.encoder.train()
        return loss_dict

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = list(range(self.opts.n_styles))

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc

        if self.opts.progressive_steps and self.opts.delta_norm_lambda > 0.:  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = list(range(self.opts.n_styles))

            first_w = latent[:, 0, :]
            for i in range(1, self.progressive_stage + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            loss_dict['total_delta_loss'] = float(total_delta_loss)
            loss += self.opts.delta_norm_lambda * total_delta_loss

        if self.opts.sncd_lambda > 0:  # calculate cos loss though lambda=0
            dims_to_discriminate = list(range(self.opts.n_styles)) if not self.is_progressive_training() else \
                list(range(self.progressive_stage + 1))
            latent_s = self.decoder.get_style_space(latent, split=True)
            latent_s = [s / s.norm(2, dim=-1, keepdim=True) for s in latent_s]
            similarity = [s0 @ s1.T for s0, s1 in zip(latent_s, self.anchor_codes)]
            sncd_loss = 0
            for dim in dims_to_discriminate:
                closs = -similarity[dim].mean()
                loss_dict[f'sncd_loss_{dim}'] = float(closs)
                sncd_loss += closs
            loss_dict[f'total_sncd_loss'] = float(sncd_loss)
            loss += sncd_loss * self.opts.sncd_lambda

        if self.opts.id_lambda > 0:  # Similarity loss
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

        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent, self.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda

        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def get_dims_to_discriminate(self):
        return list(range(self.opts.n_styles))[:self.progressive_stage + 1]

    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_w, create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def train_discriminator(self, batch):
        loss_dict = {}
        x, _ = batch
        x = x.to(self.device).float()
        requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)

        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        requires_grad(self.discriminator, False)

        return loss_dict

    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _ = test_batch
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)

            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        real_w = self.decoder.get_latent(sample_z)
        fake_w = self.encoder(x)
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w

    def save_checkpoint(self, loss_dict, is_best, is_last=False):
        if self.opts.rank == 0:
            if is_best:
                save_name = 'best_model.pt'
            elif is_last:
                save_name = 'last.pt'
            else:
                save_name = 'iteration_{}.pt'.format(self.global_step)
            save_dict = self.__get_save_dict(is_last)
            checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
            torch.save(save_dict, checkpoint_path)
            with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
                if is_best:
                    f.write(
                        '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss,
                                                                           loss_dict))
                else:
                    f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def log_metrics(self, metrics_dict, prefix):
        if self.opts.use_wandb and self.opts.rank == 0:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        if self.opts.rank == 0:
            logger.info('Metrics for {}, step {}'.format(prefix, self.global_step))
            for key, value in metrics_dict.items():
                logger.info('\t{} = {}'.format(key, value))

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        display_count = min(display_count, y.shape[0])
        if self.opts.rank == 0:
            im_data = []
            for i in range(display_count):
                cur_im_data = {
                    'input_face': common.log_input_image(x[i], self.opts),
                    'target_face': common.tensor2im(y[i]),
                    'output_face': common.tensor2im(y_hat[i]),
                }
                if id_logs is not None:
                    for key in id_logs[i]:
                        cur_im_data[key] = id_logs[i][key]
                im_data.append(cur_im_data)
            self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.log_dir, name, '{:04d}.jpg'.format(step))
        if self.opts.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def load_from_train_checkpoint(self, ckpt):
        # load training status
        logger.info('Loading previous training data...')
        self.global_step = ckpt.get('global_step', -1) + 1
        self.best_val_loss = ckpt.get('best_val_loss', 0.)
        logger.info(f'Start from step: {self.global_step}')

        # load stylegan
        self.decoder.load_state_dict(ckpt['decoder'], strict=True)

        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'], strict=False)

    def __get_save_dict(self, is_last):
        save_dict = {'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict(),
                     'opts': vars(self.opts), 'global_step': self.global_step}
        if is_last:
            save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['best_val_loss'] = self.best_val_loss
        if self.opts.w_discriminator_lambda > 0:
            save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
            if is_last:
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return save_dict
