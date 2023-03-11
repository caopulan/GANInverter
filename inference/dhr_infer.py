import math
import os
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from skimage.segmentation import slic

from criteria.lpips.lpips import LPIPS
from inference.inference import BaseInference
from models.stylegan2.model import Generator
from utils.train_utils import load_train_checkpoint
from .optim_infer import OptimizerInference
import utils.facer.facer as facer


class DHRInference(BaseInference):

    def __init__(self, opts, **kwargs):
        super(DHRInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # initial loss
        self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        self.checkpoint = load_train_checkpoint(self.opts)

        # coarse inversion
        self.coarse_inv = OptimizerInference(opts)

        # parsing
        self.detector = facer.face_detector('retinaface/mobilenet', device=self.device)
        self.parser_celeb = facer.face_parser(f'farl/celebm/448', device=self.device)

    def parsing(self, img):
        img = ((img / 2 + 0.5) * 255.).to(torch.uint8)
        with torch.no_grad():
            faces = self.detector(img, threshold=0.1)
            scores = faces['scores']
            max_idx = scores.argmax()
            faces = {k: v[max_idx][None] for k, v in faces.items()}
            faces_celeb = self.parser_celeb(img, copy.deepcopy(faces))
        parsing_result = faces_celeb['seg']['logits'].softmax(dim=1)[0]#.cpu().numpy()
        return parsing_result

    def inverse(self, images, images_resize, image_name, emb_codes, emb_images, **kwargs):
        refine_info = dict()

        # initialize decoder and regularization decoder
        feature_idx = self.opts.dhr_feature_idx  # 11
        res = [4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024][feature_idx]
        decoder = Generator(self.opts.resolution, 512, 8).to(self.device)
        decoder.train()
        if self.checkpoint is not None:
            decoder.load_state_dict(self.checkpoint['decoder'], strict=False)
        else:
            decoder_checkpoint = torch.load(self.opts.stylegan_weights, map_location='cpu')
            decoder.load_state_dict(decoder_checkpoint['g_ema'])

        # domain-specific segmentation
        bg_score = 0.6  # no use
        fg_score = self.opts.dhr_theta1
        top_score = self.opts.dhr_theta2
        score_thr = [bg_score, bg_score, fg_score, bg_score] + [top_score] * 10 + [bg_score, bg_score, bg_score,
                                                                                   bg_score, bg_score]

        # Domain-Specific Segmentation
        if os.path.exists(f'{self.opts.output_dir}/mask_refine_pt/{os.path.basename(image_name[0])[:-4]}.pt'):
            # load segmentation if exist
            m = torch.load(f'{self.opts.output_dir}/mask_refine_pt/{os.path.basename(image_name[0])[:-4]}.pt')
            m = m.cuda()
        else:
            # coarse inversion
            coarse_image, _, delta = self.coarse_inv.inverse(images, images_resize, image_name, return_lpips=True)
            refine_info['coarse_inv'] = coarse_image

            # face parsing
            parsing_result = self.parsing(images)
            mask_bg = parsing_result[[0, 3, -1, -2, -3]].sum(dim=0)
            mask_parsing = (mask_bg < 0.5).float()
            mask_parsing = mask_parsing[None, None]

            # Superpixel
            superpixel = slic(cv2.imread(image_name[0]), n_segments=200, compactness=30, sigma=1)
            mask_sp = torch.zeros((self.opts.resolution, self.opts.resolution))
            for sp_i in range(1, 1 + int(superpixel.max())):
                ds = []
                mask = torch.Tensor(superpixel == sp_i).float()[None, None].cuda()
                for idx, d in enumerate(delta):
                    shape = d.shape[2]
                    m = torch.nn.functional.interpolate(mask, (shape, shape))
                    ds.append((d * m).sum() / m.sum())
                parsing_idx = (mask[0] * parsing_result).sum(dim=[-1, -2]).argmax().item()
                mask_sp[superpixel == sp_i] = (sum(ds) < score_thr[parsing_idx]).float()

            # domain-specific segmentation result
            m = mask_sp[None, None].cuda() * mask_parsing

            refine_info['mask_refine_pt'] = m.clone()
            refine_info['mask_refine'] = (m[0, 0].cpu().numpy() * 255.).astype(np.uint8)
            refine_info['mask_superpixel'] = (mask_sp.clone().numpy() * 255.).astype(np.uint8)
            refine_info['mask_parsing'] = (mask_parsing[0, 0].cpu().numpy() * 255.).astype(np.uint8)

        mask = torch.nn.AdaptiveAvgPool2d((res, res))(m)
        mask_ori = m.clone()

        # Hybrid Refinement Modulation
        if os.path.exists(
                f'{self.opts.output_dir}/weight/{os.path.basename(image_name[0])[:-4]}.pt') and os.path.exists(
                f'{self.opts.output_dir}/feature/{os.path.basename(image_name[0])[:-4]}.pt'):
            # load weight and feature if exist.
            weight = torch.load(f'{self.opts.output_dir}/weight/{os.path.basename(image_name[0])[:-4]}.pt',
                                map_location='cpu')
            decoder.load_state_dict(weight)
            offset = torch.load(f'{self.opts.output_dir}/feature/{os.path.basename(image_name[0])[:-4]}.pt')
        else:
            # initialize modulated feature and optimizer
            with torch.no_grad():
                gen_images, _, offset = decoder([emb_codes], feature_idx=feature_idx, mask=None, input_is_latent=True,
                                                randomize_noise=False, return_featuremap=True)
            offset = torch.nn.Parameter(offset.detach()).cuda()
            optimizer_f = optim.Adam([offset], lr=self.opts.dhr_feature_lr)
            optimizer = optim.Adam(decoder.parameters(), lr=self.opts.dhr_weight_lr)

            for i in range(self.opts.dhr_feature_step):
                gen_images, _ = decoder([emb_codes], feature_idx=feature_idx, offset=offset, mask=mask,
                                        input_is_latent=True, randomize_noise=False)

                # calculate loss
                loss_lpips = self.lpips_loss(gen_images, images, keep_res=True)
                loss_mse = (F.mse_loss(gen_images, images, reduction='none')).mean()

                lpips_face, lpips_bg = [], []
                for idx, lpips in enumerate(loss_lpips):
                    shape = lpips.shape[2]
                    m = torch.nn.functional.interpolate(mask, (shape, shape))
                    lpips_face.append((lpips * m).sum() / m.sum())
                    lpips_bg.append((lpips * (1 - m)).sum() / (1 - m).sum())

                loss_lpips = torch.stack([l.mean() for l in loss_lpips]).sum()
                loss = self.opts.dhr_lpips_lambda * loss_lpips + self.opts.dhr_l2_lambda * loss_mse

                loss_lpips_bg = torch.stack(lpips_bg).sum()
                loss_mse_bg = ((F.mse_loss(gen_images, images, reduction='none') * (1 - mask_ori)).sum() / (
                            1 - mask_ori).sum())
                loss_bg = self.opts.dhr_lpips_lambda * loss_lpips_bg + self.opts.dhr_l2_lambda * loss_mse_bg

                optimizer_f.zero_grad()
                loss_bg.backward(retain_graph=True)
                optimizer_f.step()

                if i < self.opts.dhr_weight_step:
                    loss_lpips_face = torch.stack(lpips_face).sum()
                    loss_mse_face = (
                            (F.mse_loss(gen_images, images, reduction='none') * mask_ori).sum() / mask_ori.sum())
                    loss_face = self.opts.dhr_lpips_lambda * loss_lpips_face + self.opts.dhr_l2_lambda * loss_mse_face
                    optimizer.zero_grad()
                    loss_face.backward()
                    optimizer.step()

            refine_info['weight'] = decoder.state_dict()
            refine_info['feature'] = offset.clone()
            refine_info['mask'] = mask.clone()

        images, result_latent = decoder([emb_codes], feature_idx=feature_idx, offset=offset, mask=mask,
                                        input_is_latent=True, randomize_noise=False)
        return images, emb_codes, [refine_info]

    def edit(self, images, images_resize, image_paths, emb_codes, emb_images, emb_info, editor):
        images, codes, refine_info = self.inverse(images, images_resize, image_paths, emb_codes, emb_images, emb_info)
        decoder = Generator(self.opts.resolution, 512, 8).to(self.device)
        decoder.train()
        decoder.load_state_dict(refine_info['weight'], strict=True)
        edit_codes = editor.edit_code(codes)

        edit_images, edit_codes = decoder([edit_codes], feature_idx=self.opts.dhr_feature_idx,
                                          offset=refine_info['feature'], mask=refine_info['mask'], input_is_latent=True,
                                          randomize_noise=False)
        return images, edit_images, codes, edit_codes, refine_info
