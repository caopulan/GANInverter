import os
from tqdm import tqdm
import numpy as np
from criteria.lpips.lpips import LPIPS
import math
from models.stylegan2.model import Generator
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.train_utils import load_train_checkpoint
from inference.inference import BaseInference


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


    def inverse(self, images, images_resize, image_name, emb_codes, emb_images, mask_result, delta):
        refine_info = dict()
        # initialize decoder and regularization decoder
        decoder = Generator(self.opts.resolution, 512, 8).to(self.device)
        decoder.train()
        if self.checkpoint is not None:
            decoder.load_state_dict(self.checkpoint['decoder'], strict=False)
        else:
            decoder_checkpoint = torch.load(self.opts.stylegan_weights, map_location='cpu')
            decoder.load_state_dict(decoder_checkpoint['g_ema'])

        # initialize modulated feature and optimizer
        feature_idx = self.opts.dhr_feature_idx # 11
        res = [4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024][feature_idx]
        with torch.no_grad():
            gen_images, _, offset = decoder([emb_codes], feature_idx=feature_idx, mask=None, input_is_latent=True,
                                            randomize_noise=False, return_featuremap=True)
        offset = torch.nn.Parameter(offset.detach()).cuda()
        optimizer_f = optim.Adam([offset], lr=self.opts.dhr_feature_lr)
        optimizer = optim.Adam(decoder.named_parameters(), lr=self.opts.dhr_weight_lr)

        # domain-specific segmentation
        bg_score = 0.6
        fg_score = 0.7
        top_score = 0.8
        score_thr = [bg_score, bg_score, fg_score, bg_score] + [top_score] * 10 + [bg_score, bg_score, bg_score, bg_score, bg_score]
        # load segmentation if exist
        if os.path.exists(f'{self.opts.output_dir}/mask_refine_pt/{os.path.basename(image_name[0])[:-4]}.pt'):
            m = torch.load(f'{self.opts.output_dir}/mask_refine_pt/{os.path.basename(image_name[0])[:-4]}.pt')
            m = m.cuda()
        else:
            mask = np.load(image_name[0].replace('/home/ssd1/Database/celeba_hq_test_complex/', 'data/CelebA-HQ/test/').replace('test', 'test-parsing/parsing_result').replace('jpg', 'npy'))
            parsing_result = torch.Tensor(mask).cuda()
            mask_bg = mask[[0, 3, -1, -2, -3]].sum(axis=0)
            mask_bg = torch.tensor(mask_bg).cuda()
            mask_bg = (mask_bg < 0.5).float()
            mask = mask_bg[None, None]
            mask_ori = mask.clone()

            # Genearte superpixel result
            mask_sp = np.load(image_name[0].replace('/home/ssd1/Database/celeba_hq_test_complex/', 'data/CelebA-HQ/test/').replace('test', 'test-sp').replace('jpg', 'npy'))
            result = torch.zeros((1024, 1024))
            for sp_i in range(1, 1 + int(mask_sp.max())):
                ds = []
                mask = torch.Tensor(mask_sp == sp_i).float()[None, None].cuda()
                for idx, d in enumerate(delta):
                    shape = d.shape[2]
                    m = torch.nn.functional.interpolate(mask, (shape, shape))
                    ds.append((d * m).sum() / m.sum())
                parsing_idx = (mask[0] * parsing_result).sum(dim=[-1, -2]).argmax().item()
                result[mask_sp == sp_i] = (sum(ds) < score_thr[parsing_idx]).float()
            mask_result = result.clone().numpy()

            m = torch.Tensor(mask_result)[None, None].cuda() * mask_ori

            refine_info['mask_refine_pt'] = m.clone()
            refine_info['mask_refine'] = m[0, 0].cpu().numpy() * 255.
            refine_info['mask_superpixel'] = mask_result * 255.
            refine_info['mask_parsing'] = mask_ori[0, 0].cpu().numpy() * 255.
            # torch.save(m, f'{self.opts.output_dir}/mask_refine_pt/{os.path.basename(image_name[0])[:-4]}.pt')
            # cv2.imwrite(f'{self.opts.output_dir}/mask_refine/{os.path.basename(image_name[0])}', m[0, 0].cpu().numpy() * 255.)
            # cv2.imwrite(f'{self.opts.output_dir}/mask_superpixel/{os.path.basename(image_name[0])}', mask_result * 255.)
            # cv2.imwrite(f'{self.opts.output_dir}/mask_parsing/{os.path.basename(image_name[0])}', mask_ori[0, 0].cpu().numpy() * 255.)

        mask = torch.nn.AdaptiveAvgPool2d((res, res))(m)
        mask_ori = m.clone()

        pbar = tqdm(range(self.opts.dhr_feature_step))

        # load weight and feature if exist.
        if os.path.exists(f'{self.opts.output_dir}/weight/{os.path.basename(image_name[0])[:-4]}.pt') and os.path.exists(f'{self.opts.output_dir}/feature/{os.path.basename(image_name[0])[:-4]}.pt'):
            weight = torch.load(f'{self.opts.output_dir}/weight/{os.path.basename(image_name[0])[:-4]}.pt', map_location='cpu')
            decoder.load_state_dict(weight)
            offset = torch.load(f'{self.opts.output_dir}/feature/{os.path.basename(image_name[0])[:-4]}.pt')
        else:
            for i in pbar:
                gen_images, _ = decoder([emb_codes], feature_idx=feature_idx, offset=offset, mask=mask, input_is_latent=True, randomize_noise=False)

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
                loss_mse_bg = ((F.mse_loss(gen_images, images, reduction='none') * (1 - mask_ori)).sum() / (1 - mask_ori).sum())
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

                pbar.set_description(
                    (
                        f"loss: {loss.item():.4f}; mse: {loss_mse.item():.4f}"
                    )
                )

            refine_info['weight'] = decoder.state_dict()
            refine_info['feature'] = offset.clone()
            refine_info['mask'] = mask.clone()

            # Visualize the effects of feature modulation
            # images_weight, _ = decoder([emb_codes], feature_idx=feature_idx, offset=None, mask=None,
            #                         input_is_latent=True, randomize_noise=False)
            # refine_info['inversion_weight'] = images_weight

        images, result_latent = decoder([emb_codes], feature_idx=feature_idx, offset=offset, mask=mask, input_is_latent=True, randomize_noise=False)
        return images, emb_codes, refine_info

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

