import os
import sys

sys.path.append('.')
sys.path.append('..')

import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import InversionDataset
from inference import TwoStageInference
from utils.common import tensor2im
from options.test_options import TestOptions
import torchvision.transforms as transforms


def save_intermediate(info_dict, output_dir, basename, keys):
    if info_dict is None:
        return None
    for k, v in info_dict.items():
        if keys is not None and k not in k:
            continue
        os.makedirs(os.path.join(output_dir, k), exist_ok=True)
        if isinstance(v, torch.Tensor):
            # image tensor
            if v.dim() == 4 and v.shape[0] == 1 and v.shape[1] == 3:
                img = tensor2im(img[0])
                img.save(os.path.join(output_dir, 'inversion', f'{basename}.jpg'))
            elif v.dim() == 3 and v.shape[0] == 3:
                img = tensor2im(img)
                img.save(os.path.join(output_dir, 'inversion', f'{basename}.jpg'))
            else:  # tensor but not image
                torch.save(v, os.path.join(output_dir, k, f'{basename}.pt'))
        # model weight
        elif (isinstance(v, dict) and isinstance(list(v.values())[0], torch.Tensor)):
            torch.save(v, os.path.join(output_dir, k, f'{basename}.pt'))
        # numpy array
        elif isinstance(v, np.ndarray):
            np.save(v, os.path.join(output_dir, k, f'{basename}.npy'))
        else:
            pass


def main():
    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    inversion = TwoStageInference(opts)

    if opts.output_dir is None:
        opts.output_dir = os.path.join(opts.exp_dir, 'inference_results')
    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, 'inversion'), exist_ok=True)

    if opts.save_code:
        os.makedirs(os.path.join(opts.output_dir, 'code'), exist_ok=True)

    if opts.output_resolution is not None and len(opts.output_resolution) == 1:
        opts.output_resolution = (opts.output_resolution, opts.output_resolution)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_no_resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    if os.path.isdir(opts.test_dataset_path):
        dataset = InversionDataset(root=opts.test_dataset_path, transform=transform,
                                   transform_no_resize=transform_no_resize)
        dataloader = DataLoader(dataset,
                                batch_size=opts.test_batch_size,
                                shuffle=False,
                                num_workers=int(opts.test_workers),
                                drop_last=False)
    else:
        img = Image.open(opts.test_dataset_path)
        img = img.convert('RGB')
        img_aug = transform(img)
        img_aug_no_resize = transform_no_resize(img)
        dataloader = [(img_aug[None], [opts.test_dataset_path], img_aug_no_resize[None])]

    for input_batch in tqdm.tqdm(dataloader):
        images_resize, img_paths, images = input_batch
        images_resize, images = images_resize.cuda(), images.cuda()
        emb_images, emb_codes, emb_info, refine_images, refine_codes, refine_info = \
            inversion.inverse(images, images_resize, img_paths)

        H, W = emb_images.shape[2:]
        if refine_images is not None:
            images, codes = refine_images, refine_codes
        else:
            images, codes = emb_images, emb_codes

        emb_info = [None] * len(img_paths) if emb_info is None else emb_info
        refine_info = [None] * len(img_paths) if refine_info is None else refine_info

        for path, inv_img, code, e_info, r_info in zip(img_paths, images, codes, emb_info, refine_info):
            basename = os.path.basename(path).split('.')[0]
            if opts.save_code:
                torch.save(code, os.path.join(opts.output_dir, 'code', f'{basename}.pt'))
            if opts.output_resolution is not None and ((H, W) != opts.output_resolution):
                inv_img = torch.nn.functional.resize(inv_img, opts.output_resolution)
            inv_result = tensor2im(inv_img)
            inv_result.save(os.path.join(opts.output_dir, 'inversion', f'{basename}.jpg'))

            # save intermediate info
            if opts.save_intermediate:
                save_intermediate(e_info, opts.output_dir, basename, opts.save_keys)
                save_intermediate(r_info, opts.output_dir, basename, opts.save_keys)


if __name__ == '__main__':
    main()
