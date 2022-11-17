import os
import sys

sys.path.append('.')
sys.path.append('..')


from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import InversionDataset
from inference import EncoderInference, OptimizerInference, PTIInference
from utils.common import tensor2im
from options.test_options import TestOptions
import torchvision.transforms as transforms


def main():
    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    if opts.inverse_mode == 'optim':
        inversion = OptimizerInference(opts)
    elif opts.inverse_mode == 'encoder':
        inversion = EncoderInference(opts)
    elif opts.inverse_mode == 'pti':
        inversion = PTIInference(opts)
    else:
        raise Exception(f'{opts.inverese_mode} is not a valid mode. We now support "optim" and "encoder".')

    if opts.output_dir is None:
        opts.output_dir = os.path.join(opts.exp_dir, 'inference_results')
        os.makedirs(opts.output_dir, exist_ok=True)
        os.makedirs(os.path.join(opts.output_dir, 'inversion'), exist_ok=True)
        os.makedirs(os.path.join(opts.output_dir, 'embedding'), exist_ok=True)

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

    # with torch.no_grad():
    for input_batch in (dataloader):
        images_resize, img_paths, images = input_batch
        images_resize, images = images_resize.cuda(), images.cuda()
        inv_images, codes, emb_images = inversion.inverse(images, images_resize, img_paths)
        H, W = inv_images.shape[2:]

        for path, inv_img, code, emb_img in zip(img_paths, inv_images, codes, emb_images):
            basename = os.path.basename(path).split('.')[0]
            if opts.save_code:
                torch.save(code, os.path.join(opts.output_dir, 'code', f'{basename}.pt'))
            if opts.output_resolution is not None and ((H, W) != opts.output_resolution):
                inv_img = torch.nn.functional.resize(inv_img, opts.output_resolution)
            inv_result = tensor2im(inv_img)
            # Image.fromarray(np.array(inv_result)).save(os.path.join(opts.output_dir, 'inversion', f'{basename}.jpg'))
            inv_result.save(os.path.join(opts.output_dir, 'inversion', f'{basename}.jpg'))
            emb_result = tensor2im(emb_img)
            emb_result.save(os.path.join(opts.output_dir, 'embedding', f'{basename}.jpg'))


if __name__ == '__main__':
    main()
