import os
import sys

sys.path.append('.')
sys.path.append('..')

import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from datasets.inference_dataset import InversionDataset
from inference import TwoStageInference
from options.test_options import TestOptions
import torchvision.transforms as transforms
from criteria.lpips.lpips import LPIPS
import insightface


def main():

    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    inversion = TwoStageInference(opts)
    lpips_cri = LPIPS(net_type='alex').cuda().eval()

    float2uint2float = lambda x: (((x + 1) / 2 * 255.).clamp(min=0, max=255).to(torch.uint8).float().div(255.) - 0.5) / 0.5

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

    lpips, count = 0, 0.
    mse, psnr, id = torch.zeros([0]).cuda(), torch.zeros([0]).cuda(), torch.zeros([0]).cuda()
    for input_batch in tqdm.tqdm(dataloader):
        # Inversion
        images_resize, img_paths, images = input_batch
        images_resize, images = images_resize.cuda(), images.cuda()
        count += len(img_paths)
        emb_images, emb_codes, emb_info, refine_images, refine_codes, refine_info = \
            inversion.inverse(images, images_resize, img_paths)
        H, W = emb_images.shape[2:]
        if refine_images is not None:
            images_inv, codes = refine_images, refine_codes
        else:
            images_inv, codes = emb_images, emb_codes

        # from utils.common import tensor2im
        # for img, path in zip(images_inv, img_paths):
        #     tensor2im(img).save(f'samples/e4e/{os.path.basename(path)[:-4]}.png')

        # Evaluation
        images_inv = float2uint2float(images_inv)
        images_inv_resize = transforms.Resize((256, 256), antialias=True)(images_inv)
        batch_mse, batch_psnr = calculate_mse_and_psnr(images_inv_resize, images_resize)
        batch_lpips = lpips_cri(images_inv_resize, images_resize)

        mse = torch.cat([mse, batch_mse])
        psnr = torch.cat([psnr, batch_psnr])
        lpips += len(img_paths) * batch_lpips.item()

    print('MSE ', mse.mean().item())
    print('PSNR:', psnr.mean().item())
    print('LPIPS:', lpips / count)


def calculate_mse_and_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(2 / mse)
    return mse, psnr


if __name__ == '__main__':
    main()
