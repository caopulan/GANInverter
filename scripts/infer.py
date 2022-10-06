import os

from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import InversionDataset
from inference import OptimizerInference
from utils.common import tensor2im
from options.test_options import TestOptions
from inference import EncoderInference
import torchvision.transforms as transforms


def main():
    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    if opts.inverse_mode == 'optim':
        inversion = OptimizerInference(opts)
    elif opts.inverse_mode == 'encoder':
        inversion = EncoderInference(opts)
    else:
        raise Exception(f'{opts.inverese_mode} is not a valid mode. We now support "optim" and "encoder".')

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
    if os.path.isdir(opts.test_dataset_path):
        dataset = InversionDataset(root=opts.test_dataset_path, transform=transform)
        dataloader = DataLoader(dataset,
                                batch_size=opts.test_batch_size,
                                shuffle=False,
                                num_workers=int(opts.test_workers),
                                drop_last=False)
    else:
        img = Image.open(opts.test_dataset_path)
        img = img.convert('RGB')
        img = transform(img)
        dataloader = ([img[None], opts.test_dataset_path])

    # with torch.no_grad():
    for input_batch in tqdm(dataloader):
        images, img_paths = input_batch
        images = images.cuda()
        inv_images, codes = inversion.inverse(images)
        H, W = inv_images.shape[2:]

        for path, inv_img, code in zip(img_paths, inv_images, codes):
            basename = os.path.basename(path).split('.')[0]
            if opts.save_code:
                torch.save(code, os.path.join(opts.output_dir, 'code', f'{basename}.pt'))
            if opts.output_resolution is not None and ((H, W) != opts.output_resolution):
                inv_img = torch.nn.functional.resize(inv_img, opts.output_resolution)
            inv_result = tensor2im(inv_img)
            Image.fromarray(np.array(inv_result)).save(os.path.join(opts.output_dir, 'inversion', f'{basename}.jpg'))


if __name__ == '__main__':
    main()
