import os
import sys
sys.path.append('.')
sys.path.append('..')

from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import InversionDataset, InversionCodeDataset
from inference import OptimizerInference
from utils.common import tensor2im
from options.test_options import TestOptions
from inference import EncoderInference
import torchvision.transforms as transforms


def main():
    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    # load edit direction
    if opts.edit_mode == 'interfacegan':
        edit_vector = torch.load(opts.edit_path, map_location='cpu').cuda()
        factor = opts.edit_factor
        save_folder = f'{os.path.basename(opts.edit_path).split(".")[0]}_{factor}'

    if opts.inverse_mode == 'optim' or opts.inverse_mode == 'code':
        inversion = OptimizerInference(opts)
    elif opts.inverse_mode == 'encoder':
        inversion = EncoderInference(opts)
    else:
        raise Exception(f'{opts.inverese_mode} is not a valid mode. We now support "optim", "encoder", "code".')

    if opts.output_dir is None:
        opts.output_dir = os.path.join(opts.exp_dir, 'edit_results')
        os.makedirs(opts.output_dir, exist_ok=True)
        os.makedirs(os.path.join(opts.output_dir, save_folder), exist_ok=True)

    if opts.output_resolution is not None and len(opts.output_resolution) == 1:
        opts.output_resolution = (opts.output_resolution, opts.output_resolution)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    if os.path.isdir(opts.test_dataset_path):
        if opts.inverse_mode == 'code':
            dataset = InversionCodeDataset(root=opts.test_dataset_path)
        else:
            dataset = InversionDataset(root=opts.test_dataset_path, transform=transform)
        dataloader = DataLoader(dataset,
                                batch_size=opts.test_batch_size,
                                shuffle=False,
                                num_workers=int(opts.test_workers),
                                drop_last=False)
    else:
        if opts.inverse_mode == 'code':
            dataloader = (torch.load(opts.test_dataset_path, map_location='cpu')[None], opts.test_dataset_path)
        else:
            img = Image.open(opts.test_dataset_path)
            img = img.convert('RGB')
            img = transform(img)
            dataloader = ([img[None], opts.test_dataset_path])

    for input_batch in tqdm(dataloader):
        if opts.inverse_mode == 'code':
            codes, paths = input_batch
            codes = codes.cuda()
        else:
            images, paths = input_batch
            images = images.cuda()
            inv_images, codes = inversion.inverse(images, images, None)

        edit_codes = codes + edit_vector[None] * factor
        with torch.no_grad():
            edit_images = inversion.generate(edit_codes)

        H, W = edit_images.shape[2:]

        for path, edit_img in zip(paths, edit_images):
            basename = os.path.basename(path).split('.')[0]
            if opts.output_resolution is not None and ((H, W) != opts.output_resolution):
                edit_img = torch.nn.functional.resize(edit_img, opts.output_resolution)
            edit_result = tensor2im(edit_img)
            Image.fromarray(np.array(edit_result)).save(os.path.join(opts.output_dir, save_folder, f'{basename}.jpg'))


if __name__ == '__main__':
    main()
