import os
import sys
sys.path.append('.')
sys.path.append('..')

from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import InversionDataset
from inference import TwoStageInference
from editing import *
from utils.common import tensor2im
from options.test_options import TestOptions
import torchvision.transforms as transforms


def main():
    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    # load edit direction
    if opts.edit_mode == 'interfacegan':
        editor = InterFaceGAN(opts)
    elif opts.edit_mode == 'ganspace':
        editor = GANSpace(opts)
    else:
        raise ValueError(f'Undefined editing mode: {opts.edit_mode}')

    save_folder = editor.save_folder
    inversion = TwoStageInference(opts)

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

    for input_batch in tqdm(dataloader):
        images_resize, img_paths, images = input_batch
        images_resize, images = images_resize.cuda(), images.cuda()

        with torch.no_grad():
            emb_images_edit, emb_codes_edit, emb_info, refine_images_edit, refine_codes_edit, refine_info \
                = inversion.edit(images, images_resize, img_paths, editor)

        edit_images = refine_images_edit if refine_images_edit is not None else emb_images_edit

        H, W = edit_images.shape[2:]
        for path, edit_img in zip(img_paths, edit_images):
            basename = os.path.basename(path).split('.')[0]
            if opts.output_resolution is not None and ((H, W) != opts.output_resolution):
                edit_img = torch.nn.functional.resize(edit_img, opts.output_resolution)
            edit_result = tensor2im(edit_img)
            edit_result.save(os.path.join(opts.output_dir, save_folder, f'{basename}.jpg'))


if __name__ == '__main__':
    main()
