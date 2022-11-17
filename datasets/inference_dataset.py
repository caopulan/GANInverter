import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import glob
import os


class InversionDataset(Dataset):

	def __init__(self, root, transform=None, transform_no_resize=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.transform_no_resize = transform_no_resize

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')
		if self.transform:
			from_im_aug = self.transform(from_im)
		else:
			from_im_aug = from_im
		if self.transform_no_resize is not None:
			from_im_no_resize_aug = self.transform_no_resize(from_im)
			return from_im_aug, from_path, from_im_no_resize_aug
		else:
			return from_im_aug, from_path


class InversionCodeDataset(Dataset):

	def __init__(self, root):
		self.paths = sorted(glob.glob(os.path.join(root, '*.pt')))

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		code_path = self.paths[index]
		return torch.load(code_path, map_location='cpu'), code_path
