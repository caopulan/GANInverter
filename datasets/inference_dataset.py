import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import glob
import os


class InversionDataset(Dataset):

	def __init__(self, root, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im, from_path


class InversionCodeDataset(Dataset):

	def __init__(self, root):
		self.paths = sorted(glob.glob(os.path.join(root, '*.pt')))

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		code_path = self.paths[index]
		return torch.load(code_path, map_location='cpu'), code_path