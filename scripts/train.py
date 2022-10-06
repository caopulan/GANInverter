"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
import random
import numpy as np
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training import *


def main():
	opts = TrainOptions().parse()
	set_seed(opts.seed)
	setup_progressive_steps(opts)
	trainer = EncoderTrainer(opts)
	if opts.rank == 0:
		create_initial_experiment_dir(opts)
	trainer.train()


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def setup_progressive_steps(opts):
	log_size = int(math.log(opts.resolution, 2))
	num_style_layers = 2*log_size - 2
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

	assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
		"Invalid progressive training input"


def is_valid_progressive_steps(opts, num_style_layers):
	return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts):
	if not os.path.exists(opts.exp_dir):
		os.makedirs(opts.exp_dir)
	opts_dict = vars(opts)
	# pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
	main()
