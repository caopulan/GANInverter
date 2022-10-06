import os
import torch
from loguru import logger


def load_train_checkpoint(opts, best=False):
    if opts.auto_resume:
        if best:
            train_ckpt_path = os.path.join(opts.exp_dir, 'checkpoints/last.pt')
        else:
            train_ckpt_path = os.path.join(opts.exp_dir, 'checkpoints/best_model.pt')
        if os.path.isfile(train_ckpt_path):
            previous_train_ckpt = torch.load(train_ckpt_path, map_location='cpu')
        else:
            previous_train_ckpt = None
    else:
        train_ckpt_path = opts.checkpoint_path
        if train_ckpt_path is None:
            previous_train_ckpt = None
        else:
            previous_train_ckpt = torch.load(opts.checkpoint_path, map_location='cpu')

    previous_train_ckpt = convert_weight(previous_train_ckpt)
    if previous_train_ckpt is not None:
        opts.checkpoint_path = train_ckpt_path
    return previous_train_ckpt


def aggregate_loss_dict(agg_loss_dict):
    mean_vals = {}
    for output in agg_loss_dict:
        for key in output:
            mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print('{} has no value'.format(key))
            mean_vals[key] = 0
    return mean_vals


def get_train_progressive_stage(stages, step):
    if stages is None:
        return -1
    for i in range(len(stages) - 1):
        if stages[i] <= step < stages[i + 1]:
            return i
    return len(stages) - 1


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def convert_weight(weight):
    """Convert psp/e4e weights from original repo to GAN Inverter."""
    if weight is not None:
        if 'encoder' not in weight:
            logger.info('Resume from official weight. Converting to GAN Inverter weight.......')
            encoder_weight, decoder_weight = dict(), dict()
            for k, v in weight['state_dict'].items():
                if k.startswith('encoder.'):
                    encoder_weight[k] = v
                elif k.startswith('decoder.'):
                    decoder_weight[k[8:]] = v
            encoder_weight['latent_avg'] = weight['latent_avg']
            weight = dict(
                encoder=encoder_weight,
                decoder=decoder_weight,
            )
    return weight
