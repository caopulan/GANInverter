import torch
from torch import nn
from models.encoders import psp_encoders
from configs.paths_config import model_paths
from loguru import logger
from torch.nn.parallel import DistributedDataParallel


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_ = dict()
    for k, v in d.items():
        if k.startswith('module.'):
            d_[k[7:]] = v
        else:
            d_[k] = v
    d_filt = {k[len(name) + 1:]: v for k, v in d_.items() if k[:len(name)] == name}

    return d_filt


class Encoder(nn.Module):

    def __init__(self, opts, checkpoint=None, latent_avg=None, device="cuda"):
        super(Encoder, self).__init__()
        self.opts = opts

        # Define architecture
        self.encoder = self.set_encoder().to(device)
        self.log_parameters()
        self.load_weights(checkpoint, latent_avg)

        if 'dist' in opts and opts.dist:
            self.encoder = DistributedDataParallel(self.encoder, device_ids=[torch.cuda.current_device()],
                                                   find_unused_parameters=True)
            self.dist = True
        else:
            self.dist = False

    def log_parameters(self):
        parameter = 0
        for v in list(self.encoder.parameters()):
            parameter += v.view(-1).shape[0]
        logger.info(f'Encoder parameters: {parameter/1e6}M.')
        self.opts.parameters = parameter

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(self.opts.layers, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(self.opts.layers, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(self.opts.layers, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'ProgressiveBackboneEncoder':
            encoder = psp_encoders.ProgressiveBackboneEncoder(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    @staticmethod
    def check_module(ckpt, module_name):
        for key in ckpt['state_dict'].keys():
            if key.startswith(f'{module_name}.'):
                return True
        return False

    def set_progressive_stage(self, stage):
        if self.dist:
            self.encoder.module.progressive_stage = stage
        else:
            self.encoder.progressive_stage = stage

    def load_weights(self, checkpoint, latent_avg):
        if checkpoint is not None:
            logger.info('Loading Encoder from checkpoint: {}'.format(self.opts.checkpoint_path))
            encoder_load_status = self.encoder.load_state_dict(get_keys(checkpoint['encoder'], 'encoder'), strict=False)
            latent_avg = checkpoint['encoder']['latent_avg']
            logger.info(f"encoder loading results: {encoder_load_status}")
        else:
            if self.opts.layers == 50:
                logger.info('Loading encoders weights from irse50!')
                encoder_ckpt = torch.load(model_paths['ir_se50'], map_location='cpu')
                self.encoder.load_state_dict(encoder_ckpt, strict=False)
            else:
                logger.warning("Randomly initialize the Encoder!")

        self.register_buffer("latent_avg", latent_avg)

    def forward(self, x):
        codes = self.encoder(x)

        # normalize with respect to the center of an average latent codes
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w or codes.dim() == 2:
                codes = codes + self.latent_avg[0].repeat(codes.shape[0], 1)
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        return codes
