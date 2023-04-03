import math
import numpy as np
import torch

from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential
from enum import Enum

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, _upsample_add
from models.stylegan2.model import EqualLinear, EqualConv2d, ScaledLeakyReLU


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.progressive_stage = 0


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)     # 3, 256, 256 -> 64, 256, 256

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x  # 128, 64, 64
            elif i == 20:
                c2 = x  # 256, 32, 32
            elif i == 23:
                c3 = x  # 512, 16, 16

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)   # EqualLinear lr_mul=1
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x


class Encoder4Editing(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.resolution, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        features = c3
        for i in range(1, min(self.progressive_stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w


class ProgressiveBackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE50 backbone with the
    progressive training scheme from e4e_modules.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', opts=None):
        super(ProgressiveBackboneEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.resolution, 2))
        self.style_count = 2 * log_size - 2
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)
        self.progressive_stage = 18

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)

        # get initial w0 from first map2style layer
        w0 = self.styles[0](x)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)

        # learn the deltas up to the current stage
        stage = self.progressive_stage
        for i in range(1, min(stage + 1, self.style_count)):
            delta_i = self.styles[i](x)
            w[:, i] += delta_i
        return w


# ADA
class ResidualAligner(Module):
    def __init__(self, opts=None):
        super(ResidualAligner, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(6, 16, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(16),
                                      PReLU(16))

        self.conv_layer2 = Sequential(*[bottleneck_IR(16, 32, 2), bottleneck_IR(32, 32, 1), bottleneck_IR(32, 32, 1)])
        self.conv_layer3 = Sequential(*[bottleneck_IR(32, 48, 2), bottleneck_IR(48, 48, 1), bottleneck_IR(48, 48, 1)])
        self.conv_layer4 = Sequential(*[bottleneck_IR(48, 64, 2), bottleneck_IR(64, 64, 1), bottleneck_IR(64, 64, 1)])

        self.dconv_layer1 = Sequential(*[bottleneck_IR(112, 64, 1), bottleneck_IR(64, 32, 1), bottleneck_IR(32, 32, 1)])
        self.dconv_layer2 = Sequential(*[bottleneck_IR(64, 32, 1), bottleneck_IR(32, 16, 1), bottleneck_IR(16, 16, 1)])
        self.dconv_layer3 = Sequential(*[bottleneck_IR(32, 16, 1), bottleneck_IR(16, 3, 1), bottleneck_IR(3, 3, 1)])

    def forward(self, x):
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)
        feat4 = self.conv_layer4(feat3)

        feat4 = torch.nn.functional.interpolate(feat4, size=(64, 64), mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat4, feat3), 1))
        dfea1 = torch.nn.functional.interpolate(dfea1, size=(128, 128), mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat((dfea1, feat2), 1))
        dfea2 = torch.nn.functional.interpolate(dfea2, size=(256, 256), mode='bilinear')
        dfea3 = self.dconv_layer3(torch.cat((dfea2, feat1), 1))

        res_aligned = dfea3

        return res_aligned


# Consultation encoder
class ResidualEncoder(Module):
    def __init__(self,  opts=None):
        super(ResidualEncoder, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))

        self.conv_layer2 = Sequential(*[bottleneck_IR(32,48,2), bottleneck_IR(48,48,1), bottleneck_IR(48,48,1)])

        self.conv_layer3 = Sequential(*[bottleneck_IR(48,64,2), bottleneck_IR(64,64,1), bottleneck_IR(64,64,1)])

        self.condition_scale3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

        self.condition_shift3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

    def get_deltas_starting_dimensions(self):
        """Get a list of the initial dimension of every delta from which it is applied"""
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = torch.nn.functional.interpolate(scale, size=(64, 64), mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = torch.nn.functional.interpolate(shift, size=(64, 64), mode='bilinear')
        conditions.append(shift.clone())
        return conditions
