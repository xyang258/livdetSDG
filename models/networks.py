import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F
from .torch_utils import to_device
from .torch_utils import Indexflow

from .models import register_model
import functools
import numpy as np
from .rand_conv import RandConvModule, RandConv2d, MultiScaleRandConv2d, data_whiten_layer
from .functions import ReverseLayerF


def get_random_module(net, opt, data_mean, data_std):
    return RandConvModule(net,
                          in_channels=opt.input_nc,
                          out_channels=opt.output_nc,
                          kernel_size=[1,3,5,7],
                          mixing=True,
                          identity_prob=0.0,
                          rand_bias=False,
                          distribution='kaiming_normal',
                          data_mean=data_mean,
                          data_std=data_std,
                          clamp_output='norm',
                          Ualpha=1.
                          )


def get_random_module_eval(net, opt, data_mean, data_std):
    return RandConvModule(net,
                          in_channels=opt.input_nc,
                          out_channels=opt.output_nc,
                          kernel_size=[1,3,5,7],
                          mixing=True,
                          identity_prob=0.0,
                          rand_bias=False,
                          distribution='kaiming_normal',
                          data_mean=data_mean,
                          data_std=data_std,
                          clamp_output='norm',
                          Ualpha=0.5,
                          Lalpha=0.
                          )


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# lesion detection net

__all__ = ['UNet2D', 'unet2d']

def passthrough(x, **kwargs):
    return x

def convAct(nchan):
    return nn.ELU(inplace=True)

def convAct_pGRL(nchan):
    return nn.LeakyReLU(0.2, True)

class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class ConvBN(nn.Module):
    def __init__(self, nchan, inChans=None, norm_layer = get_norm_layer(norm_type='instance')):
        super(ConvBN, self).__init__()
        if inChans is None:
            inChans = nchan
        self.act = convAct(nchan)
        self.conv = nn.Conv2d(inChans, nchan, kernel_size=3, padding=1)
        self.bn = ContBatchNorm2d(nchan)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out

def _make_nConv(nchan, depth, norm_layer = get_norm_layer(norm_type='instance')):
    layers = []
    if depth >=0:
        for _ in range(depth):
            layers.append(ConvBN(nchan))
        return nn.Sequential(*layers)
    else:
        return passthrough

class InputTransition(nn.Module):
    def __init__(self,inputChans, outChans, norm_layer = get_norm_layer(norm_type='instance')):
        self.outChans = outChans
        self.inputChans = inputChans
        super(InputTransition, self).__init__()
        self.conv = nn.Conv2d(inputChans, outChans, kernel_size=3, padding=1)
        self.bn = norm_layer(outChans)
        self.relu = convAct_pGRL(outChans)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = self.relu(out)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False, norm_layer = get_norm_layer(norm_type='instance')):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1, stride=2)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct_pGRL(outChans)
        self.relu2 = convAct_pGRL(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out+down)
        return out

def match_tensor(out, refer_shape):
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0))
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]

    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row))
    else:
        crop_row = row - skiprow
        left_crop_row  = crop_row // 2

        right_row = left_crop_row + skiprow

        out = out[:,:,left_crop_row:right_row, :]

    return out


class UpConcat(nn.Module):
    def __init__(self, inChans, hidChans, outChans, nConvs, dropout=False,stride=2, norm_layer = get_norm_layer(norm_type='instance')):
        super(UpConcat, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, hidChans, kernel_size=3,
                                          padding=1, stride=stride, output_padding=1)
        self.bn1 = ContBatchNorm2d(hidChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = convAct(hidChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = match_tensor(out, skipxdo.size()[2:])

        xcat = torch.cat([out, skipxdo], 1)
        out  = self.ops(xcat)
        out  = self.relu2(out + xcat)
        return out

class UpConv(nn.Module):
    def __init__(self, inChans, outChans, nConvs,dropout=False, stride = 2, norm_layer = get_norm_layer(norm_type='instance')):
        super(UpConv, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=3,
                                          padding=1, stride = stride, output_padding=1)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

    def forward(self, x, dest_size):
        out = self.do1(x)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = match_tensor(out, dest_size)
        return out



class DomainClassifier(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(DomainClassifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4 #$3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = 8
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class OutputTransition(nn.Module):
    def __init__(self, inChans,outChans=1,hidChans=2, norm_layer = get_norm_layer(norm_type='instance')):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, hidChans, kernel_size=5, padding=2)
        self.bn1   = ContBatchNorm2d(hidChans)
        self.relu1 = convAct( hidChans)
        self.conv2 = nn.Conv2d(hidChans, outChans, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out

class ResdualBlock_KiNet(nn.Module):
    def __init__(self, inChans, outChans, nConvs=1, dropout=False, norm_layer = get_norm_layer(norm_type='instance')):
        super(ResdualBlock_KiNet, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=1)
        self.bn1 = ContBatchNorm2d(outChans) ##
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out+down)
        return out

class UNet2D(nn.Module):
    def __init__(self, num_class=1):
        super(UNet2D, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.in_tr_100     = InputTransition(1, 32)
        self.down_tr32_50  = DownTransition(32, 64, 1)
        self.down_tr64_25  = DownTransition(64, 128, 2)
        self.down_tr128_12 = DownTransition(128, 256, 2,dropout=True)
        self.down_tr256_6  = DownTransition(256, 256, 2,dropout=True)

        self.up_tr256_12   = UpConcat(256, 256, 512, 2,  dropout=True)
        self.up_tr128_25   = UpConcat(512, 128, 256, 2, dropout=True)
        self.up_tr64_50    = UpConcat(256, 64, 128, 1)
        self.up_tr32_100   = UpConcat(128, 32, 64, 1)

        self.up_12_100   = UpConv(512, 64, 2, stride = 8)
        self.up_25_100   = UpConv(256, 64, 2, stride = 4)
        self.out_tr      = OutputTransition(64*3, num_class, 32)
        
        self.domain_c = DomainClassifier(input_nc=256, ndf=64, n_layers=3, norm_layer=get_norm_layer(norm_type='instance'))

    def forward(self, x, alpha):
        x = to_device(x,self.device_id)
        out16 = self.in_tr_100(x)
        out32 = self.down_tr32_50(out16)
        out64 = self.down_tr64_25(out32)
        out128 = self.down_tr128_12(out64)
        out256 = self.down_tr256_6(out128)

        out_up_12 = self.up_tr256_12(out256, out128)
        out_up_25 = self.up_tr128_25(out_up_12, out64)
        out_up_50 = self.up_tr64_50(out_up_25, out32)
        out_up_50_100 = self.up_tr32_100(out_up_50, out16)

        out_up_12_100 = self.up_12_100(out_up_12, x.size()[2:])
        out_up_25_100 = self.up_25_100(out_up_25, x.size()[2:])
        out_cat = torch.cat([out_up_50_100,out_up_12_100, out_up_25_100 ], 1)
        out = self.out_tr(out_cat)
        
        out256_RL = ReverseLayerF.apply(out256, alpha)
        out_domain = self.domain_c(out256_RL)

        return out, out_domain, out_cat

    def predict(self, batch_data, batch_size=None):
        self.eval()
        total_num = batch_data.shape[0]
        if batch_size is None or batch_size >= total_num:
            x = to_device(batch_data, self.device_id, False).float()
            det, _, _ = self.forward(x, alpha = 0.5)
            return det.cpu().data.numpy()
        else:
            results = []
            for ind in Indexflow(total_num, batch_size, False):
                data = batch_data[ind]
                data = to_device(data, self.device_id, False).float()
                det, _, _ = self.forward(data)
                results.append(det.cpu().data.numpy())
            return np.concatenate(results,axis=0)

@register_model('unet2d')
def unet2d(num_cls=1, pretrained=True, finetune=False, out_map=True, **kwargs):
    model = UNet2D(num_class=num_cls)
    return model
