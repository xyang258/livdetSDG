# The code is based on https://github.com/wildphoton/RandConv 
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import numpy as np
import random
import collections

class RandConvModule(nn.Module):
    def __init__(self, net=None, kernel_size=3, in_channels=3, out_channels=3,
                 rand_bias=False,
                 mixing=False,
                 identity_prob=0.0, distribution='kaiming_normal',
                 data_mean=None, data_std=None, clamp_output=False, Ualpha=1., Lalpha=0.
                 ):

        super(RandConvModule, self).__init__()

        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(1, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(1, 1, 1))

        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (self.data_std is not None), "Need data mean/std to do output range adjust"
        self.register_buffer('range_up', None if not self.clamp_output else (torch.ones(1).reshape(1, 1, 1) - self.data_mean) / self.data_std)
        self.register_buffer('range_low', None if not self.clamp_output else (torch.zeros(1).reshape(1, 1, 1) - self.data_mean) / self.data_std)

        if isinstance(kernel_size, collections.Sequence) and len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        self.Ualpha = Ualpha
        self.Lalpha = Lalpha
        if mixing:
            out_channels = in_channels

        print("Add RandConv layer with kernel size {}, output channel {}".format(kernel_size, out_channels))
        self.randconv = MultiScaleRandConv2d(in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_size,
                                             stride=1, rand_bias=rand_bias,
                                             distribution=distribution,
                                             clamp_output=self.clamp_output,
                                             range_low=self.range_low,
                                             range_up=self.range_up,
                                             )


        self.mixing = mixing
        self.res_test_weight = None
        if self.mixing:
            assert in_channels == out_channels or out_channels == 1, \
                'In mixing mode, in/out channels have to be equal or out channels is 1'
            self.alpha = (random.random() + self.Lalpha) * (self.Ualpha - self.Lalpha) + self.Lalpha

        self.identity_prob = identity_prob

    def forward(self, input):

        if not (self.identity_prob > 0 and torch.rand(1) < self.identity_prob):
            output = self.randconv(input)

            if self.mixing:
                output = (self.alpha*output + (1-self.alpha)*input)

            if self.clamp_output:
                output = torch.max(torch.min(output, self.range_up), self.range_low)
        else:
            output = input

        return output

    def parameters(self, recurse=True):
        return self.randconv.parameters()

    def trainable_parameters(self, recurse=True):
        return self.randconv.trainable_parameters()

    def whiten(self, input):
        return (input - self.data_mean) / self.data_std

    def dewhiten(self, input):
        return input * self.data_std + self.data_mean

    def randomize(self):
        self.randconv.randomize()

        if self.mixing:
            self.alpha = random.random() * self.Ualpha

    def set_test_res_weight(self, w):
        self.res_test_weight = w

class RandConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rand_bias=True,
                 distribution='kaiming_normal',
                 clamp_output=None, range_up=None, range_low=None, **kwargs):

        super(RandConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=rand_bias, **kwargs)

        self.rand_bias = rand_bias
        self.distribution = distribution

        self.clamp_output = clamp_output
        self.register_buffer('range_up', None if not self.clamp_output else range_up)
        self.register_buffer('range_low', None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"


    def randomize(self):
        new_weight = torch.zeros_like(self.weight)
        with torch.no_grad():
            if self.distribution == 'kaiming_uniform':
                nn.init.kaiming_uniform_(new_weight, nonlinearity='conv2d')
            elif self.distribution == 'kaiming_normal':
                nn.init.kaiming_normal_(new_weight, nonlinearity='conv2d')
            elif self.distribution == 'kaiming_normal_clamp':
                fan = nn.init._calculate_correct_fan(new_weight, 'fan_in')
                gain = nn.init.calculate_gain('conv2d', 0)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    new_weight.normal_(0, std)
                    new_weight = new_weight.clamp(-2*std, 2*std)
            elif self.distribution == 'xavier_normal':
                nn.init.xavier_normal_(new_weight)
            else:
                raise NotImplementedError()

        self.weight = nn.Parameter(new_weight.detach())
        if self.bias is not None and self.rand_bias:
            new_bias = torch.zeros_like(self.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)
            self.bias = nn.Parameter(new_bias)

    def forward(self, input):
        output = super(RandConv2d, self).forward(input)

        if self.clamp_output == 'clamp':
            output = torch.max(torch.min(output, self.range_up), self.range_low)
        elif self.clamp_output == 'norm':
            output_low = torch.min(torch.min(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            output_up = torch.max(torch.max(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            output = (output - output_low)/(output_up-output_low)*(self.range_up-self.range_low) + self.range_low
        
        

        if output[0,0,:,:].max() - output[0,0,5,5] < output[0,0,5,5] - output[0,0,:,:].min():
            output = -output

        return output


class MultiScaleRandConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=False, range_up=None, range_low=None, **kwargs
                 ):

        super(MultiScaleRandConv2d, self).__init__()

        self.clamp_output = clamp_output
        self.register_buffer('range_up', None if not self.clamp_output else range_up)
        self.register_buffer('range_low', None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"

        self.multiscale_rand_convs = nn.ModuleDict(
            {str(kernel_size): RandConv2d(in_channels, out_channels, kernel_size, padding = kernel_size // 2,
                                          rand_bias=rand_bias, distribution=distribution,
                                          clamp_output=self.clamp_output,
                                          range_low=self.range_low, range_up=self.range_up,
                                          **kwargs) for kernel_size in kernel_sizes})

        self.scales = kernel_sizes
        self.n_scales = len(kernel_sizes)
        self.randomize()

    def randomize(self):
        self.current_scale = str(self.scales[random.randint(0, self.n_scales-1)])
        self.multiscale_rand_convs[self.current_scale].randomize()

    def forward(self, input):
        output = self.multiscale_rand_convs[self.current_scale](input)
        return output



class data_whiten_layer(nn.Module):
    def __init__(self, data_mean, data_std):
        super(data_whiten_layer, self).__init__()
        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(1, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(1, 1, 1))

    def forward(self, input):
        return (input - self.data_mean) / self.data_std
