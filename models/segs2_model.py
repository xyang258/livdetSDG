import torch
import itertools
from .base_model import BaseModel
from . import networks

import click
from .models import models
from .models import get_LD_model
from .util import weighted_loss2D
from .torch_utils import to_device
import os
from .networks import get_random_module
import torch.nn as nn
import torch.nn.functional as F

class SegS2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--model_LD', default='unet2d', type=click.Choice(models.keys()))
            parser.add_argument('--momentum', '-m', default=0.99)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['tumor_A', 'tumor_A2', 'tumor_A3', 'tumor_A4', 'cons','domain']
        visual_names_A = ['A']
        visual_names_B = ['A2']

        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ['Seg']
        else:
            self.model_names = ['Seg']

        if self.isTrain:
            self.netSeg = get_LD_model(opt.model_LD, finetune=True)
            
            self.invariant_criterion = torch.nn.MSELoss()
            self.rand_module = get_random_module(net=None, opt=opt, data_mean=((0.5,)), data_std=((0.5,)))
            self.rand_module.to(self.device)
            
            self.domain_criterion = torch.nn.MSELoss()

        if self.isTrain:

            self.optimizer_Seg = torch.optim.SGD(itertools.chain(self.netSeg.parameters()), lr=opt.lr_LD, momentum=opt.momentum, nesterov = True, weight_decay=1e-06) #
            self.optimizers.append(self.optimizer_Seg)
            

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.A = input['A'].to(device=self.device, dtype=torch.float)
        self.A_label = input['A_label'].to(self.device)
        self.A_mask = input['A_mask'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self, alpha):
        self.Seg_A, self.domain_A, self.ft_A = self.netSeg(self.A, alpha)
        self.rand_module.randomize()
        self.Seg_A2, self.domain_A2, self.ft_A2 = self.netSeg(self.rand_module(self.A), alpha)#self.netSeg(self.rand_module(self.A2))
        self.rand_module.randomize()
        self.Seg_A3, self.domain_A3, self.ft_A3 = self.netSeg(self.rand_module(self.A), alpha)
        self.rand_module.randomize()
        self.Seg_A4, self.domain_A4, self.ft_A4 = self.netSeg(self.rand_module(self.A), alpha)
        self.rand_module.randomize()
        self.Seg_A5, self.domain_A5, self.ft_A5 = self.netSeg(self.rand_module(self.A), alpha)
        self.A_dlabel = torch.zeros(self.opt.batch_size,1,self.domain_A.shape[2],self.domain_A.shape[3]).to(self.device)
        self.Ai_dlabel = torch.ones(self.opt.batch_size,1,self.domain_A2.shape[2],self.domain_A2.shape[3]).to(self.device)

    
    def backward_Seg(self):
        p = 1
        self.loss_tumor_A = weighted_loss2D(self.A_label, self.Seg_A, annotate_mask = self.A_mask, weight=5.0)
        self.loss_tumor_A2 = weighted_loss2D(self.A_label, self.Seg_A2, annotate_mask = self.A_mask, weight=5.0)
        self.loss_tumor_A3 = weighted_loss2D(self.A_label, self.Seg_A3, annotate_mask = self.A_mask, weight=5.0)
        self.loss_tumor_A4 = weighted_loss2D(self.A_label, self.Seg_A4, annotate_mask = self.A_mask, weight=5.0)
        self.loss_tumor_A5 = weighted_loss2D(self.A_label, self.Seg_A5, annotate_mask = self.A_mask, weight=5.0)
        loss_tumor_del = max([self.loss_tumor_A2, self.loss_tumor_A3, self.loss_tumor_A4, self.loss_tumor_A5]) #$min
        

        if loss_tumor_del == self.loss_tumor_A2:
            self.loss_cons = (self.invariant_criterion(self.ft_A, self.ft_A3) + self.invariant_criterion(self.ft_A, self.ft_A4) + self.invariant_criterion(self.ft_A, self.ft_A5)) / 3
            self.loss_domain = (self.domain_criterion(self.domain_A, self.A_dlabel) + self.domain_criterion(self.domain_A5, self.Ai_dlabel) + self.domain_criterion(self.domain_A3, self.Ai_dlabel) + self.domain_criterion(self.domain_A4, self.Ai_dlabel)) / 4
            self.loss_LD = (self.loss_tumor_A + self.loss_tumor_A3 + self.loss_tumor_A4 + self.loss_tumor_A5) / 4 + self.loss_cons * self.opt.wt_c + self.loss_domain * self.opt.wt_d
        elif loss_tumor_del == self.loss_tumor_A3:
            self.loss_cons = (self.invariant_criterion(self.ft_A, self.ft_A2) + self.invariant_criterion(self.ft_A, self.ft_A5) + self.invariant_criterion(self.ft_A, self.ft_A4)) / 3
            self.loss_domain = (self.domain_criterion(self.domain_A, self.A_dlabel) + self.domain_criterion(self.domain_A2, self.Ai_dlabel) + self.domain_criterion(self.domain_A5, self.Ai_dlabel) + self.domain_criterion(self.domain_A4, self.Ai_dlabel)) / 4
            self.loss_LD = (self.loss_tumor_A + self.loss_tumor_A2 + self.loss_tumor_A4 + self.loss_tumor_A5) / 4 + self.loss_cons * self.opt.wt_c + self.loss_domain * self.opt.wt_d
        elif loss_tumor_del == self.loss_tumor_A4:
            self.loss_cons = (self.invariant_criterion(self.ft_A, self.ft_A2) + self.invariant_criterion(self.ft_A, self.ft_A3) + self.invariant_criterion(self.ft_A, self.ft_A5)) / 3
            self.loss_domain = (self.domain_criterion(self.domain_A, self.A_dlabel) + self.domain_criterion(self.domain_A2, self.Ai_dlabel) + self.domain_criterion(self.domain_A3, self.Ai_dlabel) + self.domain_criterion(self.domain_A5, self.Ai_dlabel)) / 4
            self.loss_LD = (self.loss_tumor_A + self.loss_tumor_A2 + self.loss_tumor_A3 + self.loss_tumor_A5) / 4 + self.loss_cons * self.opt.wt_c + self.loss_domain * self.opt.wt_d
        else:
            self.loss_cons = (self.invariant_criterion(self.ft_A, self.ft_A2) + self.invariant_criterion(self.ft_A, self.ft_A3) + self.invariant_criterion(self.ft_A, self.ft_A4)) / 3
            self.loss_domain = (self.domain_criterion(self.domain_A, self.A_dlabel) + self.domain_criterion(self.domain_A2, self.Ai_dlabel) + self.domain_criterion(self.domain_A3, self.Ai_dlabel) + self.domain_criterion(self.domain_A4, self.Ai_dlabel)) / 4
            self.loss_LD = (self.loss_tumor_A + self.loss_tumor_A2 + self.loss_tumor_A3 + self.loss_tumor_A4) / 4 + self.loss_cons * self.opt.wt_c + self.loss_domain * self.opt.wt_d
        
        self.loss_LD.backward()
        


    def valid_Seg(self, valid_Bs, valid_B_labels, valid_B_masks):
        running_valid_loss = 0.0
        for i in range(len(valid_Bs)):
            self.rand_module.randomize()
            Seg_valid_1 = self.netSeg.predict(self.rand_module(valid_Bs[i].to(self.device)))
            self.rand_module.randomize()
            Seg_valid_2 = self.netSeg.predict(self.rand_module(valid_Bs[i].to(self.device)))
            self.rand_module.randomize()
            Seg_valid_3 = self.netSeg.predict(self.rand_module(valid_Bs[i].to(self.device)))
            Seg_valid = (Seg_valid_1 + Seg_valid_2 + Seg_valid_3) / 3
            loss_tumor_valid_GA = weighted_loss2D(valid_B_labels[i].cuda(), torch.from_numpy(Seg_valid).cuda(), annotate_mask = valid_B_masks[i].cuda(), weight=5.0)
            loss_LD_valid = loss_tumor_valid_GA.data.cpu().numpy().mean()
            running_valid_loss += loss_LD_valid
        running_valid_loss /= len(valid_Bs)
        return running_valid_loss

    
    def optimize_parameters_Seg(self, alpha):
        self.train()
        self.forward(alpha)

        self.set_requires_grad([self.netSeg], True)
        self.optimizer_Seg.zero_grad()
        self.backward_Seg()
        nn.utils.clip_grad_norm_(self.netSeg.parameters(), max_norm=2.0, norm_type=2) 
        self.optimizer_Seg.step()
        
