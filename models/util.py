
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pdb

def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()

def cell_count_loss(y_true, y_pred):
    mean_diff = torch.mean(torch.abs(y_pred - y_true))
    cc_loss = 1.0 - 1.0 / (1.0 + mean_diff)
    return cc_loss

def log_cosh_loss(y_true, y_pred, label_weights, norm_const=None):
    if norm_const is not None:
        diff = (y_pred - y_true) / (norm_const + 1e-12);
    else:
        diff = y_pred - y_true
    weighted_logcosh = torch.mean(torch.sum(torch.log(torch.cosh(diff + 1e-12)) * label_weights, dim=1))
    return weighted_logcosh

def get_weight_mask(y_true, params = None):
    if params is not None:
        y_true = y_true.float() / 255.0 * params['scale']
        mean_label = torch.mean(torch.mean(y_true, dim = -1, keepdim=True), dim=-2, keepdim=True)
        y_mask = y_true / params['scale'] + params['alpha'] * mean_label / params['scale']
    else:
        y_mask = torch.ones(y_true.size())
    return y_true, y_mask

def get_scale_label(y_true, params = None):
    if params is not None:
        y_true = y_true.float() / 255.0 * params['scale']
    return y_true

def mean_squared_error(y_true, y_pred, y_mask, annotate_mask=None):
    diff = y_pred - y_true
    naive_loss = diff*diff
    masked = torch.zeros_like(naive_loss)
    if annotate_mask is not None:
        nN, nH, nW, nChs = y_pred.size()
        for i in range(nN):
            if hasattr(y_pred, 'requires_grad'):
                if torch.sum(annotate_mask[i]).data.cpu().numpy() > 0:
                    masked[i] = naive_loss[i] * y_mask[i] * annotate_mask[i] * (nH * nW * nChs / torch.sum(annotate_mask[i]))
                else:
                    masked[i] = naive_loss[i] * y_mask[i] * annotate_mask[i]
            else:
                if torch.sum(annotate_mask[i]) > 0:
                    masked[i] = naive_loss[i] * y_mask[i] * annotate_mask[i] * (nH * nW * nChs / torch.sum(annotate_mask[i]))
                else:
                    masked[i] = naive_loss[i] * y_mask[i] * annotate_mask[i]
    else:
        masked = naive_loss * y_mask
    return torch.mean(masked)

def weighted_loss(y_true, y_pred, y_mask, annotate_mask=None):
    assert y_pred.dim() == 4, 'dimension is not matched!!'
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if annotate_mask is not None:
        if annotate_mask.dim() == 3:
            annotate_mask = annotate_mask.unsqueeze(1)
        annotate_mask = annotate_mask.float() / 255.0
        annotate_mask = annotate_mask.permute(0,2,3,1)
    y_true = y_true.permute(0,2,3,1)
    y_pred = y_pred.permute(0,2,3,1)
    y_mask = y_mask.permute(0,2,3,1)
    masked_loss = mean_squared_error(y_true,y_pred,y_mask,annotate_mask)
    return masked_loss

def binary_crossentropy(y_true, y_pred, annotate_mask=None, weight=1.0):

    if annotate_mask is not None:
        sumit = torch.zeros_like(y_pred)
        nN, nChs, ND, nH, nW = y_pred.size()
        for i in range(nN):
            loss = -1.0 * weight*y_true[i]*torch.log(torch.sigmoid(y_pred[i]) +1e-10) - (1-y_true[i]) * torch.log(1-torch.sigmoid(y_pred[i])+1e-10)
            if hasattr(y_pred, 'requires_grad'):
                if torch.sum(annotate_mask[i]).data.cpu().numpy() > 0:
                    sumit[i] = loss * annotate_mask[i] * (nChs * ND * nH * nW / torch.sum(annotate_mask[i]))
                else:
                    sumit[i] = loss * annotate_mask[i]
            else:
                if torch.sum(annotate_mask[i]) > 0:
                    sumit[i] = loss * annotate_mask[i] * (nChs * ND * nH * nW / torch.sum(annotate_mask[i]))
                else:
                    sumit[i] = loss * annotate_mask[i]
    else:
        sumit = -1.0 * weight*y_true*torch.log(torch.sigmoid(y_pred) + 1e-10) - (1-y_true) * torch.log(1-torch.sigmoid(y_pred) + 1e-10)
    return torch.mean(sumit)

def weighted_loss3D(y_true, y_pred, annotate_mask=None, weight=1.0):
    assert y_pred.dim() == 5, 'dimension is not matched!!'
    if y_true.dim() == 4:
        y_true = y_true.unsqueeze(1)
    if annotate_mask is not None:
        if annotate_mask.dim() == 4:
            annotate_mask = annotate_mask.unsqueeze(1)
        annotate_mask = annotate_mask.float() / 255.0
    masked_loss = binary_crossentropy(y_true, y_pred, annotate_mask=annotate_mask, weight=weight)
    return masked_loss

def binary_crossentropy2D(y_true, y_pred, annotate_mask=None, weight=1.0):
    if annotate_mask is not None:
        sumit = torch.zeros_like(y_pred)
        nN, nChs, nH, nW = y_pred.size()
        for i in range(nN):
            loss = -1.0 * weight*y_true[i]*torch.log(torch.sigmoid(y_pred[i]) +1e-10) - (1-y_true[i]) * torch.log(1-torch.sigmoid(y_pred[i])+1e-10)
            if hasattr(y_pred, 'requires_grad'):
                if torch.sum(annotate_mask[i]).data.cpu().numpy() > 0:
                    sumit[i] = loss * annotate_mask[i] * (nChs * nH * nW / torch.sum(annotate_mask[i]))
                else:
                    sumit[i] = loss * annotate_mask[i]
            else:
                if torch.sum(annotate_mask[i]) > 0:
                    sumit[i] = loss * annotate_mask[i] * (nChs * nH * nW / torch.sum(annotate_mask[i]))
                else:
                    sumit[i] = loss * annotate_mask[i]
    else:
        sumit = -1.0 * weight*y_true*torch.log(torch.sigmoid(y_pred) + 1e-10) - (1-y_true) * torch.log(1-torch.sigmoid(y_pred) + 1e-10)
    return torch.mean(sumit)

def dice(y_true, y_pred, annotate_mask=None, smooth=1e-6, epsilon=1e-6, weight=None):
    if annotate_mask is not None:
        assert y_pred.size() == annotate_mask.size()
        y_pred_n = torch.sigmoid(y_pred)
        
        y_pred_masked = y_pred_n * annotate_mask
        assert y_pred_masked.size() == y_pred.size()

        inputs = y_pred_masked.view(-1)
        targets = y_true.view(-1)

        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice = 1 - dice_coeff
    else:
        y_pred_n = torch.sigmoid(y_pred)

        y_pred_masked = y_pred_n * annotate_mask
        assert y_pred_masked.size() == y_pred.size()

        inputs = y_pred_masked.view(-1)
        targets = y_true.view(-1)

        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice = 1 - dice_coeff
    return dice

def bce_dice(y_true, y_pred, annotate_mask=None, weight=1.0, alpha=1.0, beta=1.0, smooth=1e-6):
    bce_masked_loss = binary_crossentropy2D(y_true, y_pred, annotate_mask=annotate_mask, weight=weight)
    dice_masked_loss = dice(y_true, y_pred, annotate_mask=annotate_mask, smooth=smooth)
    combo_loss = (alpha * bce_masked_loss) + (beta * dice_masked_loss)
    return combo_loss

def focal_loss(y_true, y_pred, annotate_mask=None, alpha=0.25, gamma=2):
	bce = binary_crossentropy2D(y_true, y_pred, annotate_mask=annotate_mask)
	bce_exp = torch.exp(-bce)
	focal_loss = alpha * ((1 - bce_exp)**gamma) * bce
	return focal_loss

def weighted_loss2D(y_true, y_pred, annotate_mask=None, weight=1.0):
    assert y_pred.dim() == 4, 'dimension is not matched!!'
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if annotate_mask is not None:
        if annotate_mask.dim() == 3:
            annotate_mask = annotate_mask.unsqueeze(1)
        annotate_mask = annotate_mask.float()
    masked_loss = bce_dice(y_true, y_pred, annotate_mask=annotate_mask, weight=weight, alpha=6.0, beta=1.0, smooth=1e-6)
    return masked_loss
