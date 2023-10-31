
import time
import os
import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from data.base_dataset import get_transform
import torch
import numpy as np


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    
    # get validation dataset
    params_Bv = dict()
    params_Bv['crop_pos'] = (0, 0)
    params_Bv['flip'] = False
    params_Bv['colorjitter'] = False
    params_Bv['new_w'] = opt.crop_size
    params_Bv['new_h'] = opt.crop_size 
    # source validation
    transform_A = get_transform(opt, params=params_Bv, grayscale=(opt.output_nc == 1), normalize=True)
    transform_A_labelmask = get_transform(opt, params=params_Bv, grayscale=(opt.output_nc == 1), normalize=False)
    sourcedata_dir = 'sourcedata_dir'
    source_data = 'source_dir'
    A_valid_paths = []
    valid_A_data = ('',) # tuple of validation subjects
    for id in valid_A_data:
        A_valid_paths += ''# path to validation images
    
    A_valids = []
    A_valid_labels = []
    A_valid_masks = []
    for idx, A_path in enumerate(A_valid_paths):
        A_valid_path = A_valid_paths[idx]
        A_valid_path_list = A_valid_path.split('/')
        A_valid_label_path = ''# path to lesion label
        A_valid_mask_path = ''# path to liver mask
        A_valid_img_PIL = Image.open(A_valid_path).convert('L') #$
        A_valid_label_PIL = Image.open(A_valid_label_path).convert('L') #$
        A_valid_mask_PIL = Image.open(A_valid_mask_path).convert('L') #$
        A_valids.append(transform_A(A_valid_img_PIL).unsqueeze(0))
        A_valid_labels.append((transform_A_labelmask(A_valid_label_PIL)>0.5).long().unsqueeze(0))
        A_valid_masks.append((transform_A_labelmask(A_valid_mask_PIL)>0.5).long().unsqueeze(0))

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0
    
    
    # logger
    logPath = 'log/'+opt.model_name+'_run'+str(opt.run_number)
    writer = SummaryWriter(logPath)
    
    # for validation
    valid_vals_EMA, coef_EMA, count_EMA, count_ = [], 0.9, 1, 0
    best_score = 10000.0
    best_score_EMA = 10000.0
    tolerance = 5000
    tol_total_iters = 99999999

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            
            p = float(i + epoch * dataset_size) / (opt.n_epochs+opt.n_epochs_decay) / dataset_size
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            model.optimize_parameters_Seg(alpha)


            if total_iters % opt.print_freq == 0 and total_iters>0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                for k, v in losses.items():
                    writer.add_scalar(k, v, total_iters)
                
                # validation
                loss_Seg_valid = model.valid_Seg(valid_Bs=A_valids, valid_B_labels=A_valid_labels, valid_B_masks=A_valid_masks)
                writer.add_scalar('Seg_validA', loss_Seg_valid, total_iters)
                if not valid_vals_EMA: # exponential moving weighted average
                    valid_val_cur = 0.0
                valid_val_cur = coef_EMA * valid_val_cur + (1 - coef_EMA) * loss_Seg_valid
                valid_val_cur_biascorr = valid_val_cur / (1 - coef_EMA ** count_EMA)
                valid_vals_EMA.append(valid_val_cur_biascorr)
                count_EMA += 1

                print('\nValidation A loss-EMA: {0}, best score-EMA: {1}'.format(valid_val_cur_biascorr, best_score_EMA))
                if valid_val_cur_biascorr <=  best_score_EMA:
                    best_score_EMA = valid_val_cur_biascorr
                    print('update to new bestA_score_EMA: {}'.format(best_score_EMA))
                    model.save_networks('bestA_EMA')
                    print('Save best weights to: bestA_EMA*.pth')
                    count_ = 0
                else:
                    count_ = count_ + 1
                print('\nValidation A loss: {}, best_score: {}'.format(loss_Seg_valid, best_score))#losses['Seg_valid']
                if loss_Seg_valid <=  best_score:
                    best_score = loss_Seg_valid
                    print('update to new bestA_score: {}'.format(best_score))
                    model.save_networks('bestA')
                    print('Save best weights to: bestA*.pth')

                if count_ >= tolerance:
                    if tol_total_iters > total_iters:
                        tol_total_iters = total_iters
                    print('performance not imporoved for so long, since ', tol_total_iters)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        
    writer.close()
