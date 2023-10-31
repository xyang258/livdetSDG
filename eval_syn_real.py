import os, sys
import time
from tqdm import *

import numpy as np
from skimage.feature import peak_local_max
from skimage import measure
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from scipy.ndimage import uniform_filter

from options.eval_options import EvalOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from torchvision import transforms
from PIL import Image

from tools.analysis_util import get_seed_name, get_labelmap_name, printCoords_seg_slc, removeSmallRegions
from tools.util import sigmoid

import scipy.io as sio
import copy
import pdb

import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    opt = EvalOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()
    train_val_test = opt.train_val_test
    datasetname = opt.datasetname
    datadir = opt.datadir
    eval_result_folder = opt.eval_result_folder
    filtering = opt.filtering
    fix_test = opt.fix_test
    test_all = opt.test_all
    name = opt.name
    epoch = opt.epoch

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids[0])

    data_split = train_val_test
    if data_split == 'test':

        num_cls = 1
        local_min_len = 1
        avg_filter_size = 2 * local_min_len + 1
        iou_pool = np.arange(0.0, 1.01, 0.05)
        area_pool = [5,10,20]

        savefolder = os.path.join(eval_result_folder, data_split, datasetname, opt.model_name+'_run'+opt.run_number+'/ep' + epoch + opt.model_suffix)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        resultsDict = {}
        votingmap_name = 'votingmap'
        voting_time_name = 'prediction_time'
        mask_name = 'mask'
        mask = None
        threshold_pool = np.arange(0.0, 1.01, 0.05)

        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            
            img_path = model.get_image_paths()
            subject_name = img_path[0].split('/')[-3]
            im_name_ext = img_path[0].split('/')[-1]
            im_name = im_name_ext.split('.')[0]
            im_idx = int(im_name)-1

            
            print('im_name:',im_name)
            print('subject_name:',subject_name)

            votingStarting_time = time.time()

            VotingMap = torch.from_numpy((visuals['pred_1'][0,0,:,:] + visuals['pred_2'][0,0,:,:] + visuals['pred_3'][0,0,:,:]) / 3).cpu().numpy()
            mask = visuals['test_B_mask'][0,0,:,:].cpu().numpy()
            mask_ori = (visuals['test_B_mask_ori'][0,:,:]).cpu().numpy()


            votingEnding_time = time.time()
            resultsDict[voting_time_name] = votingEnding_time - votingStarting_time

            max_pred = np.max(np.multiply(VotingMap, mask))
            max_pred_nomask = np.max(VotingMap)

            print('subject name: ', subject_name)
            savefolder_subject = os.path.join(savefolder, subject_name)
            if not os.path.exists(savefolder_subject):
                os.makedirs(savefolder_subject)

            print('prediction of ' + im_name + '.png')
            resultDictPath_mat = os.path.join(savefolder_subject, im_name + '.mat')
            cur_Votingmap = copy.deepcopy(VotingMap)
            cur_mask = copy.deepcopy(mask)
            resultsDict[votingmap_name] = np.copy(cur_Votingmap)
            curr_dir = os.getcwd()

            original_pet_image = os.path.join(datadir, 'liver', subject_name, 'images', im_name_ext)
            print(original_pet_image)
            if mask is not None:
                resultsDict[mask_name] = np.copy(cur_mask)

            print(f'localmax_len = {local_min_len}, avgfilter_size = {avg_filter_size}, filtering = {filtering}.')
            if mask is not None:
                if filtering:
                    VotingMap_filter = copy.deepcopy(cur_Votingmap)
                    VotingMap_filter[VotingMap_filter < 0] = 0.0
                    VotingMap_filter = uniform_filter(VotingMap_filter, size=avg_filter_size)
                    VotingMap_filter = np.multiply(VotingMap_filter, cur_mask)
                    VotingMap_filter_orig = copy.deepcopy(cur_Votingmap)
                    VotingMap_filter_orig[VotingMap_filter_orig < 0] = 0.0
                    VotingMap_filter_orig = uniform_filter(VotingMap_filter_orig, size=avg_filter_size)
                else:
                    VotingMap_filter = np.multiply(copy.deepcopy(cur_Votingmap), cur_mask)
                    VotingMap_filter[VotingMap_filter < 0] = 0.0
                    VotingMap_filter_orig = copy.deepcopy(cur_Votingmap)
                    VotingMap_filter_orig[VotingMap_filter_orig < 0] = 0.0
                    
                VotingMap_filter = cv2.resize(VotingMap_filter, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                VotingMap_filter = np.multiply(VotingMap_filter, mask_ori)

                for threshhold in threshold_pool:
                    for area_thd in area_pool:
                        VotingMap_copy = copy.deepcopy(VotingMap_filter)
                        VotingMap_copy[VotingMap_copy <= threshhold*max_pred] = 0
                        VotingMap_copy[VotingMap_copy > threshhold*max_pred] = 1
                        labelmapname = get_labelmap_name(threshhold, area_thd)
                        labelnumname = get_labelmap_name(threshhold, area_thd) + '_number'
                        labelmaptime = get_labelmap_name(threshhold, area_thd) + '_time'
                        thisStart = time.time()
                        map_label, num_label = measure.label(VotingMap_copy.astype(int), return_num = True)
                        thisEnd = time.time()
                        if num_label == 0:
                            print("No detection for img:{s} for parameter t_{thd:3.2f} and a_{area:02d}".format(s=im_name[0], thd=threshhold, area=area_thd))
                        else:
                            map_label, num_label = removeSmallRegions(map_label, area_thd)

                        resultsDict[labelmapname] = map_label
                        resultsDict[labelnumname] = num_label
                        resultsDict[labelmaptime] = thisEnd - thisStart +  resultsDict[voting_time_name]
            else: # no mask
                if filtering: # using box filtering
                    VotingMap_filter = copy.deepcopy(cur_Votingmap)
                    VotingMap_filter[VotingMap_filter < 0] = 0.0
                    VotingMap_filter = uniform_filter(VotingMap_filter, size=avg_filter_size)
                else:
                    VotingMap_filter = copy.deepcopy(cur_Votingmap)
                    VotingMap_filter[VotingMap_filter < 0] = 0.0
                    
                VotingMap_filter = cv2.resize(VotingMap_filter, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

                for threshhold in threshold_pool:
                    for area_thd in area_pool:
                        VotingMap_copy = copy.deepcopy(VotingMap_filter)
                        VotingMap_copy[VotingMap_copy <= threshhold*max_pred_nomask] = 0
                        VotingMap_copy[VotingMap_copy > threshhold*max_pred_nomask] = 1
                        labelmapname = get_labelmap_name(threshhold, area_thd)
                        labelnumname = get_labelmap_name(threshhold, area_thd) + '_number'
                        labelmaptime = get_labelmap_name(threshhold, area_thd) + '_time'
                        thisStart = time.time()
                        map_label, num_label = measure.label(VotingMap_copy.astype(int), return_num = True)
                        thisEnd = time.time()
                        if num_label == 0:
                            print("No detection for img:{s} for parameter t_{thd:3.2f} and a_{area:02d}".format(s=im_name[0], thd=threshhold, area=area_thd))
                        else:
                            map_label, num_label = removeSmallRegions(map_label, area_thd)

                        resultsDict[labelmapname] = map_label
                        resultsDict[labelnumname] = num_label
                        resultsDict[labelmaptime] = thisEnd - thisStart +  resultsDict[voting_time_name]

            sio.savemat(resultDictPath_mat, resultsDict)

            # overlay predictions on images
            if datasetname == 'liver2D':
                imgfolder = os.path.join(datadir, datasetname[0:5], subject_name, 'images')
            else:
                imgfolder = os.path.join(datadir, datasetname, subject_name, 'images')
            resultfolder = savefolder_subject
            printCoords_seg_slc(savefolder_subject, resultfolder, im_name, imgfolder, ['.png', '.jpg', '.bmp'], threshhold=threshold_pool[1], area_thd=area_pool[0], mask=mask, alpha=1)
       
