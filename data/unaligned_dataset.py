import os
import torch
from data.base_dataset import BaseDataset, get_transform, get_params, get_params_s1
from data.image_folder import make_dataset
from PIL import Image
import random
import glob
import numpy as np
import skimage
from scipy.ndimage import gaussian_filter
from skimage.filters import unsharp_mask

class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.data_dir = 'sourcedata_dir'
        self.realdata_dir = 'targetdata_dir'
        self.base_data = 'source_dir'
        self.style_data = 'target_dir'

        self.A_paths = []
        train_base_data =  (' ',) # tuple of training subjects
        for id in train_base_data:
            self.A_paths += '' # training image files (paths)
            
        self.A_size = len(self.A_paths)
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        params_A = get_params(self.opt, [128, 128])
        params_B = get_params(self.opt, [128, 128])

        self.transform_A = get_transform(self.opt, params=params_A, grayscale=(input_nc == 1), normalize=True)
        self.transform_A_labelmask = get_transform(self.opt, params=params_A, grayscale=(input_nc == 1), normalize=False)

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]
        A_path_list = A_path.split('/')
        A_label_path = os.path.join('/','') # path to the lesion label
        A_mask_path = os.path.join('/','') # path to the liver mask
        
        A_img = Image.open(A_path).convert('L') #$
        A_label_PIL = Image.open(A_label_path).convert('L') #$
        A_mask_PIL = Image.open(A_mask_path).convert('L') #$
        
        # apply image transformation
        if random.random() < 0.5:
            A = torch.from_numpy(skimage.util.random_noise(self.transform_A(A_img), mode='speckle'))
        else:
            A = self.transform_A(A_img)
            
        A_label = (self.transform_A_labelmask(A_label_PIL) > 0.5).long()
        A_mask = (self.transform_A_labelmask(A_mask_PIL) > 0.5).long()
        
        return {'A': A, 'A_label': A_label, 'A_mask': A_mask, 'A_paths': A_path}

    def __len__(self):
        return self.A_size
