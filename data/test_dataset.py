from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import glob
import os
import numpy as np


class TestDataset(BaseDataset):

    def __init__(self, opt):
        
        BaseDataset.__init__(self, opt)
        
        self.data_dir = 'sourcedata_dir'
        self.realdata_dir = 'targetdata_dir'
        self.base_data = 'source_dir'
        self.style_data = 'target_dir'
        
        self.B_test_paths = []
        test_style_data = (' ',) # tuple of testing subjects
        for id in test_style_data:
            self.B_test_paths += '' # testing image files (paths)
        
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1), normalize=True)
        self.transform_B_labelmask = get_transform(opt, grayscale=(output_nc == 1), normalize=False)

    def __getitem__(self, index):
        
        B_test_path = self.B_test_paths[index]
        B_test_path_list = B_test_path.split('/')
        B_test_mask_path = os.path.join('/','') # path to the liver mask
        
        B_test_img_PIL = Image.open(B_test_path).convert('L')
        B_test_mask_PIL = Image.open(B_test_mask_path).convert('L')
        
        B_test = self.transform(B_test_img_PIL)
        B_test_mask = (self.transform_B_labelmask(B_test_mask_PIL) > 0.5).long()
        B_test_mask_ori = np.array(B_test_mask_PIL) / 255
        return {'B_test': B_test, 'B_test_mask': B_test_mask, 'B_test_mask_ori': B_test_mask_ori, 'B_test_paths': B_test_path}


    def __len__(self):
        return len(self.B_test_paths)
