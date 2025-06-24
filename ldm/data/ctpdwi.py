import os
from glob import glob
import numpy as np
import json
import random
import ants
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Pool
from scipy.ndimage import morphology
from .transforms import *


class CTPDWI(Dataset):
    def __init__(self, dataroot, phase, transform):
        super().__init__()
        self.dataroot = dataroot
        self.phase = phase
        self.transform = transform

        dataset_slice_idx_dict_path = os.path.join(self.dataroot, 'CTPDWI_slice_idx.json')
        dataset_slice_idx_dict = json.load(open(dataset_slice_idx_dict_path, 'r'))
        self.dataset = [[k, slice_idx] for k, v in dataset_slice_idx_dict.items() for slice_idx in v['slice_idx']]
        self.dataset_slice_idx_dict = dataset_slice_idx_dict

    def __getitem__(self, index):
        data_id, i = self.dataset[index]
        ctp_path = os.path.join(self.dataroot, data_id, f'ctp_{i}.npy')
        dwi_path = os.path.join(self.dataroot, data_id, f'dwi_{i}.npy')
        adc_path = os.path.join(self.dataroot, data_id, f'adc_{i}.npy')
        mask_path = os.path.join(self.dataroot, data_id, f'mask_{i}.npy')
        lesion_path = os.path.join(self.dataroot, data_id, f'lesion_{i}.npy')
        dwi_min_max_path = os.path.join(self.dataroot, data_id, f'dwi_quantile_min_max.npy')
        adc_min_max_path = os.path.join(self.dataroot, data_id, f'adc_quantile_min_max.npy')
        
        ctp = np.load(ctp_path)
        dwi = np.load(dwi_path)
        adc = np.load(adc_path)
        mask = np.load(mask_path)
        lesion = np.load(lesion_path)
        dwi_minv, dwi_maxv = np.load(dwi_min_max_path)
        adc_minv, adc_maxv = np.load(adc_min_max_path)
        
        data = self.transform({'ctp': ctp, 'dwi': dwi, 'adc': adc, 'mask': mask, 'lesion': lesion, 'dwi_minv': dwi_minv, 'dwi_maxv': dwi_maxv, 'adc_minv': adc_minv, 'adc_maxv': adc_maxv})
        data.update({'data_id': data_id, 'slice': i})
        
        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)

class CTPDWITrain(CTPDWI):
    def __init__(self, dataroot):
        transform = get_train_transforms()
        super().__init__(dataroot, phase='train', transform=transform)


class CTPDWIValidation(CTPDWI):
    def __init__(self, dataroot):
        transform = get_val_transforms()
        super().__init__(dataroot, phase='val', transform=transform)


class CTPDWITest(CTPDWI):
    def __init__(self, dataroot):
        transform = get_val_transforms()
        super().__init__(dataroot, phase='test', transform=transform)