import numpy as np
import torch
from monai import transforms



class LastTransfromCTPDWI(transforms.MapTransform):
    def __init__(
        self,
        keys = ('dwi', 'adc', 'ctp', 'mask'),
        rand_gaussian_noise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.rand_gaussian_noise = rand_gaussian_noise

    def __call__(self, data):
        d = dict(data)
        
        if 'dwi' in d:
            dwi = d['dwi']
            adc = d['adc']
            mask = d['mask']
            
            dwi_minv, dwi_maxv = d['dwi_minv'], d['dwi_maxv']
            adc_minv, adc_maxv = d['adc_minv'], d['adc_maxv']
            
            dwi = np.clip(dwi, dwi_minv, dwi_maxv)
            adc = np.clip(adc, adc_minv, adc_maxv)
            
            dwi = (dwi - dwi_minv) / (dwi_maxv - dwi_minv + 1e-8)
            adc = (adc - adc_minv) / (adc_maxv - adc_minv + 1e-8)
            
            if self.rand_gaussian_noise:
                rand_apply = np.random.rand()
                if rand_apply < 0.5:
                    dwi_noise = np.random.normal(0, 0.2, dwi.shape)
                    adc_noise = np.random.normal(0, 0.2, adc.shape)
                    dwi = dwi + dwi_noise
                    adc = adc + adc_noise
                    dwi = np.clip(dwi, 0, 1)
                    adc = np.clip(adc, 0, 1)
            
            mri = np.concatenate([dwi, adc], axis=0)
            mri = mri * mask
            mri = mri * 2 - 1
            
            d['mri'] = mri
            d['mask'] = mask
            d['dwi_minv'] = float(dwi_minv)
            d['dwi_maxv'] = float(dwi_maxv)
            d['adc_minv'] = float(adc_minv)
            d['adc_maxv'] = float(adc_maxv)
            
        if 'ct' in d:
            ct = d['ct']
            mask = d['mask']
            
            ct_minv, ct_maxv = 0., 200.
            
            ct = np.clip(ct, ct_minv, ct_maxv)
            ct = (ct - ct_minv) / (ct_maxv - ct_minv + 1e-8)
            ct = ct * mask
            ct = ct * 2 - 1
            
            d['ct'] = ct
            d['mask'] = mask
            d['ct_minv'] = float(ct_minv)
            d['ct_maxv'] = float(ct_maxv)
            
        if 'ctp' in d:
            ctp = d['ctp']
            mask = d['mask']
            
            ctp_minv, ctp_maxv = 0., 200.
            
            ctp = np.clip(ctp, ctp_minv, ctp_maxv)
            ctp = (ctp - ctp_minv) / (ctp_maxv - ctp_minv + 1e-8)
            ctp = ctp * mask
            ctp = ctp * 2 - 1
            
            d['ctp'] = ctp
            d['mask'] = mask
            d['ctp_minv'] = float(ctp_minv)
            d['ctp_maxv'] = float(ctp_maxv)
            
        d['lesion'] = d['lesion'].astype(np.float32)
        # lesion = np.zeros((2, *d['lesion'].shape[1:]), dtype=np.int64)
        # lesion[0, d['lesion'][0] == 0] = 1
        # lesion[1, d['lesion'][0] == 1] = 1
        # d['lesion'] = lesion
        
        return d

def get_train_transforms():
    train_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], channel_dim='no_channel', allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
            transforms.RandAffined(
                keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"],
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest", "nearest"),
                prob=0.9,
                rotate_range=(np.pi/16, np.pi/16),
                translate_range=(32, 32),
                scale_range=(0.001, 0.001),
                padding_mode="border",
                allow_missing_keys=True,
            ),
            LastTransfromCTPDWI(),
            transforms.ToTensord(keys=["dwi", "adc", "ct", "mri", "ctp", "mask", "lesion", "gt", "input"], allow_missing_keys=True),
        ]
    )
    return train_transform

def get_val_transforms():
    val_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], channel_dim='no_channel', allow_missing_keys=True),
            LastTransfromCTPDWI(),
            transforms.ToTensord(keys=["dwi", "adc", "ct", "mri", "ctp", "mask", "lesion", "gt", "input"], allow_missing_keys=True),
        ]
    )
    return val_transform



def get_train_bigaug_transforms():
    train_transform = transforms.Compose(
        [
            # 기존 변환
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "mask", "lesion"], channel_dim='no_channel', allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "mask", "lesion"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "mask", "lesion"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
            transforms.RandGaussianSmoothd(
                keys=["dwi", "adc"],
                sigma_x=(0.25, 1.0),  # 상한값 감소
                sigma_y=(0.25, 1.0),
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            transforms.RandGaussianSharpend(
                keys=["dwi", "adc"],
                sigma1_x=(0.25, 0.5),  # 상한값 감소
                sigma1_y=(0.25, 0.5),
                sigma2_x=(0.8, 1.2),  # 상한값 감소
                sigma2_y=(0.8, 1.2),
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            transforms.RandAdjustContrastd(
                keys=["dwi", "adc"],
                gamma=(0.8, 2.0),  # 범위 축소
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            transforms.RandShiftIntensityd(
                keys=["dwi", "adc"],
                offsets=(-0.03, 0.03),  # 범위 축소
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            transforms.RandScaleIntensityd(
                keys=["dwi", "adc"],
                factors=(0.95, 1.05),  # 범위 축소
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            transforms.RandRotated(
                keys=["dwi", "adc", "mask", "lesion"],
                range_x=(-0.2, 0.2),  # 회전 범위 축소 (≈ ±11.5°)
                range_y=(-0.2, 0.2),
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            transforms.RandZoomd(
                keys=["dwi", "adc", "mask", "lesion"],
                min_zoom=0.8,  # 최소 확대 비율 증가
                max_zoom=1.2,  # 최대 확대 비율 감소
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            transforms.RandGridDistortiond(
                keys=["dwi", "adc", "mask", "lesion"],
                distort_limit=(-0.01, 0.01),  # 왜곡 범위 축소
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                prob=0.3,  # 적용 확률 감소
                allow_missing_keys=True,
            ),
            LastTransfromCTPDWI(rand_gaussian_noise=True),
            transforms.ToTensord(keys=["dwi", "adc", "mri", "mask", "lesion", "gt", "input"], allow_missing_keys=True),
        ]
    )
    return train_transform
