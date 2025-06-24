from torch.multiprocessing import Pool
import os
from glob import glob
import numpy as np
import json
import random
import ants
from tqdm import tqdm
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess CTP and DWI images')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with nifti files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for numpy files')
    return parser.parse_args()
    
def _single_scan(data_dir):
    data_id = data_dir.split('/')[-1]
    dwi_path = os.path.join(data_dir, 'dwi.nii.gz')
    adc_path = os.path.join(data_dir, 'adc.nii.gz')
    ctp_path = os.path.join(data_dir, 'ctp.nii.gz')        
    mask_path = os.path.join(data_dir, 'mask.nii.gz')
    lesion_path = os.path.join(data_dir, 'lesion.nii.gz')
    
    # Create output directory for this scan if it doesn't exist
    os.makedirs(os.path.join(output_dir, data_id), exist_ok=True)
    
    dwi = ants.image_read(dwi_path).numpy()
    adc = ants.image_read(adc_path).numpy()
    ctp = ants.image_read(ctp_path).numpy()
    mask = ants.image_read(mask_path).numpy().astype(np.uint8)
    lesion = ants.image_read(lesion_path).numpy()
    
    dwi[dwi < 0] = 0
    adc[adc < 0] = 0

    dwi_signal = dwi[mask > 0]
    adc_signal = adc[mask > 0]
    
    dwi_minv, dwi_maxv = np.quantile(dwi_signal, 0.005), np.quantile(dwi_signal, 0.995)
    adc_minv, adc_maxv = np.quantile(adc_signal, 0.005), np.quantile(adc_signal, 0.995)
    np.save(os.path.join(output_dir, data_id, f'dwi_quantile_min_max.npy'), np.array([dwi_minv, dwi_maxv]))
    np.save(os.path.join(output_dir, data_id, f'adc_quantile_min_max.npy'), np.array([adc_minv, adc_maxv]))

    num_slices = dwi.shape[2]
    data = []
    for i in range(num_slices):
        dwi_slice = dwi[:, :, i]
        adc_slice = adc[:, :, i]
        ctp_slice = ctp[:, :, i]
        mask_slice = mask[:, :, i]
        lesion_slice = lesion[:, :, i]
        
        dwi_signal = dwi_slice[mask_slice > 0]
        adc_signal = adc_slice[mask_slice > 0]
        ctp_signal = ctp_slice[mask_slice > 0]
        if dwi_signal.sum() == 0 or adc_signal.sum() == 0 or ctp_signal.sum() == 0:
            continue

        ctp_slice = ctp_slice.transpose(2, 0, 1)
        np.save(os.path.join(output_dir, data_id, f'dwi_{i}.npy'), dwi_slice)
        np.save(os.path.join(output_dir, data_id, f'adc_{i}.npy'), adc_slice)
        np.save(os.path.join(output_dir, data_id, f'ctp_{i}.npy'), ctp_slice)
        np.save(os.path.join(output_dir, data_id, f'mask_{i}.npy'), mask_slice)
        np.save(os.path.join(output_dir, data_id, f'lesion_{i}.npy'), lesion_slice)

        data.append(i)

    return {'data_id': data_id, 'slice_idx': data, 'n': len(data)}

def main():
    args = parse_args()
    global output_dir
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of data directories
    data_list = [d for d in glob(os.path.join(args.input_dir, '*')) if os.path.isdir(d)]
    print(f"Found {len(data_list)} subjects in {args.input_dir}")
    
    with Pool(16) as pool:
        dataset = list(tqdm(pool.imap(_single_scan, data_list), total=len(data_list)))
        
    CTPDWI_slice_idx_dict = {}
    for d in dataset:
        CTPDWI_slice_idx_dict[f'{d["data_id"]}'] = {'slice_idx': d['slice_idx'], 'n': d['n']}

    # save dataset as json
    with open(f'{output_dir}/CTPDWI_slice_idx.json', 'w') as f:
        json.dump(CTPDWI_slice_idx_dict, f)
    
    print(f"Preprocessing completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main()