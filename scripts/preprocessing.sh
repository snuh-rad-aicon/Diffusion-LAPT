#!/bin/bash

python ldm/data/preprocessing.py --input_dir path/to/nifti/data/train --output_dir path/to/npy/data/train
python ldm/data/preprocessing.py --input_dir path/to/nifti/data/val --output_dir path/to/npy/data/val