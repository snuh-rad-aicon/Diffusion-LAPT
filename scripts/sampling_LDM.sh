#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/sample_LDM.py \
    -r path/to/LDM/checkpoint.ckpt \
    --n_samples 1 \
    --batch_size 48 \
    --steps 2