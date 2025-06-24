#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --base configs/latent-diffusion/ldm.yaml -t --gpus 0, --project Diffusion-LAPT
