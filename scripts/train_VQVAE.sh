#!/bin/bash

CUDA_VISIBLE_DEVICES=0, python main.py --base configs/autoencoder/VQGAN.yaml -t --gpus 0, --project Diffusion-LAPT
