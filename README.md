# Lesion-Aware Post-Training of Latent Diffusion Models for Synthesizing Diffusion MRI from CT Perfusion

[![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025-blue)](https://miccai.org/en/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./paper.pdf)

**Provisionally accepted at MICCAI 2025 (Top 9% Submission)**

## Authors

**Junhyeok Lee**\*, **Hyunwoong Kim**\*, **Hyungjin Chung**, **Heeseong Eom**, **Joon Jang**, **Chul-Ho Sohn**, **Kyu Sung Choi**†

<sup>*</sup>These authors contributed equally to this paper.  
<sup>†</sup>Corresponding author

This repository contains the official PyTorch implementation for the paper "Lesion-Aware Post-Training of Latent Diffusion Models for Synthesizing Diffusion MRI from CT Perfusion".

## 📖 Abstract

Image-to-Image translation models can help mitigate various challenges inherent to medical image acquisition. Latent diffusion models (LDMs) leverage efficient learning in compressed latent space and constitute the core of state-of-the-art generative image models. However, this efficiency comes with a trade-off, potentially compromising crucial pixel-level detail essential for high-fidelity medical images. This limitation becomes particularly critical when generating clinically significant structures, such as lesions, which often occupy only a small portion of the image. Failure to accurately reconstruct these regions can severely impact diagnostic reliability and clinical decision-making. To overcome this limitation, we propose a novel post-training framework for LDMs in medical image-to-image translation by incorporating lesion-aware medical pixel space objectives. This approach is essential, as it not only enhances overall image quality but also improves the precision of lesion delineation. We evaluate our framework on brain CT-to-MRI translation in acute ischemic stroke patients, where early and accurate diagnosis is critical for optimal treatment selection and improved patient outcomes. While diffusion MRI is the gold standard for stroke diagnosis, its clinical utility is often constrained by high costs and low accessibility. Using a dataset of 817 patients, we demonstrate that our framework improves overall image quality and enhances lesion delineation when synthesizing DWI and ADC images from CT perfusion scans, outperforming existing image-to-image translation models. Furthermore, our post-training strategy is easily adaptable to pre-trained LDMs and exhibits substantial potential for broader applications across diverse medical image translation tasks.

## 🚀 Getting Started

#### Data Preprocessing

We provide a preprocessing script to convert NIFTI files to NPY format, which is used for training:

```bash
# Run preprocessing
bash scripts/preprocessing.sh
```

You may need to modify the input and output paths in the script:
```bash
python ldm/data/preprocessing.py --input_dir path/to/nifti/data/train --output_dir path/to/npy/data/train
python ldm/data/preprocessing.py --input_dir path/to/nifti/data/val --output_dir path/to/npy/data/val
```

## 💻 Usage

This project uses `main.py` as the main entry point for training and testing, configured by `.yaml` files.

### Training Pipeline

Our model training follows a three-step process:

#### 1. Train the VQGAN Autoencoder

First, train the VQGAN model to learn a compressed latent space representation:

```bash
bash scripts/train_VQVAE.sh
```

This uses the configuration in `configs/autoencoder/VQGAN.yaml`.

#### 2. Train the Base Latent Diffusion Model (LDM)

Next, train the base LDM for DWI synthesis from CTP:

```bash
bash scripts/train_LDM.sh
```

This uses the configuration in `configs/latent-diffusion/ldm.yaml`.

#### 3. Apply Lesion-Aware Post-Training (LAPT)

Finally, apply our novel LAPT to enhance lesion accuracy:

```bash
bash scripts/train_LDM_LAPT.sh
```

This uses the configuration in `configs/latent-diffusion/ldm_lapt.yaml`.

### Inference

To run inference with a trained model, you can use the provided sampling script:

```bash
bash scripts/sampling_LDM.sh
```

You should modify the script to point to your trained model checkpoint:
```bash
python scripts/sample_LDM.py \
    -r path/to/LDM/checkpoint.ckpt \
    --n_samples 1 \
    --batch_size 48 \
    --steps 2
```

Parameters for sampling:
- `-r, --resume`: Path to the model checkpoint
- `--n_samples`: Number of samples to generate per input
- `--batch_size`: Batch size for inference
- `--steps`: Number of DDIM sampling steps (more steps = higher quality but slower)

The sampling script will output the synthesized DWI images in NIFTI format in a directory structure organized by sample ID.

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

<!-- ```bibtex
@inproceedings{lee2025lapt,
  title={Lesion-Aware Post-Training of Latent Diffusion Models for Synthesizing Diffusion MRI from CT Perfusion},
  author={Lee, Junhyeok and Kim, Hyunwoong and Chung, Hyungjin and Eom, Heeseong and Jang, Joon and Sohn, Chul-Ho and Choi, Kyu Sung},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2025},
  organization={Springer},
  note={$^\star$J. Lee and H. Kim contributed equally to this paper. $^\dag$K.S. Choi is the corresponding author.}
}
``` -->
*(BibTeX entry will be updated upon publication)*

## 🙏 Acknowledgements

We would like to thank the authors of the following repositories for their valuable contributions:
- [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) for their VQGAN implementation
- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) for their Latent Diffusion Models framework