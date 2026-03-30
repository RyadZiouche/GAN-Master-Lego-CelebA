# GAN-Master-Lego-CelebA
# GAN Master — Lego Bricks & CelebA Faces

> A progressive deep learning project exploring **Generative Adversarial Networks** through three architectures of increasing complexity, applied to two different image domains.

---

## Overview

This repository implements three GAN variants from scratch using **PyTorch**, trained on two real-world datasets: Lego brick images and the CelebA face dataset. Each model builds on the limitations of the previous one, illustrating the practical evolution of GAN training techniques.

| Module | Architecture | Dataset | Task |
|--------|-------------|---------|------|
| `01_DCGAN_Lego` | DCGAN | Lego Bricks (Kaggle) | Unconditional image generation |
| `02_WGAN-GP_Faces` | WGAN-GP | CelebA | Stable face generation with Wasserstein loss |
| `03_Conditional_WGAN_GP` | Conditional WGAN-GP | CelebA | Attribute-controlled face generation |

---

## Architecture Progression

### 01 — DCGAN on Lego Bricks

A **Deep Convolutional GAN** trained to generate 64×64 grayscale images of Lego bricks.

- Convolutional generator and discriminator (no fully connected layers)
- Batch normalization for training stability
- Latent space dimension: `Z = 100`
- Training objective: minimax binary cross-entropy loss

**Key challenge addressed:** learning spatial structure from unlabeled image data with no class supervision.

---

### 02 — WGAN-GP on CelebA Faces

Replaces the standard discriminator with a **Wasserstein critic** and adds a **Gradient Penalty** to enforce the Lipschitz constraint.

- Critic outputs a real-valued score (not a probability)
- **Wasserstein loss:** maximizes `E[critic(real)] - E[critic(fake)]`
- **Gradient Penalty (λ=10):** penalizes deviations of the critic's gradient norm from 1 on interpolated samples
- Multi-step critic training (`n_critic` iterations per generator step)

**Key improvement over DCGAN:** eliminates mode collapse and training instability by replacing JS divergence with the Earth Mover's Distance.

---

### 03 — Conditional WGAN-GP on CelebA

Extends WGAN-GP with **class conditioning** to enable attribute-guided generation.

- Label information (e.g. `Blond_Hair`: 0 or 1) is injected into both the generator and the critic
- The critic jointly evaluates image realism *and* label consistency
- Enables targeted generation: produce faces with a specific attribute on demand
- Latent dimension reduced to `Z = 32` for faster convergence

**Key improvement over WGAN-GP:** moves from unconditional to *controllable* generation.

---

## Project Structure

```
GAN-Master-Lego-CelebA/
│
├── 01_DCGAN_Lego/            # DCGAN — Lego brick image generation
│
├── 02_WGAN-GP_Faces/         # WGAN-GP — Stable face generation
│
├── 03_Conditional_WGAN_GP/   # Conditional WGAN-GP — Attribute-controlled generation
│
├── weights/                  # Saved model checkpoints
│
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib tqdm numpy kaggle
```

### Datasets

**Lego Bricks (Module 01)**
```bash
pip install kaggle
kaggle datasets download -d joosthazelzet/lego-brick-images --unzip
```

**CelebA (Modules 02 & 03)**

Download from the [official source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or via `torchvision.datasets.CelebA`.

### Training

Navigate to the relevant module directory and run the notebook or training script:

```bash
cd 01_DCGAN_Lego
jupyter notebook
```

---

## Key Hyperparameters

| Parameter | DCGAN | WGAN-GP | Cond. WGAN-GP |
|-----------|-------|---------|----------------|
| Image size | 64×64 | 64×64 | 64×64 |
| Channels | 1 (grayscale) | 3 (RGB) | 3 (RGB) |
| Latent dim (Z) | 100 | 128 | 32 |
| Batch size | 128 | 64 | 128 |
| Critic steps | — | 5 | 3 |
| GP weight (λ) | — | 10 | 10 |
| Optimizer | Adam | Adam | Adam (β₁=0.5) |

---

## Concepts Covered

- Convolutional generator & discriminator design
- Batch normalization and activation function choices (ReLU, LeakyReLU, Tanh)
- Wasserstein distance vs. Jensen-Shannon divergence
- Gradient Penalty and Lipschitz constraint enforcement
- Conditional generation via label embedding
- Mode collapse diagnosis and mitigation

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

---

## Author

**Ryad Ziouche**
[GitHub](https://github.com/RyadZiouche)