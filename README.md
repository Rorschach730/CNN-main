## TriDo-CNN: Triple-Domain CNN for PET Denoising

[![Architecture](https://img.shields.io/badge/Backbone-ResNet%20U--Net%20CNN-blue)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)]()

<p align="center">
  <b>Triple-Domain Architecture: Sinogram + Image + Frequency</b><br>
  Domain ② Backbone: ResNet U-Net CNN (upgraded from JiT Transformer)
</p>

---

This is a PyTorch implementation of TriDo-CNN, a triple-domain PET denoising framework for low-dose PET image quality enhancement. The project is forked from JiT-main (Li & He, 2025), with the core architectural change:

> **Domain ② Image Backbone: JiT Transformer → ResNet U-Net CNN**

The CNN backbone provides better inductive bias for medical image denoising, lower memory footprint, and proven effectiveness on PET/CT reconstruction tasks.

### Three-Domain Architecture

```
  ┌─────────────────────────────────────────────────────────────┐
  │  ① Sinogram Domain   ② Image Domain     ③ Frequency Domain  │
  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
  │  │ Radon → Sino │→  │ FBP → ResNet │→  │ GFP Enhance  │→   │
  │  │   Encoder    │   │   U-Net CNN  │   │   Module     │    │
  │  └──────────────┘   └──────────────┘   └──────────────┘    │
  └─────────────────────────────────────────────────────────────┘
```

1. **Sinogram Domain** — Body-part conditioned sino encoder (FiLM modulation)
2. **Image Domain** — ResNet U-Net CNN with FiLM conditioning (4-stage encoder-decoder)
3. **Frequency Domain** — GFP high-frequency enhancement (DCT-based)

### Key Features

- Flow Matching v-prediction for stable denoising
- Body part embedding for anatomy-adaptive processing
- FGW (Fused Gromov-Wasserstein) structural regularization
- HALO compound frequency loss (FFT + DWT)
- Classifier-Free Guidance (CFG) with adaptive NFE
- AMP (bfloat16) training with gradient accumulation
- EMA weight tracking

### Model Variants

| Model | Base Channels | ResBlocks/Stage | ~Params |
|-------|--------------|-----------------|---------|
| TriDoCNN-Small | 32 | 1 | ~3M |
| TriDoCNN-Base | 64 | 2 | ~14M |
| TriDoCNN-Large | 128 | 3 | ~52M |

### Training

```bash
# Full triple-domain training
python trido_ud/main_trido.py --data_path ./processed_data_trido --output_dir ./trido_output

# Image-only (ablation: no sino domain)
python trido_ud/main_trido.py --no_sino_domain

# Small model for quick experiments
python trido_ud/main_trido.py --model_size Small
```

### Dataset

The dataset follows the TriDo format:
- v4 .pt files: `[3, H, W]` tensors with `[body_part, condition, target]`
- Legacy .pt files: `[2, H, W]` tensors with `[condition, target]`

### Inference

```bash
python trido_ud/infer_single.py --ckpt_path ./trido_output/checkpoint-final.pth \
    --data_path ./processed_data_trido/test --cfg_scale 0.6
```

### Architecture Change from JiT-main

| Component | JiT-main | CNN-main |
|-----------|----------|----------|
| Domain ② Backbone | JiT Transformer (Attention) | ResNet U-Net (Convolution) |
| Position Encoding | RoPE + Sinusoidal | Implicit (CNN spatial) |
| Patch Processing | Patchify → Tokens | Direct 2D Convolution |
| Conditioning | adaLN (6-dim modulation) | FiLM (per-layer scale+shift) |
| Memory Complexity | O(N²) (attention) | O(N) (convolution) |

### References

- Li, T. & He, K. (2025). "Back to Basics: Let Denoising Generative Models Denoise." arXiv:2511.13720
- He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016
- Ronneberger, O. et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015

### Acknowledgements

Forked from JiT-main (Rorschach730). Original JiT implementation by Li & He (MIT).
