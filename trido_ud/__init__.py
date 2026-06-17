"""
TriDo-CNN: Triple-Domain Denoising with CNN Flow Matching
==========================================================
Three domains:
  1. Sinogram Domain — sino encoder with body-part conditioning (Cross-Domain Reconstruction)
  2. Image Domain     — ResNet U-Net CNN refinement (Flow Matching v-prediction)
  3. Frequency Domain — GFP high-frequency enhancement (DCT-based)

Architecture change from JiT-main:
  Domain ②: JiT Transformer (Attention-based) → ResNet U-Net CNN (Convolution-based)

References:
  - Prior Knowledge-Guided Triple-Domain Transformer-GAN
  - Cross-Domain Reconstruction (Sinogram domain processing)
  - ResNet / U-Net (CNN backbone)
"""
