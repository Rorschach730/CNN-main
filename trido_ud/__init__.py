"""
TriDo-JiT: Triple-Domain Denoising with Joint-in-Time Flow Matching
====================================================================
Three domains:
  1. Sinogram Domain — sino encoder with body-part conditioning (Cross-Domain Reconstruction)
  2. Image Domain     — JiT transformer-based refinement (Flow Matching v-prediction)
  3. Frequency Domain — GFP high-frequency enhancement (DCT-based)

References:
  - Prior Knowledge-Guided Triple-Domain Transformer-GAN for Direct PET Reconstruction
  - Cross-Domain Reconstruction (Sinogram domain processing)
  - SiT / Lightning-DiT (JiT backbone)
"""
