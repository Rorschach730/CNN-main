"""
TriDo-Simple: 三域简化解耦 CNN 去噪（无扩散模型）
==================================================
三域级联 (Sinogram → Image → Frequency)，每域独立 ResNet。
纯前馈，无 Flow Matching、无 ODE、无 CFG。

与 trido_ud/ (扩散版) 的区别:
  - 无 timestep 采样
  - 无辅助损失 (FGW/HALO/GFP freq/sino consistency)
  - 训练: 纯 L1 loss
  - 推理: model(condition) 一次前馈

文件:
  model_simple.py    — TriDoSimpleCNN + 子网络
  denoiser_simple.py — SimpleDenoiser 包装器
  engine_simple.py   — train_one_epoch
  main_simple.py     — 训练入口
"""
