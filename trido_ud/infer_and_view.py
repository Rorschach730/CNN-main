"""
TriDo-CNN 推理与多域可视化脚本 (Inference & Multi-Domain Visualization)
====================================================================
此脚本生成 3x3 对比图，全面展示模型在 图像域、频域(Fourier) 和 弦图域(Sinogram) 的表现。
"""

import os
import torch
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端生成图片
import matplotlib.pyplot as plt

try:
    from skimage.transform import radon
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[!] 警告: 未安装 scikit-image，无法生成弦图(Sinogram)。请执行 pip install scikit-image")

# 导入你的数据集和模型架构
from trido_ud.pet_dataset_trido import TriDoPETDataset
from trido_ud.denoiser_trido import TriDoDenoiser

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./trido_output/checkpoint-50.pth', help='权重文件路径')
    parser.add_argument('--data_path', type=str, default='I:/processed_data_trido/test', help='测试/验证集路径')
    parser.add_argument('--output_dir', type=str, default='./inference_results_multidomain', help='输出图片保存路径')

    parser.add_argument('--model_size', type=str, default='Large')
    parser.add_argument('--use_sino_domain', action='store_true', default=True)
    parser.add_argument('--use_freq_domain', action='store_true', default=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)

    parser.add_argument('--num_samples', type=int, default=5, help='要随机抽查的切片数量')
    parser.add_argument('--nfe', type=int, default=50, help='ODE 求解的函数评估次数 (步数)')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def get_freq_mag(img_np):
    """计算图像的频域幅度谱 (Log Magnitude)"""
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    # 取对数以压缩动态范围，便于人眼观察
    mag = np.log(np.abs(fshift) + 1e-8)
    return mag

def get_sino(img_np):
    """计算图像的物理投影弦图 (Sinogram)"""
    if not HAS_SKIMAGE:
        return np.zeros_like(img_np)
    # 模拟 180 度的探测器环绕采集
    theta = np.linspace(0., 180., max(img_np.shape), endpoint=False)
    sinogram = radon(img_np, theta=theta, circle=True)
    return sinogram

@torch.no_grad()
def sample_flow_matching(model_net, condition, body_part, nfe=50, device='cuda'):
    """Flow Matching 欧拉常微分方程 (ODE) 求解器"""
    model_net.eval()
    B, C, H, W = condition.shape
    x = torch.randn((B, 1, H, W), device=device)
    dt = 1.0 / nfe

    for i in range(nfe):
        t_val = i / nfe
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        model_input = torch.cat([x, condition], dim=1)
        v_pred = model_net(model_input, t_tensor, body_part=body_part)
        x = x + v_pred * dt

    return x

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[*] 正在加载验证集数据: {args.data_path}")
    dataset = TriDoPETDataset(args.data_path, img_size=args.img_size)

    print(f"[*] 正在初始化模型并加载权重...")
    model = TriDoDenoiser(args)
    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)

    if 'model_ema' in ckpt:
        print("[*] 成功提取 EMA 平滑权重！")
        model.load_ema_state_dict(ckpt['model_ema'])
        net = model.ema_net if hasattr(model, 'ema_net') else model.net
    else:
        model.net.load_state_dict(ckpt['model'])
        net = model.net

    net.to(device)
    net.eval()

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    for idx_count, idx in enumerate(indices):
        print(f"  -> 正在生成多域切片 {idx_count+1}/{len(indices)} (Dataset ID: {idx})...")
        target, condition, body_part = dataset[idx]

        target_tensor = target.unsqueeze(0).to(device)
        condition_tensor = condition.unsqueeze(0).to(device)
        bp_tensor = body_part.unsqueeze(0).to(device)

        if hasattr(model, 'sample'):
            pred_tensor = model.sample(condition_tensor, body_part=bp_tensor, num_steps=args.nfe)
        else:
            pred_tensor = sample_flow_matching(net, condition_tensor, bp_tensor, nfe=args.nfe, device=device)

        # --- 数据准备 ---
        img_cond = condition_tensor.squeeze().cpu().numpy()
        img_pred = pred_tensor.squeeze().cpu().numpy()
        img_targ = target_tensor.squeeze().cpu().numpy()

        # --- 绘图布局 (3x3) ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        # 1. 图像域 (Spatial Domain)
        vmax = img_targ.max() * 0.8
        axes[0, 0].imshow(img_cond, cmap='gray', vmin=0, vmax=vmax)
        axes[0, 0].set_title(f"Low Dose Input (Image)", fontsize=16)
        axes[0, 1].imshow(img_pred, cmap='gray', vmin=0, vmax=vmax)
        axes[0, 1].set_title(f"TriDo-CNN Denoised (Image)", fontsize=16, color='green')
        axes[0, 2].imshow(img_targ, cmap='gray', vmin=0, vmax=vmax)
        axes[0, 2].set_title(f"Full Dose Target (Image)", fontsize=16)

        # 2. 频域 (Frequency Domain - Fourier)
        freq_cond = get_freq_mag(img_cond)
        freq_pred = get_freq_mag(img_pred)
        freq_targ = get_freq_mag(img_targ)

        f_vmax = freq_targ.max() * 0.95
        f_vmin = freq_targ.min()
        axes[1, 0].imshow(freq_cond, cmap='magma', vmin=f_vmin, vmax=f_vmax)
        axes[1, 0].set_title(f"Frequency Spectrum (Noise scatter)", fontsize=16)
        axes[1, 1].imshow(freq_pred, cmap='magma', vmin=f_vmin, vmax=f_vmax)
        axes[1, 1].set_title(f"Frequency Spectrum (Cleaned)", fontsize=16, color='green')
        axes[1, 2].imshow(freq_targ, cmap='magma', vmin=f_vmin, vmax=f_vmax)
        axes[1, 2].set_title(f"Frequency Spectrum (GT)", fontsize=16)

        # 3. 弦图域 (Sinogram Domain - Radon Transform)
        sino_cond = get_sino(img_cond)
        sino_pred = get_sino(img_pred)
        sino_targ = get_sino(img_targ)

        s_vmax = sino_targ.max() * 0.8
        axes[2, 0].imshow(sino_cond, cmap='gray', vmin=0, vmax=s_vmax, aspect='auto')
        axes[2, 0].set_title(f"Sinogram Projection (Streak artifacts)", fontsize=16)
        axes[2, 1].imshow(sino_pred, cmap='gray', vmin=0, vmax=s_vmax, aspect='auto')
        axes[2, 1].set_title(f"Sinogram Projection (Restored)", fontsize=16, color='green')
        axes[2, 2].imshow(sino_targ, cmap='gray', vmin=0, vmax=s_vmax, aspect='auto')
        axes[2, 2].set_title(f"Sinogram Projection (GT)", fontsize=16)

        for ax in axes.flatten(): ax.axis('off')

        bp_name = {0: "Brain", 1: "Chest", 2: "Abdomen"}.get(body_part.item(), "Unknown")
        plt.suptitle(f"TriDo-CNN Multi-Domain Inference | Slice ID: {idx} | Anatomy: {bp_name}", fontsize=20, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_file = os.path.join(args.output_dir, f"result_multidomain_{idx}.png")
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n[√] 成功！多域对比图已保存在: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()