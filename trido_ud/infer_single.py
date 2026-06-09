"""
TriDo-JiT 极速单张多域推理 (Single Slice Inference)
===================================================
彻底修复:
1. 强制调用模型原生的 generate() (Heun + CFG)。
2. 引入 99% 动态阈值截断，彻底解决 PET 热像素导致的“全黑”现象。
3. 新增量化指标评估 (PSNR, SSIM, NMSE) 并在图像上直观展示。
"""

import os
import torch
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from skimage.transform import radon
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[!] 警告: 未安装 scikit-image，无法生成弦图和计算 SSIM。请执行 pip install scikit-image")

from trido_ud.pet_dataset_trido import TriDoPETDataset
from trido_ud.denoiser_trido import TriDoDenoiser

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./trido_output/checkpoint-final.pth')
    parser.add_argument('--data_path', type=str, default='I:/processed_data_trido/test')
    parser.add_argument('--output_dir', type=str, default='./inference_results_multidomain')
    parser.add_argument('--num_samples', type=int, default=1)

    # 架构必须与训练对齐
    parser.add_argument('--model_size', type=str, default='Large')
    parser.add_argument('--use_sino_domain', action='store_true', default=True)
    parser.add_argument('--use_freq_domain', action='store_true', default=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)

    # CFG 参数
    parser.add_argument('--cfg_scale', type=float, default=0.6)
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--P_mean', type=float, default=-0.5)
    parser.add_argument('--P_std', type=float, default=1.2)

    parser.add_argument('--nfe', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def get_freq_mag(img_np):
    f = np.fft.fft2(img_np)
    return np.log(np.abs(np.fft.fftshift(f)) + 1e-8)

def get_sino(img_np):
    if not HAS_SKIMAGE: return np.zeros_like(img_np)
    img_np = np.clip(img_np, 0, None)
    return radon(img_np, theta=np.linspace(0., 180., max(img_np.shape), endpoint=False), circle=True)

def calculate_metrics(pred, target):
    """计算核心图像去噪指标: PSNR (越高越好), SSIM (越高越好), NMSE (越低越好)"""
    p = pred.astype(np.float64)
    t = target.astype(np.float64)

    # 获取图像动态范围以防止除以零
    data_range = t.max() - t.min()
    if data_range == 0:
        data_range = 1.0

    # 计算均方误差 (MSE) 和 归一化均方误差 (NMSE)
    mse = np.mean((p - t) ** 2)
    t_var = np.mean(t ** 2)
    nmse = mse / t_var if t_var > 0 else float('inf')

    # 计算峰值信噪比 (PSNR)
    psnr_val = 20 * np.log10(data_range / np.sqrt(mse)) if mse > 0 else float('inf')

    # 计算结构相似度 (SSIM)
    if HAS_SKIMAGE:
        ssim_val = ssim(p, t, data_range=data_range)
    else:
        ssim_val = 0.0

    return psnr_val, ssim_val, nmse

def get_robust_vmax(img_np, percentile=99.5):
    """鲁棒阈值: 过滤极高亮热像素(如膀胱)，防止画面全黑"""
    vmax = np.percentile(img_np[img_np > 0], percentile) if (img_np > 0).any() else 1.0
    return max(vmax, 1e-3)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = TriDoPETDataset(args.data_path, img_size=args.img_size)
    model = TriDoDenoiser(args).to(device)

    ckpt = torch.load(args.ckpt_path, map_location='cuda', weights_only=False)

    # 强制将权重加载到实际进行前向传播的 model.net 中
    if 'model_ema' in ckpt:
        try:
            model.net.load_state_dict(ckpt['model_ema'], strict=False)
            print("[*] 成功将 EMA 权重灌入推理网络！")
        except:
            model.net.load_state_dict(ckpt['model'])
            print("[*] EMA 不兼容，成功加载常规权重！")
    else:
        model.net.load_state_dict(ckpt['model'])
        print("[*] 成功加载常规权重！")

    model.eval()

    idx = random.randint(0, len(dataset) - 1)
    print(f"[*] 开始极速推理单张切片 (Dataset ID: {idx})...")

    target, condition, body_part = dataset[idx]

    target_tensor = target.unsqueeze(0).to(device)
    condition_tensor = condition.unsqueeze(0).to(device)
    bp_tensor = body_part.unsqueeze(0).to(device)

    # 【核心修复 1】直接调用你原生自带的 generate 方法！(含 Heun + CFG)
    with torch.no_grad():
        if hasattr(model, 'generate'):
            pred_tensor = model.generate(condition_tensor, bp_tensor, steps=args.nfe, cfg_scale=args.cfg_scale)
        else:
            raise RuntimeError("在模型中未找到 generate() 方法！")

    # 截断负数漂移
    img_cond = np.clip(condition_tensor.squeeze().cpu().numpy(), 0, None)
    img_pred = np.clip(pred_tensor.squeeze().cpu().numpy(), 0, None)
    img_targ = np.clip(target_tensor.squeeze().cpu().numpy(), 0, None)

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))

    # 【核心修复 2】使用 99.5% 百分位数作为阈值，防止热点导致全黑
    vmax = get_robust_vmax(img_targ, 99.5)

    # ==========================================
    # 计算医学图像评估指标
    # ==========================================
    cond_psnr, cond_ssim, cond_nmse = calculate_metrics(img_cond, img_targ)
    pred_psnr, pred_ssim, pred_nmse = calculate_metrics(img_pred, img_targ)

    # 1. Image Domain
    axes[0, 0].imshow(img_cond, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 0].set_title(f"Input (Low Dose)\\nPSNR: {cond_psnr:.2f} | SSIM: {cond_ssim:.4f} | NMSE: {cond_nmse:.4f}")

    axes[0, 1].imshow(img_pred, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 1].set_title(f"Denoised (TriDo-JiT)\\nPSNR: {pred_psnr:.2f} | SSIM: {pred_ssim:.4f} | NMSE: {pred_nmse:.4f}", color='green')

    axes[0, 2].imshow(img_targ, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 2].set_title("Target (Full Dose Image)\\nReference")

    # 2. Freq Domain
    f_cond, f_pred, f_targ = get_freq_mag(img_cond), get_freq_mag(img_pred), get_freq_mag(img_targ)
    fvmax, fvmin = f_targ.max() * 0.95, f_targ.min()
    axes[1, 0].imshow(f_cond, cmap='magma', vmin=fvmin, vmax=fvmax); axes[1, 0].set_title("Input (Freq Spectrum)")
    axes[1, 1].imshow(f_pred, cmap='magma', vmin=fvmin, vmax=fvmax); axes[1, 1].set_title("Denoised (Freq Spectrum)", color='green')
    axes[1, 2].imshow(f_targ, cmap='magma', vmin=fvmin, vmax=fvmax); axes[1, 2].set_title("Target (Freq Spectrum)")

    # 3. Sino Domain
    s_cond, s_pred, s_targ = get_sino(img_cond), get_sino(img_pred), get_sino(img_targ)
    svmax = get_robust_vmax(s_targ, 99.5)
    axes[2, 0].imshow(s_cond, cmap='gray', vmin=0, vmax=svmax, aspect='auto'); axes[2, 0].set_title("Input (Sinogram)")
    axes[2, 1].imshow(s_pred, cmap='gray', vmin=0, vmax=svmax, aspect='auto'); axes[2, 1].set_title("Denoised (Sinogram)", color='green')
    axes[2, 2].imshow(s_targ, cmap='gray', vmin=0, vmax=svmax, aspect='auto'); axes[2, 2].set_title("Target (Sinogram)")

    for ax in axes.flatten(): ax.axis('off')

    bp_name = {0: "Brain", 1: "Chest", 2: "Abdomen"}.get(body_part.item(), "Unknown")

    # 动态显示提升效果
    psnr_gain = pred_psnr - cond_psnr
    ssim_gain = pred_ssim - cond_ssim
    nmse_drop = cond_nmse - pred_nmse

    plt.suptitle(f"TriDo-JiT (Fast Single View) | ID: {idx} | Anatomy: {bp_name}\n"
                 f"Improvement: PSNR +{psnr_gain:.2f}dB | SSIM +{ssim_gain:.4f} | NMSE -{nmse_drop:.4f}",
                 fontsize=16, fontweight='bold', y=0.98, color='blue')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_file = os.path.join(args.output_dir, f"fast_result_{idx}.png")
    plt.savefig(save_file, dpi=150)
    plt.close()

    print(f"[√] 抽查完成！图片已保存至: {os.path.abspath(save_file)}")
    print(f"  -> 指标提升: PSNR +{psnr_gain:.2f}dB, SSIM +{ssim_gain:.4f}, NMSE -{nmse_drop:.4f}")

if __name__ == '__main__':
    main()