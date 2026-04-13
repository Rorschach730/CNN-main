import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from denoiser_new import Denoiser
from util.pet_dataset_new import PETDenoisingDataset


class InferenceConfig:
    device = "cuda"
    checkpoint_path = "./output_dir_new/checkpoint-330.pth"  # [物理警告] 点火后，请根据实际训练进度修改此处
    data_folder = "./processed_data_3d_osem/val"
    img_size = 128

    attn_dropout = 0.0
    proj_dropout = 0.0
    P_mean = -0.5
    P_std = 1.2
    patch_size = 8
    cond_drop_prob = 0.1
    cfg_scale = 0.6
    accum_iter = 4
    sampling_steps = 50  # 锁定 Heun 二阶积分 50 步

    use_roi = True  # 锁定 True：启动核医学临床指标；False：启动 CV 通用指标


def get_mock_args():
    class MockArgs: pass

    args = MockArgs()
    args.img_size = InferenceConfig.img_size
    args.attn_dropout = InferenceConfig.attn_dropout
    args.proj_dropout = InferenceConfig.proj_dropout
    args.P_mean = InferenceConfig.P_mean
    args.P_std = InferenceConfig.P_std
    args.accum_iter = InferenceConfig.accum_iter
    args.patch_size = InferenceConfig.patch_size
    args.cond_drop_prob = InferenceConfig.cond_drop_prob
    args.cfg_scale = InferenceConfig.cfg_scale
    return args


def calculate_snr(clean, test):
    var_clean = np.var(clean)
    var_noise = np.var(clean - test)
    if var_noise == 0: return float('inf')
    return 10 * np.log10(var_clean / var_noise)


def calculate_clinical_metrics(gt_roi, pred_roi):
    """
    执行无掩膜状态下的临床代谢逼近量化
    包含：SUVmax偏差、SUVmean MAPE、高频对比度恢复度(CR)
    """
    eps = 1e-8

    # 方案 B: 绝对误差法
    max_gt = gt_roi.max()
    max_pred = pred_roi.max()
    bias_suvmax = ((max_pred - max_gt) / (max_gt + eps)) * 100.0

    mean_gt = gt_roi.mean()
    mean_pred = pred_roi.mean()
    mape_suvmean = (np.abs(mean_pred - mean_gt) / (mean_gt + eps)) * 100.0

    # 方案 A: 高频代谢区阈值逼近法 (Top 5%)
    thresh95 = np.percentile(gt_roi, 95)
    hotspot_mask = gt_roi >= thresh95
    mean_hotspot_gt = gt_roi[hotspot_mask].mean()
    mean_hotspot_pred = pred_roi[hotspot_mask].mean()

    cr = (mean_hotspot_pred / (mean_hotspot_gt + eps)) * 100.0

    return bias_suvmax, mape_suvmean, cr


def robust_windowing(img_gt):
    valid_pixels = img_gt[img_gt > 0.01]
    if len(valid_pixels) == 0: return 0.0, 1.0
    vmax = np.percentile(valid_pixels, 99)
    return 0.0, vmax


def get_roi_bbox(img_raw, threshold_ratio=0.05, padding=4):
    threshold = img_raw.max() * threshold_ratio
    rows = np.any(img_raw > threshold, axis=1)
    cols = np.any(img_raw > threshold, axis=0)
    if not np.any(rows) or not np.any(cols): return 0, img_raw.shape[0], 0, img_raw.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return max(0, rmin - padding), min(img_raw.shape[0], rmax + padding), max(0, cmin - padding), min(img_raw.shape[1],
                                                                                                      cmax + padding)


def main():
    device = torch.device(InferenceConfig.device)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    print(f"--- Check Inference on {device} (use_roi={InferenceConfig.use_roi}) ---")

    args = get_mock_args()
    model = Denoiser(args)
    model.to(device)

    print(f"[*] Loading checkpoint: {InferenceConfig.checkpoint_path}")
    try:
        ckpt = torch.load(InferenceConfig.checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[!] 致命错误：找不到权重文件 {InferenceConfig.checkpoint_path}，请先执行训练。")
        return

    state_dict = ckpt.get('model_ema', ckpt['model'])
    new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.net.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dataset = PETDenoisingDataset(InferenceConfig.data_folder, img_size=InferenceConfig.img_size)
    if len(dataset) == 0:
        print("[!] 测试集为空，请检查路径。")
        return

    idx = random.randint(0, len(dataset) - 1)
    #idx = 39
    for _ in range(20):
        target, condition = dataset[idx]
        if target.max() > 0.05: break
        idx = random.randint(0, len(dataset) - 1)

    target_t = target.unsqueeze(0).to(device)
    condition_t = condition.unsqueeze(0).to(device)

    print(f"[*] Generating Heun 2nd-Order... (Steps: {InferenceConfig.sampling_steps}, CFG: {InferenceConfig.cfg_scale})")
    with torch.no_grad():
        output_tensor = model.generate(condition_t, steps=InferenceConfig.sampling_steps, cfg_scale=InferenceConfig.cfg_scale)

    # 1. 提取未经任何截断的纯正物理张量 (专用于核医学量化计算)
    out_raw = output_tensor.squeeze().cpu().numpy()
    gt_raw = target_t.squeeze().cpu().numpy()
    in_raw = condition_t.squeeze().cpu().numpy()

    # 2. 提取截断张量 (仅用于生成外围 Bbox 轮廓及最终的画图可视化)
    out_01 = np.clip(out_raw, 0.0, 1.0)
    gt_01 = np.clip(gt_raw, 0.0, 1.0)
    in_01 = np.clip(in_raw, 0.0, 1.0)

    rmin, rmax, cmin, cmax = get_roi_bbox(gt_01)

    if InferenceConfig.use_roi:
        # [物理修正]：必须使用 raw 数据进入临床指标运算，严禁削平热点！
        eval_gt = gt_raw[rmin:rmax, cmin:cmax]
        eval_in = in_raw[rmin:rmax, cmin:cmax]
        eval_out = out_raw[rmin:rmax, cmin:cmax]
        title_suffix = "ROI Clinical Metrics"

        bias_in, mape_in, cr_in = calculate_clinical_metrics(eval_gt, eval_in)
        bias_out, mape_out, cr_out = calculate_clinical_metrics(eval_gt, eval_out)

        str_in = f"Noisy Input ({title_suffix})\nBias_max: {bias_in:.1f}% | MAPE_mean: {mape_in:.1f}% | CR: {cr_in:.1f}%"
        str_out = f"JiT-Heun 50 steps ({title_suffix})\nBias_max: {bias_out:.1f}% | MAPE_mean: {mape_out:.1f}% | CR: {cr_out:.1f}%"
    else:
        # CV 通用指标仍沿用 [0, 1] 截断数据以符合 skimage 库的计算标准
        eval_gt, eval_in, eval_out = gt_01, in_01, out_01
        title_suffix = "Whole Image Metrics"

        psnr_in = compare_psnr(eval_gt, eval_in, data_range=1.0)
        ssim_in = compare_ssim(eval_gt, eval_in, data_range=1.0)
        snr_in = calculate_snr(eval_gt, eval_in)

        psnr_out = compare_psnr(eval_gt, eval_out, data_range=1.0)
        ssim_out = compare_ssim(eval_gt, eval_out, data_range=1.0)
        snr_out = calculate_snr(eval_gt, eval_out)

        str_in = f"Noisy Input ({title_suffix})\nPSNR: {psnr_in:.2f} | SSIM: {ssim_in:.3f} | SNR: {snr_in:.2f}"
        str_out = f"JiT-Heun 50 steps ({title_suffix})\nPSNR: {psnr_out:.2f} | SSIM: {ssim_out:.3f} | SNR: {snr_out:.2f}"

    vmin, vmax = robust_windowing(gt_01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(in_01, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].add_patch(plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
    axes[0].set_title(str_in)
    axes[0].axis('off')

    axes[1].imshow(out_01, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].add_patch(plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
    axes[1].set_title(str_out)
    axes[1].axis('off')

    axes[2].imshow(gt_01, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].add_patch(plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
    axes[2].set_title("Ground Truth (Clean)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()