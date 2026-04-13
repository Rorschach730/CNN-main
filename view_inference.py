import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from denoiser_new import Denoiser
from util.pet_dataset_new import PETDenoisingDataset
import torch.nn.functional as F


class InferenceConfig:
    device = "cuda"
    checkpoint_path = "./output_dir_new/checkpoint-330.pth"  # [物理警告] 点火后，请根据实际训练进度修改此处
    data_folder = "./processed_data_3d_osem/test"
    img_size = 128

    # [物理控制锚点]: 定点病人与定点层数配置
    target_patient = "Lung_Dx-A0233_M_216.8MBq_Body"  # 目标病人 3D npy 文件名本体
    target_slice = "Z253"  # 目标层数ID

    attn_dropout = 0.0
    proj_dropout = 0.0
    P_mean = -0.5
    P_std = 1.2
    patch_size = 8
    cond_drop_prob = 0.1
    cfg_scale = 0.6
    accum_iter = 4
    sampling_steps = 50  # 锁定 Heun 二阶积分 50 步

    use_roi = False  # 锁定 True：启动核医学临床指标；False：启动 CV 通用指标


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
    eps = 1e-8
    max_gt = gt_roi.max()
    max_pred = pred_roi.max()
    bias_suvmax = ((max_pred - max_gt) / (max_gt + eps)) * 100.0

    mean_gt = gt_roi.mean()
    mean_pred = pred_roi.mean()
    mape_suvmean = (np.abs(mean_pred - mean_gt) / (mean_gt + eps)) * 100.0

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

    target_patient = InferenceConfig.target_patient
    target_slice = InferenceConfig.target_slice

    try:
        slice_idx = int(target_slice.replace("Z", ""))
    except ValueError:
        print(f"[!] 致命错误：无法解析目标层数 {target_slice}，请确保格式形如 'Z039'")
        return

    target_filename = f"{target_patient}.npy"
    target_path = os.path.join(InferenceConfig.data_folder, target_filename)

    if not os.path.exists(target_path):
        print(f"[!] 致命错误：在 {InferenceConfig.data_folder} 目录下未能找到病人 3D 卷文件 [{target_filename}]")
        return

    print(f"[*] 成功锁定 3D 物理卷文件: {target_path}")
    print(f"[*] 正在双线解构字典并强行切出第 {slice_idx} 层...")

    data_obj = np.load(target_path, allow_pickle=True).item()

    cond_vol = None
    tgt_vol = None

    if isinstance(data_obj, dict):
        # 嗅探输入端与真值端
        for k in ['input', 'low', 'condition']:
            if k in data_obj: cond_vol = data_obj[k]; break
        for k in ['target', 'full', 'label', 'gt']:
            if k in data_obj: tgt_vol = data_obj[k]; break

    if cond_vol is None:
        print(
            f"[!] 致命错误：在字典中未找到低剂量输入。可用键值: {list(data_obj.keys()) if isinstance(data_obj, dict) else '非字典'}")
        return

    if slice_idx >= cond_vol.shape[0]:
        print(f"[!] 越界错误：请求的层数 {slice_idx} 超过了物理卷最大厚度 {cond_vol.shape[0] - 1}")
        return

    cond_arr = cond_vol[slice_idx, :, :]

    if tgt_vol is not None:
        tgt_arr = tgt_vol[slice_idx, :, :]
    else:
        print(f"[*] 警告：未在物理字典中探测到 full-dose，采用 condition 作为 target 占位（临床对比将失效）。")
        tgt_arr = np.copy(cond_arr)

    # 张量化并升维至网络可用格式 [1, 1, H, W]
    cond_t = torch.from_numpy(cond_arr).unsqueeze(0).unsqueeze(0).float()
    tgt_t = torch.from_numpy(tgt_arr).unsqueeze(0).unsqueeze(0).float()

    # 物理分辨率强锁对齐
    if cond_t.shape[-1] != InferenceConfig.img_size:
        cond_t = F.interpolate(cond_t, size=(InferenceConfig.img_size, InferenceConfig.img_size), mode='bilinear')
        tgt_t = F.interpolate(tgt_t, size=(InferenceConfig.img_size, InferenceConfig.img_size), mode='bilinear')

    condition_t = cond_t.to(device)
    target_t = tgt_t.to(device)

    print(
        f"[*] Generating Heun 2nd-Order... (Steps: {InferenceConfig.sampling_steps}, CFG: {InferenceConfig.cfg_scale})")
    with torch.no_grad():
        output_tensor = model.generate(condition_t, steps=InferenceConfig.sampling_steps,
                                       cfg_scale=InferenceConfig.cfg_scale)

    out_raw = output_tensor.squeeze().cpu().numpy()
    gt_raw = target_t.squeeze().cpu().numpy()
    in_raw = condition_t.squeeze().cpu().numpy()

    out_01 = np.clip(out_raw, 0.0, 1.0)
    gt_01 = np.clip(gt_raw, 0.0, 1.0)
    in_01 = np.clip(in_raw, 0.0, 1.0)

    rmin, rmax, cmin, cmax = get_roi_bbox(gt_01)

    if InferenceConfig.use_roi:
        eval_gt = gt_raw[rmin:rmax, cmin:cmax]
        eval_in = in_raw[rmin:rmax, cmin:cmax]
        eval_out = out_raw[rmin:rmax, cmin:cmax]
        title_suffix = "ROI Clinical Metrics"

        bias_in, mape_in, cr_in = calculate_clinical_metrics(eval_gt, eval_in)
        bias_out, mape_out, cr_out = calculate_clinical_metrics(eval_gt, eval_out)

        str_in = f"Noisy Input ({title_suffix})\nBias_max: {bias_in:.1f}% | MAPE_mean: {mape_in:.1f}% | CR: {cr_in:.1f}%"
        str_out = f"JiT-Heun 50 steps ({title_suffix})\nBias_max: {bias_out:.1f}% | MAPE_mean: {mape_out:.1f}% | CR: {cr_out:.1f}%"
    else:
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