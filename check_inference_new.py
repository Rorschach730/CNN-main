import torch
import array_api_compat.torch as torch_api
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import parallelproj
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


def build_2d_pet_projector(device, img_size):
    """
    根据 prepare_data_3d.py 提取并降维的 2D 物理前向投影算子 (单环)
    """
    num_rings = 1
    scanner = parallelproj.RegularPolygonPETScannerGeometry(
        xp=torch_api,
        dev=device,
        radius=350.0,
        num_sides=288,
        num_lor_endpoints_per_side=1,
        lor_spacing=4.0,
        ring_positions=torch_api.tensor([0.0], dtype=torch_api.float32, device=device),
        symmetry_axis=2
    )

    tof_params = parallelproj.TOFParameters(
        num_tofbins=29,
        tofbin_width=13.0,
        sigma_tof=37.0 / 2.355
    )

    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
        scanner,
        max_ring_difference=0
    )

    proj = parallelproj.RegularPolygonPETProjector(
        lor_descriptor=lor_desc,
        img_shape=(img_size, img_size, num_rings),
        voxel_size=(3.0, 3.0, 3.0)
    )
    proj.tof_parameters = tof_params

    return proj


def generate_sinogram_2d(proj, img_2d_raw, device):
    """
    执行物理前向投影以获取 2D 正弦图
    """
    # 转换为物理引擎要求的格式：增加 Z 轴维度 (H, W, 1)，并裁剪负值防止物理发散
    img_tensor = torch.tensor(img_2d_raw, dtype=torch.float32, device=device).unsqueeze(-1)
    img_tensor = torch.clamp(img_tensor, min=0.0)

    with torch.no_grad():
        sino = proj(img_tensor)

    # Sino 默认输出带有 TOF 和 Rings 维度，通过求和坍缩回经典 2D 正弦图 (Views, Radial)
    if len(sino.shape) == 4:
        sino_2d = sino.sum(dim=-1).sum(dim=-1)
    else:
        sino_2d = sino.squeeze()

    return sino_2d.cpu().numpy()


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

    dataset = PETDenoisingDataset(InferenceConfig.data_folder, img_size=InferenceConfig.img_size)
    if len(dataset) == 0:
        print("[!] 测试集为空，请检查路径。")
        return

    #idx = random.randint(0, len(dataset) - 1)
    idx = 145
    for _ in range(20):
        target, condition = dataset[idx]
        if target.max() > 0.05: break
        idx = random.randint(0, len(dataset) - 1)

    target_t = target.unsqueeze(0).to(device)
    condition_t = condition.unsqueeze(0).to(device)

    print(
        f"[*] Generating Heun 2nd-Order... (Steps: {InferenceConfig.sampling_steps}, CFG: {InferenceConfig.cfg_scale})")
    with torch.no_grad():
        output_tensor = model.generate(condition_t, steps=InferenceConfig.sampling_steps,
                                       cfg_scale=InferenceConfig.cfg_scale)

    # 1. 提取原始数据
    out_raw = output_tensor.squeeze().cpu().numpy()
    gt_raw = target_t.squeeze().cpu().numpy()
    in_raw = condition_t.squeeze().cpu().numpy()

    # 2. 物理前向投影生成 Sinogram (严格使用未经截断的 Raw 数据生成物理逼真投影)
    print("[*] 正在执行系统矩阵前向投影，生成 Sinogram 空间映射...")
    proj_engine = build_2d_pet_projector(device, InferenceConfig.img_size)

    # 增加 .T 进行矩阵转置，使 Views 映射至 Y 轴，Radial Bins 映射至 X 轴
    sino_in = generate_sinogram_2d(proj_engine, in_raw, device).T
    sino_out = generate_sinogram_2d(proj_engine, out_raw, device).T
    sino_gt = generate_sinogram_2d(proj_engine, gt_raw, device).T

    # 3. 截断张量用于基础画图
    out_01 = np.clip(out_raw, 0.0, 1.0)
    gt_01 = np.clip(gt_raw, 0.0, 1.0)
    in_01 = np.clip(in_raw, 0.0, 1.0)

    rmin, rmax, cmin, cmax = get_roi_bbox(gt_01)

    # 4. 指标计算
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

    # 5. 可视化 (2x3 布局)
    vmin, vmax = robust_windowing(gt_01)

    # 动态设定 Sinogram 的窗宽窗位，使用 GT Sinogram 作为基准
    s_vmin, s_vmax = 0.0, np.percentile(sino_gt[sino_gt > 0], 99) if np.any(sino_gt > 0) else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行：图像空间
    axes[0, 0].imshow(in_01, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].add_patch(
        plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
    axes[0, 0].set_title(str_in, fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(out_01, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].add_patch(
        plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
    axes[0, 1].set_title(str_out, fontsize=11)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gt_01, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].add_patch(
        plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
    axes[0, 2].set_title("Ground Truth (Clean)", fontsize=11)
    axes[0, 2].axis('off')

    # 第二行：投影空间 (Sinogram) - 标签已修正为竖向逻辑
    axes[1, 0].imshow(sino_in, cmap='viridis', aspect='auto', vmin=s_vmin, vmax=s_vmax)
    axes[1, 0].set_title("Noisy Sinogram", fontsize=11)
    axes[1, 0].set_xlabel("Radial Bins")
    axes[1, 0].set_ylabel("Views")

    axes[1, 1].imshow(sino_out, cmap='viridis', aspect='auto', vmin=s_vmin, vmax=s_vmax)
    axes[1, 1].set_title("Denoised Sinogram (JiT)", fontsize=11)
    axes[1, 1].set_xlabel("Radial Bins")
    axes[1, 1].set_ylabel("Views")  # 保持一致的 Y 轴标签

    axes[1, 2].imshow(sino_gt, cmap='viridis', aspect='auto', vmin=s_vmin, vmax=s_vmax)
    axes[1, 2].set_title("Ground Truth Sinogram", fontsize=11)
    axes[1, 2].set_xlabel("Radial Bins")
    axes[1, 2].set_ylabel("Views")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()