import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from denoiser_ud import Denoiser
from util.pet_dataset_ud import PETDenoisingDataset


class InferenceConfig:
    device = "cuda"
    checkpoint_path = "./output_dir_ud/checkpoint-199.pth"
    data_folder = "./processed_data_udpet/test"

    img_size = 256
    patch_size = 16
    accum_iter = 11

    attn_dropout = 0.0
    proj_dropout = 0.0
    P_mean = -0.5
    P_std = 1.2
    cond_drop_prob = 0.1
    cfg_scale = 0.6

    sampling_steps = 50

    use_roi = True
    calc_bias = True
    calc_cv = True

    # 提取样本的物理剂量设定 (10代表1/10, 4代表1/4, 2代表1/2)
    target_dose = 10
    target_patient = None
    target_z_slice = None

    # 新增：全量测试标签池 (无论输入什么剂量，都强制走完这三个标签)
    test_dose_labels = [0.1, 0.25, 0.5]


def get_mock_args():
    class MockArgs:
        pass

    args = MockArgs()
    for k, v in InferenceConfig.__dict__.items():
        if not k.startswith('__'): setattr(args, k, v)
    return args


def calculate_snr(clean, test):
    var_clean = np.var(clean)
    var_noise = np.var(clean - test)
    if var_noise == 0: return float('inf')
    return 10 * np.log10(var_clean / var_noise)


def calculate_suv_bias(gt_roi, pred_roi):
    eps = 1e-8
    max_gt = np.percentile(gt_roi, 98)
    max_pred = np.percentile(pred_roi, 98)
    bias_suvmax = ((max_pred - max_gt) / (max_gt + eps)) * 100.0
    return bias_suvmax


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


def targeted_or_random_probe(data_dir, device, threshold=0.05, target_dose=None, target_patient=None, target_z=None):
    if target_patient is not None and target_z is not None:
        p_dir = os.path.join(data_dir, target_patient)
        z_str = f"_Z{target_z:04d}.pt"
        dose_str = f"_D{target_dose}_" if target_dose is not None else ""
        target_file = next((f for f in os.listdir(p_dir) if z_str in f and dose_str in f), None)
        if target_file is None: raise ValueError(f"[!] 找不到文件")
        file_path = os.path.join(p_dir, target_file)
        tensor_pair = torch.load(file_path, map_location='cpu', weights_only=True)
        match = re.search(r'_D(\d+)_', target_file)
        true_dose = 1.0 / float(match.group(1)) if match else 0.1
        return tensor_pair[1:2, :, :], tensor_pair[0:1, :, :], torch.tensor([true_dose],
                                                                            dtype=torch.float32), target_file, true_dose

    patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    attempts = 0
    while True:
        attempts += 1
        p_dir = random.choice(patient_dirs)
        full_p_dir = os.path.join(data_dir, p_dir)
        pt_files = [f for f in os.listdir(full_p_dir) if f.endswith('.pt')]
        if target_dose is not None: pt_files = [f for f in pt_files if f"_D{target_dose}_" in f]
        if not pt_files: continue
        f_name = random.choice(pt_files)
        file_path = os.path.join(full_p_dir, f_name)
        tensor_pair = torch.load(file_path, map_location='cpu', weights_only=True)
        if tensor_pair[1].max() > threshold:
            print(f"[*] 盲抽命中 (次数: {attempts}): {p_dir}/{f_name}")
            match = re.search(r'_D(\d+)_', f_name)
            true_dose = 1.0 / float(match.group(1)) if match else 0.1
            return tensor_pair[1:2, :, :], tensor_pair[0:1, :, :], torch.tensor([true_dose],
                                                                                dtype=torch.float32), f_name, true_dose


def main():
    device = torch.device(InferenceConfig.device)
    if device.type == 'cuda': torch.set_float32_matmul_precision('high')

    args = get_mock_args()
    model = Denoiser(args)
    model.to(device)

    ckpt = torch.load(InferenceConfig.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_ema', ckpt['model'])
    model.net.load_state_dict({k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()},
                              strict=False)
    model.eval()

    # 根据 target_dose 抽取对应物理剂量的样本
    target_t, condition_t, dose_t, f_name, true_dose = targeted_or_random_probe(
        InferenceConfig.data_folder, device, target_dose=InferenceConfig.target_dose,
        target_patient=InferenceConfig.target_patient, target_z=InferenceConfig.target_z_slice
    )

    out_raw_dict = {}

    # 遍历所有标签进行推理 (包含匹配与错配)
    for test_label in InferenceConfig.test_dose_labels:
        force_dose_t = torch.tensor([test_label], dtype=torch.float32)
        with torch.no_grad():
            out_tensor = model.generate(condition_t.unsqueeze(0).to(device), force_dose_t.unsqueeze(0).to(device),
                                        steps=InferenceConfig.sampling_steps, cfg_scale=InferenceConfig.cfg_scale)
        out_raw_dict[test_label] = out_tensor.squeeze().cpu().numpy()

    gt_raw = target_t.squeeze().cpu().numpy()
    in_raw = condition_t.squeeze().cpu().numpy()

    rmin, rmax, cmin, cmax = get_roi_bbox(gt_raw)

    if InferenceConfig.use_roi:
        eval_gt_raw = gt_raw[rmin:rmax, cmin:cmax]
        eval_in_raw = in_raw[rmin:rmax, cmin:cmax]
        eval_out_dict = {k: v[rmin:rmax, cmin:cmax] for k, v in out_raw_dict.items()}
        title_suffix = "(ROI)"
    else:
        eval_gt_raw = gt_raw
        eval_in_raw = in_raw
        eval_out_dict = out_raw_dict
        title_suffix = "(Whole Image)"

    str_in = f"Noisy Input {title_suffix}\n"
    str_out_dict = {}
    for k in InferenceConfig.test_dose_labels:
        # 判定当前标签是否与物理剂量匹配
        status = "Matched" if abs(k - true_dose) < 1e-4 else "Mismatched"
        str_out_dict[k] = f"JiT Denoised (Label: {k:.3f} | {status})\n"

    if InferenceConfig.calc_bias:
        bias_in = calculate_suv_bias(eval_gt_raw, eval_in_raw)
        str_in += f"SUVmax Bias(98%): {bias_in:.2f}%\n"
        for k, eval_out in eval_out_dict.items():
            bias_out = calculate_suv_bias(eval_gt_raw, eval_out)
            str_out_dict[k] += f"SUVmax Bias(98%): {bias_out:.2f}%\n"

    if InferenceConfig.calc_cv:
        dyn_range = eval_gt_raw.max() - eval_gt_raw.min()
        if dyn_range == 0: dyn_range = 1.0

        psnr_in = compare_psnr(eval_gt_raw, eval_in_raw, data_range=dyn_range)
        ssim_in = compare_ssim(eval_gt_raw, eval_in_raw, data_range=dyn_range)
        snr_in = calculate_snr(eval_gt_raw, eval_in_raw)
        mse_in = np.mean((eval_gt_raw - eval_in_raw) ** 2)
        str_in += f"PSNR: {psnr_in:.2f} | SSIM: {ssim_in:.3f} | SNR: {snr_in:.2f} | MSE: {mse_in:.4f}\n"

        for k, eval_out in eval_out_dict.items():
            psnr_out = compare_psnr(eval_gt_raw, eval_out, data_range=dyn_range)
            ssim_out = compare_ssim(eval_gt_raw, eval_out, data_range=dyn_range)
            snr_out = calculate_snr(eval_gt_raw, eval_out)
            mse_out = np.mean((eval_gt_raw - eval_out) ** 2)
            str_out_dict[k] += f"PSNR: {psnr_out:.2f} | SSIM: {ssim_out:.3f} | SNR: {snr_out:.2f} | MSE: {mse_out:.4f}\n"

    vmin, vmax = robust_windowing(gt_raw)

    total_plots = 2 + len(InferenceConfig.test_dose_labels)
    fig, axes = plt.subplots(1, total_plots, figsize=(4.5 * total_plots, 6))
    fig.suptitle(f"[{f_name}]  |  Input Physical Dose: {true_dose:.3f}", fontsize=14, fontweight='bold')

    axes[0].imshow(np.clip(in_raw, 0, vmax), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(str_in.strip(), fontsize=10)
    axes[0].axis('off')

    # 按 0.1, 0.25, 0.5 的顺序输出
    plot_idx = 1
    for k in InferenceConfig.test_dose_labels:
        axes[plot_idx].imshow(np.clip(out_raw_dict[k], 0, vmax), cmap='gray', vmin=vmin, vmax=vmax)
        axes[plot_idx].set_title(str_out_dict[k].strip(), fontsize=10)
        axes[plot_idx].axis('off')
        plot_idx += 1

    axes[-1].imshow(np.clip(gt_raw, 0, vmax), cmap='gray', vmin=vmin, vmax=vmax)
    axes[-1].set_title("Ground Truth", fontsize=10)
    axes[-1].axis('off')

    if InferenceConfig.use_roi:
        for ax in axes:
            ax.add_patch(
                plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))

    save_dir = "./npy_results_probe"
    os.makedirs(save_dir, exist_ok=True)
    base_name = f_name.replace('.pt', '')
    np.save(os.path.join(save_dir, f"{base_name}_roi_gt.npy"), eval_gt_raw)
    np.save(os.path.join(save_dir, f"{base_name}_roi_in.npy"), eval_in_raw)

    for k, eval_out in eval_out_dict.items():
        np.save(os.path.join(save_dir, f"{base_name}_roi_out_D{k:.3f}.npy"), eval_out)

    print(f"[*] 原始 ROI 矩阵及推理矩阵已落盘至: {save_dir}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


if __name__ == "__main__":
    main()