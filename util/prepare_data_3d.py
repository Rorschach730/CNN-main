import os
import pydicom
import math
import torch
import array_api_compat.torch as torch_api
import numpy as np
import random
import time
from tqdm import tqdm
import parallelproj
from skimage.transform import resize
from skimage.morphology import closing, disk, remove_small_objects


class SimConfig3D:
    # 物理拓扑
    IMG_SIZE = 128
    DEPTH_PATCH = 20  # [极限退守] 强行压制在 16GB 显存内，切断 Windows PCIe 共享内存陷阱
    STRIDE = 10  # [相干性对齐] 滑动步幅，维持 50% 物理重叠率
    PAD_SIZE = 5  # [防畸变截断] 首尾切除区大小，保留中心 10 层绝对安全区

    # 剂量与迭代
    DRF_RANGE = [2.0, 10.0]  # [物理修正] 收缩极端散粒噪声域
    TOTAL_COUNTS_HIGH = 1e7  # 3D 体素块的全局计数基准
    OSEM_ITERATIONS = 50  # 纯正泊松 MLEM 迭代收敛深度


def get_metadata(dcm_dir):
    dcm_files = [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
    if not dcm_files: return None
    try:
        ds = pydicom.dcmread(os.path.join(dcm_dir, dcm_files[0]))
        dose = 0.0
        if 'RadiopharmaceuticalInformationSequence' in ds:
            radio_info = ds.RadiopharmaceuticalInformationSequence[0]
            if 'RadionuclideTotalDose' in radio_info:
                dose = float(radio_info.RadionuclideTotalDose) / 1e6
        return {
            "pid": str(ds.PatientID) if 'PatientID' in ds else "Unknown",
            "sex": str(ds.PatientSex) if 'PatientSex' in ds else "U",
            "dose": f"{dose:.1f}MBq",
            "part": str(ds.BodyPartExamined) if 'BodyPartExamined' in ds else "Body",
            "src_dir": dcm_dir
        }
    except:
        return None


def is_valid_volume(dcm_dir):
    dcm_files = [os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
    if len(dcm_files) < 10: return False
    return True


def build_clinical_3d_pet_engine(device, num_rings):
    scanner = parallelproj.RegularPolygonPETScannerGeometry(
        xp=torch_api,
        dev=device,
        radius=350.0,
        num_sides=288,
        num_lor_endpoints_per_side=1,
        lor_spacing=4.0,
        ring_positions=torch_api.linspace(- (num_rings * 2.0), (num_rings * 2.0), num_rings, dtype=torch_api.float32,
                                          device=device),
        symmetry_axis=2
    )

    tof_params = parallelproj.TOFParameters(
        num_tofbins=29,
        tofbin_width=13.0,
        sigma_tof=37.0 / 2.355
    )

    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
        scanner,
        max_ring_difference=scanner.num_rings - 1
    )

    proj = parallelproj.RegularPolygonPETProjector(
        lor_descriptor=lor_desc,
        img_shape=(SimConfig3D.IMG_SIZE, SimConfig3D.IMG_SIZE, num_rings),
        voxel_size=(3.0, 3.0, 3.0)
    )
    proj.tof_parameters = tof_params

    return proj


def generate_synthetic_mu_map_3d(volume_clean):
    mu_map = np.zeros_like(volume_clean, dtype=np.float32)
    max_val = volume_clean.max()
    if max_val <= 1e-4: return mu_map

    for i in range(volume_clean.shape[2]):
        slice_2d = volume_clean[:, :, i]
        body_mask = slice_2d > (max_val * 0.02)
        body_mask = closing(body_mask, disk(5))
        body_mask = remove_small_objects(body_mask, max_size=199)
        if not np.any(body_mask): continue

        low_uptake_mask = (slice_2d < max_val * 0.05) & body_mask
        lung_mask = remove_small_objects(low_uptake_mask, max_size=149)

        mu_slice = np.zeros_like(slice_2d)
        mu_slice[body_mask] = 0.096
        mu_slice[lung_mask] = 0.026
        mu_map[:, :, i] = mu_slice

    return mu_map


def simulate_and_reconstruct_3d_chunk(proj, volume_gt, device, drf):
    img_tensor = torch.clamp(torch.tensor(volume_gt, dtype=torch.float32, device=device), min=0.0)
    mu_map = generate_synthetic_mu_map_3d(volume_gt)
    mu_tensor = torch.tensor(mu_map, dtype=torch.float32, device=device)

    sino_mu = proj(mu_tensor)
    atten_factor = torch.exp(-sino_mu)
    atten_factor = torch.clamp(atten_factor, 1e-4, 1.0)

    sino_clean = proj(img_tensor)
    current_sum = sino_clean.sum()
    if current_sum < 1e-6: return volume_gt

    scale_factor = SimConfig3D.TOTAL_COUNTS_HIGH / current_sum
    sino_ideal = (sino_clean * scale_factor * atten_factor) / drf

    sino_ideal = torch.clamp(sino_ideal, min=0.0)
    sino_noisy = torch.poisson(sino_ideal)

    recon_img = torch.ones_like(img_tensor)
    sens_img = proj.adjoint(atten_factor)

    for _ in range(SimConfig3D.OSEM_ITERATIONS):
        expected = atten_factor * proj(recon_img)
        ratio = sino_noisy / (expected + 1e-6)
        backproj = proj.adjoint(atten_factor * ratio)
        update_factor = backproj / (sens_img + 1e-6)
        recon_img = recon_img * update_factor

    recon_img = (recon_img * drf) / scale_factor
    return torch.clamp(recon_img, 0.0, 1.0).cpu().numpy()


def process_patient_3d(patient_info, proj_engine, device):
    src_dir = patient_info['src_dir']
    output_path = patient_info['output_path']
    pid = patient_info['pid']

    dcm_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.dcm')]
    slices = []
    for f in dcm_files:
        try:
            slices.append(pydicom.dcmread(f))
        except:
            pass
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    try:
        volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    except:
        return

    d, h, w = volume.shape
    vol_clean = np.zeros((d, SimConfig3D.IMG_SIZE, SimConfig3D.IMG_SIZE), dtype=np.float32)
    for i in range(d):
        vol_clean[i] = resize(volume[i], (SimConfig3D.IMG_SIZE, SimConfig3D.IMG_SIZE), anti_aliasing=True)

    # ---------------------------------------------------------
    # [核心物理修复]: 废弃绝对最大值归一化，采用 99.5% 鲁棒截断
    # ---------------------------------------------------------
    vol_clean = np.clip(vol_clean, 0.0, None)
    valid_pixels = vol_clean[vol_clean > 1e-4]

    if len(valid_pixels) > 0:
        vmax = np.percentile(valid_pixels, 99.5)
        # 将超过 vmax 的部分截断，并安全归一化到 [0, 1]
        vol_clean = np.clip(vol_clean, 0.0, vmax) / vmax
    else:
        # 极端空张量保护
        vol_clean = np.zeros_like(vol_clean)
    # ---------------------------------------------------------

    target_d = math.ceil(d / SimConfig3D.STRIDE) * SimConfig3D.STRIDE
    pad_top = SimConfig3D.PAD_SIZE
    pad_bottom = target_d - d + SimConfig3D.PAD_SIZE

    vol_clean_padded = np.pad(vol_clean, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant')
    vol_noisy_padded = np.zeros_like(vol_clean_padded)
    d_padded = vol_clean_padded.shape[0]

    # 生成物理降低因子
    current_drf = np.random.uniform(SimConfig3D.DRF_RANGE[0], SimConfig3D.DRF_RANGE[1])
    # [修正注入]: 将 DRF 刻印进元数据字典，消灭可视化时的 Unknown
    patient_info['drf'] = current_drf

    z_starts = list(range(0, d_padded - SimConfig3D.DEPTH_PATCH + 1, SimConfig3D.STRIDE))
    total_chunks = len(z_starts)

    for chunk_idx, start_z in enumerate(z_starts, 1):
        end_z = start_z + SimConfig3D.DEPTH_PATCH
        chunk_clean = vol_clean_padded[start_z:end_z].transpose(1, 2, 0)

        if np.max(chunk_clean) > 0.001:
            chunk_noisy = simulate_and_reconstruct_3d_chunk(proj_engine, chunk_clean, device, current_drf)
            chunk_noisy = chunk_noisy.transpose(2, 0, 1)
            vol_noisy_padded[start_z + pad_top: start_z + pad_top + SimConfig3D.STRIDE] = \
                chunk_noisy[pad_top: pad_top + SimConfig3D.STRIDE]
        else:
            vol_noisy_padded[start_z + pad_top: start_z + pad_top + SimConfig3D.STRIDE] = \
                vol_clean_padded[start_z + pad_top: start_z + pad_top + SimConfig3D.STRIDE]

        tqdm.write(f"        [-] 病例 {pid} | Chunk {chunk_idx}/{total_chunks} 物理仿真完成")

    vol_noisy = vol_noisy_padded[pad_top: pad_top + d]

    data_dict = {
        "input": vol_noisy,
        "target": vol_clean,
        "metadata": patient_info
    }
    np.save(output_path, data_dict)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_data_dir = '../lung_pet_data'
    output_root = '../processed_data_3d_osem'

    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)

    all_patients = []
    patient_dirs = [os.path.join(raw_data_dir, d) for d in os.listdir(raw_data_dir)
                    if os.path.isdir(os.path.join(raw_data_dir, d))]

    print("[*] 扫描病例元数据并同步验证...")
    for p_dir in tqdm(patient_dirs):
        if is_valid_volume(p_dir):
            meta = get_metadata(p_dir)
            if meta: all_patients.append(meta)

    random.seed(42)
    random.shuffle(all_patients)
    num_p = len(all_patients)
    train_end = int(num_p * 0.7)
    val_end = train_end + int(num_p * 0.15)

    splits = {
        'train': all_patients[:train_end],
        'val': all_patients[train_end:val_end],
        'test': all_patients[val_end:]
    }

    print(f"[*] 初始化 3D 物理引擎 (Patch Depth: {SimConfig3D.DEPTH_PATCH}, Stride: {SimConfig3D.STRIDE})...")
    proj_engine = build_clinical_3d_pet_engine(device, SimConfig3D.DEPTH_PATCH)

    print(f"[*] 启动全量 3D 临床物理仿真 (无缝拼接版，DRF: {SimConfig3D.DRF_RANGE})...")

    for split_name, patients in splits.items():
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n--- 正在处理 {split_name} 集合 ({len(patients)} 例) ---")

        for p in tqdm(patients):
            start_time = time.time()
            fname = f"{p['pid']}_{p['sex']}_{p['dose']}_{p['part']}.npy"
            p['output_path'] = os.path.join(split_dir, fname)

            process_patient_3d(p, proj_engine, device)

            cost_time = time.time() - start_time
            tqdm.write(f"    [+] 病例 {p['pid']} 重建落盘完毕 | 耗时: {cost_time:.1f} 秒")

    print(f"\n[完成] 所有 3D 临床物理数据已严格按照 {output_root} 中的 train/val/test 切分并保存完毕。")


if __name__ == "__main__":
    main()