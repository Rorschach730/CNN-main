import os
import pydicom
import numpy as np
import random
from tqdm import tqdm
from skimage.transform import radon, iradon, resize
from skimage.morphology import closing, disk, remove_small_objects
from concurrent.futures import ProcessPoolExecutor


# ==========================================
#              物理仿真参数配置
# ==========================================
class SimConfig:
    # [战术重构] 动态混合剂量课程式学习区间 (对应 25% 到 5% 的临床剂量)
    DRF_RANGE = [4.0, 20.0]

    # 虚拟全剂量光子数基准
    TOTAL_COUNTS_HIGH = 5e6

    IMG_SIZE = 128
    NUM_ANGLES = 180


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


def generate_synthetic_mu_map(image_clean):
    """
    [物理升维] 基于纯净 PET 图像生成 3-Class (空气、肺、软组织) 合成衰减图
    修复 skimage >= 0.26.0 的所有 API 弃用警告
    """
    mu_map = np.zeros_like(image_clean, dtype=np.float32)
    max_val = image_clean.max()

    if max_val <= 1e-4:
        return mu_map

    body_mask = image_clean > (max_val * 0.02)
    body_mask = closing(body_mask, disk(5))
    body_mask = remove_small_objects(body_mask, max_size=199)

    if not np.any(body_mask):
        return mu_map

    low_uptake_mask = (image_clean < max_val * 0.05) & body_mask
    lung_mask = remove_small_objects(low_uptake_mask, max_size=149)

    mu_map[body_mask] = 0.096
    mu_map[lung_mask] = 0.026

    return mu_map


def physics_based_noise(image, drf, pixel_spacing_cm):
    """
    [核心算法] 搭载量纲修复与动态剂量的 PET 物理仿真引擎
    """
    image = np.clip(image, 0, None)
    if image.max() <= 1e-4: return image

    img_size = max(image.shape)
    theta = np.linspace(0., 180., img_size, endpoint=False)

    sino_clean = radon(image, theta=theta, circle=False)
    current_sum = sino_clean.sum()
    if current_sum == 0: return image

    scale_factor = SimConfig.TOTAL_COUNTS_HIGH / current_sum

    mu_map = generate_synthetic_mu_map(image)

    # ==========================================
    # [量纲修复核心] 将纯数学的像素累加，转化为具有绝对物理标度的积分(cm)
    # ==========================================
    sino_mu = radon(mu_map, theta=theta, circle=False) * pixel_spacing_cm
    atten_factor = np.exp(-sino_mu)
    atten_factor = np.clip(atten_factor, 1e-4, 1.0)

    sino_ideal_counts = (sino_clean * scale_factor * atten_factor) / drf
    sino_noisy_counts = np.random.poisson(sino_ideal_counts).astype(np.float32)
    sino_ac_counts = sino_noisy_counts / atten_factor

    reconstruction = iradon(sino_ac_counts, theta=theta, circle=False, filter_name='hann')
    reconstruction = (reconstruction * drf) / scale_factor

    return np.clip(reconstruction.astype(np.float32), 0.0, 1.0)


def process_patient(patient_info):
    src_dir = patient_info['src_dir']
    output_path = patient_info['output_path']

    dcm_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.dcm')]
    slices = []
    for f in dcm_files:
        try:
            d = pydicom.dcmread(f)
            slices.append(d)
        except:
            pass

    if not slices: return
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # [DICOM 物理量纲提取]
    try:
        # 提取 PixelSpacing (格式: [相邻行间距mm, 相邻列间距mm])
        orig_spacing_mm = float(slices[0].PixelSpacing[0])
        orig_spacing_cm = orig_spacing_mm / 10.0
        orig_size = float(slices[0].Rows)

        # 计算 Resize 到 128x128 后的新物理像素尺寸
        new_spacing_cm = orig_spacing_cm * (orig_size / SimConfig.IMG_SIZE)
    except:
        # 极少数缺乏物理标定的冗余数据，采用常规躯干 60cm FOV 作为托底
        new_spacing_cm = 60.0 / SimConfig.IMG_SIZE

    # [混合剂量采样] 为当前患者随机抽取一个 DRF 难度
    current_drf = random.uniform(SimConfig.DRF_RANGE[0], SimConfig.DRF_RANGE[1])
    patient_info['drf_applied'] = current_drf

    try:
        volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    except:
        return

    d, h, w = volume.shape
    vol_clean = np.zeros((d, SimConfig.IMG_SIZE, SimConfig.IMG_SIZE), dtype=np.float32)

    for i in range(d):
        vol_clean[i] = resize(volume[i], (SimConfig.IMG_SIZE, SimConfig.IMG_SIZE), anti_aliasing=True)

    max_val = np.max(vol_clean)
    if max_val > 0:
        vol_clean = vol_clean / max_val

    vol_noisy = np.zeros_like(vol_clean)
    for i in range(d):
        if np.max(vol_clean[i]) > 0.001:
            vol_noisy[i] = physics_based_noise(vol_clean[i], drf=current_drf, pixel_spacing_cm=new_spacing_cm)
        else:
            vol_noisy[i] = vol_clean[i]

    data_dict = {
        "input": vol_noisy,
        "target": vol_clean,
        "metadata": patient_info
    }
    np.save(output_path, data_dict)


def main():
    raw_data_dir = '../lung_pet_data'
    output_root = '../processed_data_sinogram_new'

    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)

    all_patients = []
    patient_dirs = [os.path.join(raw_data_dir, d) for d in os.listdir(raw_data_dir)
                    if os.path.isdir(os.path.join(raw_data_dir, d))]

    print("扫描病例元数据...")
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

    print(f"开始生成量纲修复版混合物理数据 (DRF 范围={SimConfig.DRF_RANGE})...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for split_name, patients in splits.items():
            split_dir = os.path.join(output_root, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for p in patients:
                fname = f"{p['pid']}_{p['sex']}_{p['dose']}_{p['part']}.npy"
                p['output_path'] = os.path.join(split_dir, fname)
                futures.append(executor.submit(process_patient, p))

        for _ in tqdm(futures, total=len(futures)):
            pass

    print(f"\n[完成] 所有数据已保存至: {output_root}")


if __name__ == "__main__":
    main()