import os
import pydicom
import numpy as np
import random
from tqdm import tqdm
from skimage.transform import radon, iradon, resize
from concurrent.futures import ProcessPoolExecutor


# ==========================================
#              物理仿真参数配置
# ==========================================
class SimConfig:
    # 1. 模拟设置
    DRF = 20.0

    # 虚拟全剂量光子数基准 (Virtual High-Dose Counts)
    # 经定量视觉标定，5e6 为保留泊松特性与解剖结构的甜点区
    TOTAL_COUNTS_HIGH = 5e6

    # 2. 图像设置
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


def physics_based_noise(image, drf=20.0):
    """
    [核心算法] 标准 PET 低剂量物理仿真流程
    Image (Clean) -> Radon -> Sinogram -> Scaling -> Poisson -> FBP(Hann) -> Image (Noisy)
    """
    image = np.clip(image, 0, None)
    if image.max() == 0: return image

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram_ideal = radon(image, theta=theta, circle=False)

    current_sum = sinogram_ideal.sum()
    if current_sum == 0: return image

    scale_factor = SimConfig.TOTAL_COUNTS_HIGH / current_sum
    sinogram_counts_high = sinogram_ideal * scale_factor

    sinogram_counts_low = sinogram_counts_high / drf
    sinogram_noisy_counts = np.random.poisson(sinogram_counts_low).astype(np.float32)

    # 使用 hann 滤波器压制 FBP 极限低计数下的放射状伪影
    reconstruction = iradon(sinogram_noisy_counts, theta=theta, circle=False, filter_name='hann')
    reconstruction = reconstruction / scale_factor * drf

    return reconstruction.astype(np.float32)


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
            vol_noisy[i] = physics_based_noise(vol_clean[i], drf=SimConfig.DRF)
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
    output_root = '../processed_data_sinogram'

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

    print(f"开始生成物理仿真数据 (DRF={SimConfig.DRF}, Counts={SimConfig.TOTAL_COUNTS_HIGH})...")
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