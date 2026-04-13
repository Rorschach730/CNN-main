import os
import shutil
import pydicom
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm


class UDPETCleanerConfig:
    ROOT_DIRS = [
        "H:/Bern-Inselspital-2022",
        "H:/Shanghai-Ruijin-Hospital-2022",
        "H:/Shanghai-Ruijin-Hospital-2023"
    ]
    OUTPUT_DIR = "../processed_data_udpet"
    DOSE_TARGET = "1/10"
    TARGET_SIZE = 256
    TARGET_FOV_MM = 500.0
    TRAIN_SIZE = 342
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_suv_factor(dcm_hdr):
    try:
        weight_kg = float(getattr(dcm_hdr, 'PatientWeight', 0))
        if weight_kg <= 0:
            return None

        seq = dcm_hdr.RadiopharmaceuticalInformationSequence[0]
        total_dose = float(getattr(seq, 'RadionuclideTotalDose', 0))
        half_life = float(getattr(seq, 'RadionuclideHalfLife', 0))
        if total_dose <= 0 or half_life <= 0:
            return None

        inj_time_str = getattr(seq, 'RadiopharmaceuticalStartTime', None)
        acq_time_str = getattr(dcm_hdr, 'AcquisitionTime', None)
        if not inj_time_str or not acq_time_str:
            return None

        inj_time = datetime.strptime(inj_time_str[:6], "%H%M%S")
        acq_time = datetime.strptime(acq_time_str[:6], "%H%M%S")
        delta_seconds = (acq_time - inj_time).total_seconds()

        if delta_seconds < 0:
            delta_seconds += 86400

        decay_factor = 2.0 ** (-delta_seconds / half_life)
        return (weight_kg * 1000.0) / (total_dose * decay_factor)
    except:
        return None


def physical_resample_and_crop_gpu(slice_2d, original_spacing_xy):
    target_spacing = UDPETCleanerConfig.TARGET_FOV_MM / UDPETCleanerConfig.TARGET_SIZE
    zoom_factor = original_spacing_xy[0] / target_spacing
    h, w = slice_2d.shape
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

    t = torch.tensor(slice_2d, dtype=torch.float32, device=UDPETCleanerConfig.DEVICE).unsqueeze(0).unsqueeze(0)
    t_res = F.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
    t_res = t_res.squeeze(0).squeeze(0)

    pad_h, pad_w = max(0, 256 - new_h), max(0, 256 - new_w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        t_res = F.pad(t_res, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    curr_h, curr_w = t_res.shape
    if curr_h > 256 or curr_w > 256:
        sh, sw = (curr_h - 256) // 2, (curr_w - 256) // 2
        t_res = t_res[sh:sh + 256, sw:sw + 256]

    return t_res


def truncate_bladder_and_legs(volume_3d):
    search_start = int(volume_3d.shape[0] * 0.5)
    if search_start >= volume_3d.shape[0]:
        return volume_3d

    lower_half = volume_3d[search_start:]
    bladder_idx = np.argmax(np.max(lower_half, axis=(1, 2)))
    return volume_3d[:max(1, search_start + bladder_idx - 5)]


def build_dataset_inventory():
    print("=" * 70)
    print(f"[*] 阶段一：全量物理映射字典构建与过滤 (运算后端: {UDPETCleanerConfig.DEVICE.type.upper()})")
    print("=" * 70)
    inventory = []

    for root_dir in UDPETCleanerConfig.ROOT_DIRS:
        if not os.path.exists(root_dir):
            continue

        for root, dirs, _ in os.walk(root_dir):
            root_lower = root.lower()
            if not ('full' in root_lower or 'normal' in root_lower):
                continue

            patient_dir = os.path.dirname(root)
            low_dose_dir = None
            available_folders = os.listdir(patient_dir)

            for d in available_folders:
                full_d_path = os.path.join(patient_dir, d)
                if not os.path.isdir(full_d_path):
                    continue

                d_lower = d.lower()
                # 数据源路由级匹配隔离：修复瑞金医院多命名规范并存问题
                if 'shanghai-ruijin' in root_lower:
                    if ('d10' in d_lower or '1-10' in d_lower or '1_10' in d_lower) and '100' not in d_lower:
                        low_dose_dir = full_d_path
                        break
                elif 'bern-inselspital' in root_lower:
                    if ('1-10' in d_lower or '1_10' in d_lower) and '100' not in d_lower:
                        low_dose_dir = full_d_path
                        break

            if not low_dose_dir:
                continue

            try:
                dicom_files = [f for f in os.listdir(root) if f.lower().endswith(('.dcm', '.ima'))]
                if not dicom_files:
                    continue

                sample_file = os.path.join(root, dicom_files[0])
                dcm = pydicom.dcmread(sample_file, stop_before_pixels=True)

                tracer = getattr(dcm.RadiopharmaceuticalInformationSequence[0], 'Radiopharmaceutical',
                                 'Unknown') if 'RadiopharmaceuticalInformationSequence' in dcm else 'Unknown'
                if 'FDG' not in tracer.upper() and 'FLUORODEOXYGLUCOSE' not in tracer.upper():
                    continue

                age_raw = getattr(dcm, 'PatientAge', '000Y')

                inventory.append({
                    'uid': getattr(dcm, 'StudyInstanceUID', patient_dir),
                    'full_path': root,
                    'low_path': low_dose_dir,
                    'gender': getattr(dcm, 'PatientSex', 'Unknown'),
                    'age': f"{age_raw[:2]}0s" if len(age_raw) >= 2 else 'Unknown',
                    'manufacturer': getattr(dcm, 'Manufacturer', 'Unknown')
                })
            except:
                continue

    if len(inventory) == 0:
        exit(1)

    df = pd.DataFrame(inventory).sort_values('uid').reset_index(drop=True)
    df['patient_id'] = [f"P{i:04d}" for i in range(len(df))]
    print(f"[*] 映射完毕。共锁定 {len(df)} 例双端配对纯正 FDG 数据。")
    return df


def stratified_split(df):
    print("\n" + "=" * 70 + "\n[*] 阶段二：联合分层抽样 (底层安全版)\n" + "=" * 70)
    sampled_dfs = []

    for _, group in df.groupby(['manufacturer', 'gender', 'age']):
        n_sample = int(np.round(len(group) / len(df) * UDPETCleanerConfig.TRAIN_SIZE))
        if n_sample > 0:
            sampled_dfs.append(group.sample(n_sample, random_state=UDPETCleanerConfig.SEED))

    if sampled_dfs:
        train_df = pd.concat(sampled_dfs, ignore_index=True)
    else:
        train_df = pd.DataFrame(columns=df.columns)

    if len(train_df) < UDPETCleanerConfig.TRAIN_SIZE:
        rem_df = df[~df['uid'].isin(train_df['uid'])]
        train_df = pd.concat([
            train_df,
            rem_df.sample(UDPETCleanerConfig.TRAIN_SIZE - len(train_df), random_state=UDPETCleanerConfig.SEED)
        ], ignore_index=True)
    elif len(train_df) > UDPETCleanerConfig.TRAIN_SIZE:
        train_df = train_df.sample(UDPETCleanerConfig.TRAIN_SIZE, random_state=UDPETCleanerConfig.SEED).reset_index(
            drop=True)

    test_df = df[~df['uid'].isin(train_df['uid'])].reset_index(drop=True)
    print(f"    [数据集切分完成] Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df


def process_and_save(df, split_name):
    print("\n" + "=" * 70 + f"\n[*] 阶段三：{split_name.upper()} 病理级分装落盘 (支持断点续传)\n" + "=" * 70)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}", unit="pat"):
        patient_save_dir = os.path.join(UDPETCleanerConfig.OUTPUT_DIR, split_name, row['patient_id'])

        # 前置脏数据销毁机制
        if os.path.exists(patient_save_dir):
            if len([f for f in os.listdir(patient_save_dir) if f.endswith('.pt')]) > 10:
                continue
            else:
                shutil.rmtree(patient_save_dir)

        full_files = [os.path.join(row['full_path'], f) for f in os.listdir(row['full_path']) if
                      f.lower().endswith(('.dcm', '.ima'))]
        low_files = [os.path.join(row['low_path'], f) for f in os.listdir(row['low_path']) if
                     f.lower().endswith(('.dcm', '.ima'))]

        # 配对不齐的数据直接斩断
        if len(full_files) != len(low_files) or len(full_files) == 0:
            continue

        full_slices, low_slices = [], []
        suv_factor, spacing_xy = None, None

        try:
            for ff, lf in zip(sorted(full_files), sorted(low_files)):
                fdcm, ldcm = pydicom.dcmread(ff), pydicom.dcmread(lf)

                if not suv_factor:
                    suv_factor = calculate_suv_factor(fdcm)
                    spacing_xy = [float(x) for x in fdcm.PixelSpacing]
                if not suv_factor:
                    break

                f_px = fdcm.pixel_array * float(getattr(fdcm, 'RescaleSlope', 1)) + float(
                    getattr(fdcm, 'RescaleIntercept', 0))
                l_px = ldcm.pixel_array * float(getattr(ldcm, 'RescaleSlope', 1)) + float(
                    getattr(ldcm, 'RescaleIntercept', 0))

                full_slices.append((float(fdcm.ImagePositionPatient[2]), f_px * suv_factor))
                low_slices.append((float(ldcm.ImagePositionPatient[2]), l_px * suv_factor))
        except:
            continue

        if not suv_factor or len(full_slices) == 0:
            continue

        full_slices.sort(key=lambda x: x[0], reverse=True)
        low_slices.sort(key=lambda x: x[0], reverse=True)

        vol_full = np.stack([s[1] for s in full_slices])
        vol_low = np.stack([s[1] for s in low_slices])

        vol_full = truncate_bladder_and_legs(vol_full)
        vol_low = vol_low[:vol_full.shape[0]]

        # 仅当所有数据在内存中合法存在后，才创建文件夹
        os.makedirs(patient_save_dir, exist_ok=True)

        for z_idx in range(vol_full.shape[0]):
            f_res = physical_resample_and_crop_gpu(vol_full[z_idx], spacing_xy)
            l_res = physical_resample_and_crop_gpu(vol_low[z_idx], spacing_xy)
            tensor_pair = torch.stack([l_res, f_res], dim=0).cpu()
            torch.save(tensor_pair, os.path.join(patient_save_dir, f"{row['patient_id']}_Z{z_idx:04d}.pt"))


if __name__ == "__main__":
    df_inventory = build_dataset_inventory()
    df_train, df_test = stratified_split(df_inventory)
    process_and_save(df_train, 'train')
    process_and_save(df_test, 'test')