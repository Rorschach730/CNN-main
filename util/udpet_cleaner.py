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
    # 输出到混合剂量专属目录
    OUTPUT_DIR = "../processed_data_udpet"
    TARGET_SIZE = 256

    # 我们基于"病人数量"进行控制，但实际生成的切片会因为多剂量而翻倍
    TRAIN_SIZE = 100
    TEST_SIZE = 30
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 剂量关键字与分母的映射表
    DOSE_MAPPING = {
        '1-2': 2, 'd2': 2, '1_2': 2,
        '1-4': 4, 'd4': 4, '1_4': 4,
        '1-10': 10, 'd10': 10, '1_10': 10,
    }


def calculate_suv_factor(dcm_hdr):
    try:
        weight_kg = float(getattr(dcm_hdr, 'PatientWeight', 0))
        if weight_kg <= 0: return None

        seq = dcm_hdr.RadiopharmaceuticalInformationSequence[0]
        total_dose = float(getattr(seq, 'RadionuclideTotalDose', 0))
        half_life = float(getattr(seq, 'RadionuclideHalfLife', 0))
        if total_dose <= 0 or half_life <= 0: return None

        inj_time_str = getattr(seq, 'RadiopharmaceuticalStartTime', None)
        acq_time_str = getattr(dcm_hdr, 'AcquisitionTime', None)
        if not inj_time_str or not acq_time_str: return None

        inj_time = datetime.strptime(inj_time_str[:6], "%H%M%S")
        acq_time = datetime.strptime(acq_time_str[:6], "%H%M%S")
        delta_seconds = (acq_time - inj_time).total_seconds()
        if delta_seconds < 0: delta_seconds += 86400

        decay_factor = 2.0 ** (-delta_seconds / half_life)
        return (weight_kg * 1000.0) / (total_dose * decay_factor)
    except:
        return None


def center_crop_gpu(slice_2d):
    target = UDPETCleanerConfig.TARGET_SIZE
    t = torch.tensor(slice_2d, dtype=torch.float32, device=UDPETCleanerConfig.DEVICE)
    h, w = t.shape

    pad_h, pad_w = max(0, target - h), max(0, target - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        h, w = t.shape

    if h > target or w > target:
        sh, sw = (h - target) // 2, (w - target) // 2
        t = t[sh:sh + target, sw:sw + target]

    return t


def build_dataset_inventory():
    print("=" * 70)
    print(f"[*] 阶段一：多剂量物理映射字典构建 (运算后端: {UDPETCleanerConfig.DEVICE.type.upper()})")
    print("=" * 70)
    inventory = []

    for root_dir in UDPETCleanerConfig.ROOT_DIRS:
        if not os.path.exists(root_dir): continue

        for root, dirs, _ in os.walk(root_dir):
            root_lower = root.lower()
            if not ('full' in root_lower or 'normal' in root_lower): continue

            patient_dir = os.path.dirname(root)
            available_folders = os.listdir(patient_dir)

            # [核心改造] 搜刮该病人下的所有不同剂量分布
            low_dose_candidates = []
            for d in available_folders:
                full_d_path = os.path.join(patient_dir, d)
                if not os.path.isdir(full_d_path): continue
                d_lower = d.lower()

                # 避开 1000 避免把 1/1000（若有）误认为 1/10
                for keyword, dose_denom in UDPETCleanerConfig.DOSE_MAPPING.items():
                    if keyword in d_lower and '1000' not in d_lower:
                        low_dose_candidates.append((full_d_path, dose_denom))
                        break

            if not low_dose_candidates: continue

            try:
                dicom_files = [f for f in os.listdir(root) if f.lower().endswith(('.dcm', '.ima'))]
                if not dicom_files: continue

                sample_file = os.path.join(root, dicom_files[0])
                dcm = pydicom.dcmread(sample_file, stop_before_pixels=True)

                tracer = getattr(dcm.RadiopharmaceuticalInformationSequence[0], 'Radiopharmaceutical',
                                 'Unknown') if 'RadiopharmaceuticalInformationSequence' in dcm else 'Unknown'
                if 'FDG' not in tracer.upper() and 'FLUORODEOXYGLUCOSE' not in tracer.upper(): continue

                age_raw = getattr(dcm, 'PatientAge', '000Y')
                uid = getattr(dcm, 'StudyInstanceUID', patient_dir)
                gender = getattr(dcm, 'PatientSex', 'Unknown')
                age = f"{age_raw[:2]}0s" if len(age_raw) >= 2 else 'Unknown'
                manufacturer = getattr(dcm, 'Manufacturer', 'Unknown')

                # 一个病人可以产生多个（如 1/4, 1/10）数据对
                for low_path, dose_denom in low_dose_candidates:
                    inventory.append({
                        'uid': uid,
                        'full_path': root,
                        'low_path': low_path,
                        'dose_denom': dose_denom,
                        'gender': gender,
                        'age': age,
                        'manufacturer': manufacturer
                    })
            except:
                continue

    if len(inventory) == 0: exit(1)

    df = pd.DataFrame(inventory).sort_values('uid').reset_index(drop=True)

    # 根据唯一UID分配病人编号，确保同一个病人的所有剂量数据都被分在同一个训练集或测试集
    unique_uids = df['uid'].unique()
    uid_to_pid = {uid: f"P{i:04d}" for i, uid in enumerate(unique_uids)}
    df['patient_id'] = df['uid'].map(uid_to_pid)

    print(f"[*] 映射完毕。共锁定 {len(unique_uids)} 位病人，累计产生 {len(df)} 组不同剂量的配对数据。")
    return df


def exact_stratified_sample_by_uid(df, target_size, seed):
    # 为防止同一个病人的不同剂量被割裂到Train和Test，我们按 UID 抽取病人
    patient_df = df.drop_duplicates('uid')
    sampled_dfs = []

    for _, group in patient_df.groupby(['manufacturer', 'gender', 'age']):
        n_sample = int(np.round(len(group) / len(patient_df) * target_size))
        if n_sample > 0:
            sampled_dfs.append(group.sample(min(n_sample, len(group)), random_state=seed))

    res_pat_df = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame(columns=patient_df.columns)

    if len(res_pat_df) < target_size:
        rem_df = patient_df[~patient_df['uid'].isin(res_pat_df['uid'])]
        needed = target_size - len(res_pat_df)
        res_pat_df = pd.concat([res_pat_df, rem_df.sample(min(needed, len(rem_df)), random_state=seed)],
                               ignore_index=True)
    elif len(res_pat_df) > target_size:
        res_pat_df = res_pat_df.sample(target_size, random_state=seed).reset_index(drop=True)

    # 还原为包含所有剂量的全量数据
    return df[df['uid'].isin(res_pat_df['uid'])].reset_index(drop=True)


def stratified_split(df):
    print("\n" + "=" * 70 + "\n[*] 阶段二：病人级隔离的联合分层抽样\n" + "=" * 70)

    train_df = exact_stratified_sample_by_uid(df, UDPETCleanerConfig.TRAIN_SIZE, UDPETCleanerConfig.SEED)

    remaining_df = df[~df['uid'].isin(train_df['uid'])].reset_index(drop=True)
    test_df = exact_stratified_sample_by_uid(remaining_df,
                                             min(remaining_df['uid'].nunique(), UDPETCleanerConfig.TEST_SIZE),
                                             UDPETCleanerConfig.SEED)

    print(f"    [分配完成] Train: {train_df['uid'].nunique()} 位病人 | Test: {test_df['uid'].nunique()} 位病人")
    return train_df, test_df


def process_and_save(df, split_name):
    print("\n" + "=" * 70 + f"\n[*] 阶段三：{split_name.upper()} 多剂量混合靶向落盘\n" + "=" * 70)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}", unit="pair"):
        patient_save_dir = os.path.join(UDPETCleanerConfig.OUTPUT_DIR, split_name, row['patient_id'])

        # 因为现在一个病人有多组数据，不能随意删除之前建好的文件夹，只判断具体文件即可
        full_files = [os.path.join(row['full_path'], f) for f in os.listdir(row['full_path']) if
                      f.lower().endswith(('.dcm', '.ima'))]
        low_files = [os.path.join(row['low_path'], f) for f in os.listdir(row['low_path']) if
                     f.lower().endswith(('.dcm', '.ima'))]

        if len(full_files) != len(low_files) or len(full_files) == 0: continue

        full_slices, low_slices = [], []
        suv_factor = None

        try:
            for ff, lf in zip(sorted(full_files), sorted(low_files)):
                fdcm, ldcm = pydicom.dcmread(ff), pydicom.dcmread(lf)
                if not suv_factor:
                    suv_factor = calculate_suv_factor(fdcm)
                if not suv_factor: break

                f_px = fdcm.pixel_array * float(getattr(fdcm, 'RescaleSlope', 1)) + float(
                    getattr(fdcm, 'RescaleIntercept', 0))
                l_px = ldcm.pixel_array * float(getattr(ldcm, 'RescaleSlope', 1)) + float(
                    getattr(ldcm, 'RescaleIntercept', 0))

                full_slices.append((float(fdcm.ImagePositionPatient[2]), f_px * suv_factor))
                low_slices.append((float(ldcm.ImagePositionPatient[2]), l_px * suv_factor))
        except:
            continue

        if not suv_factor or len(full_slices) == 0: continue

        full_slices.sort(key=lambda x: x[0], reverse=True)
        low_slices.sort(key=lambda x: x[0], reverse=True)

        top_z = full_slices[0][0]
        filtered_full, filtered_low = [], []

        for (fz, fpx), (lz, lpx) in zip(full_slices, low_slices):
            dist_from_top = abs(top_z - fz)
            if 220.0 <= dist_from_top <= (220.0 + 450.0):
                filtered_full.append((fz, fpx))
                filtered_low.append((lz, lpx))

        if len(filtered_full) == 0: continue

        vol_full = np.stack([s[1] for s in filtered_full])
        vol_low = np.stack([s[1] for s in filtered_low])

        os.makedirs(patient_save_dir, exist_ok=True)
        dose_label = row['dose_denom']

        for z_idx in range(vol_full.shape[0]):
            f_res = center_crop_gpu(vol_full[z_idx])
            l_res = center_crop_gpu(vol_low[z_idx])
            tensor_pair = torch.stack([l_res, f_res], dim=0).cpu()
            # [架构桥接] 动态将 _D{剂量}_ 写入文件名
            save_name = f"{row['patient_id']}_D{dose_label}_Z{z_idx:04d}.pt"
            torch.save(tensor_pair, os.path.join(patient_save_dir, save_name))


if __name__ == "__main__":
    df_inventory = build_dataset_inventory()
    df_train, df_test = stratified_split(df_inventory)
    process_and_save(df_train, 'train')
    process_and_save(df_test, 'test')