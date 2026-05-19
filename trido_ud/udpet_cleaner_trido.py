import os
import pydicom
import numpy as np
import torch
from scipy.signal import find_peaks
import re
from tqdm import tqdm

class UDPETCleanerConfig:
    # 硬编码多数据源路径，自动遍历
    ROOT_DIRS = [
        "H:/Bern-Inselspital-2022",
        "H:/Shanghai-Ruijin-Hospital-2022",
        "H:/Shanghai-Ruijin-Hospital-2023"
    ]
    # 硬编码输出路径
    OUTPUT_DIR = "../processed_data_trido"
    TARGET_SIZE = 256

def calculate_suv_factor(dcm_hdr):
    try:
        weight = float(dcm_hdr.PatientWeight) * 1000.0  
        rad_seq = dcm_hdr.RadiopharmaceuticalInformationSequence[0]
        dose = float(rad_seq.RadionuclideTotalDose)
        slope = getattr(dcm_hdr, 'RescaleSlope', 1.0)
        intercept = getattr(dcm_hdr, 'RescaleIntercept', 0.0)
        factor = (slope * weight) / dose
        return factor, intercept
    except Exception:
        return None, None

def extract_body_parts_indices(full_dose_volumes):
    if len(full_dose_volumes) == 0: return []
    z_suv_profile = np.array([np.max(vol) for vol in full_dose_volumes])
    peaks, _ = find_peaks(z_suv_profile, distance=40, prominence=2.0)
    total_slices = len(z_suv_profile)
    
    part_labels = []
    if len(peaks) < 2:
        idx_brain_end = int(total_slices * 0.15)
        idx_chest_end = int(total_slices * 0.45)
    else:
        idx_brain_end = peaks[0] + 20
        idx_chest_end = peaks[1] + 40
        
    for i in range(total_slices):
        if i <= idx_brain_end: part_labels.append(0)
        elif i <= idx_chest_end: part_labels.append(1)
        else: part_labels.append(2)
    return part_labels

def center_crop_numpy(img_array, target_size=256):
    h, w = img_array.shape
    th, tw = target_size, target_size
    if h < th or w < tw:
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        img_array = np.pad(img_array, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
        h, w = img_array.shape
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img_array[i:i+th, j:j+tw]

def process_patient(patient_dir, output_dir, target_size):
    patient_id = os.path.basename(patient_dir)
    dose_folders = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
    
    full_dose_folder = None
    for folder in dose_folders:
        if 'full dose' in folder.lower() or '100 dose' in folder.lower() or 'drf_100' in folder.lower():
            full_dose_folder = folder
            break
            
    if not full_dose_folder: return
        
    full_slices_data = []
    for f in os.listdir(full_dose_folder):
        if not (f.endswith('.dcm') or f.endswith('.IMA')): continue
        try:
            dcm = pydicom.dcmread(os.path.join(full_dose_folder, f))
            factor, intercept = calculate_suv_factor(dcm)
            if factor is None: continue
            z_pos = float(dcm.ImagePositionPatient[2])
            pixel_array = dcm.pixel_array.astype(np.float32)
            suv_array = pixel_array * factor + intercept
            full_slices_data.append((z_pos, suv_array))
        except Exception:
            continue
            
    if not full_slices_data: return
    full_slices_data.sort(key=lambda x: x[0], reverse=True)
    full_volumes = [s[1] for s in full_slices_data]
    full_z_coords = [s[0] for s in full_slices_data]
    part_labels = extract_body_parts_indices(full_volumes)
    
    for dose_folder in dose_folders:
        if dose_folder == full_dose_folder: continue 
        dose_name = os.path.basename(dose_folder)
        match = re.search(r'(\d+)[_-](\d+)\s*dose', dose_name.lower())
        match_drf = re.search(r'drf_(\d+)', dose_name.lower())
        if match: dose_denom = int(match.group(2))
        elif match_drf: dose_denom = int(match_drf.group(1))
        else: dose_denom = 10
            
        low_slices_dict = {}
        for f in os.listdir(dose_folder):
            if not (f.endswith('.dcm') or f.endswith('.IMA')): continue
            try:
                dcm = pydicom.dcmread(os.path.join(dose_folder, f))
                factor, intercept = calculate_suv_factor(dcm)
                if factor is None: continue
                z_pos = float(dcm.ImagePositionPatient[2])
                suv_array = dcm.pixel_array.astype(np.float32) * factor + intercept
                low_slices_dict[z_pos] = center_crop_numpy(suv_array, target_size)
            except:
                continue
                
        save_patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(save_patient_dir, exist_ok=True)
        
        for i, z in enumerate(full_z_coords):
            matched_z = [lz for lz in low_slices_dict.keys() if abs(lz - z) < 0.5]
            if matched_z:
                target_crop = center_crop_numpy(full_volumes[i], target_size)
                cond_crop = low_slices_dict[matched_z[0]]
                body_part = part_labels[i]
                tensor_pair = torch.stack([torch.from_numpy(target_crop), torch.from_numpy(cond_crop)])
                save_name = f"{patient_id}_Part{body_part}_D{dose_denom}_Z{i:04d}.pt"
                torch.save(tensor_pair, os.path.join(save_patient_dir, save_name))

def main():
    config = UDPETCleanerConfig()
    
    for root_dir in config.ROOT_DIRS:
        if not os.path.exists(root_dir):
            print(f"[跳过] 输入路径不存在: {root_dir}")
            continue
            
        print(f"[*] 开始扫描并处理数据源: {root_dir}")
        # 获取二级目录中的具体病人文件夹
        # 假设结构: H:/Bern-Inselspital-2022/SubFolder/PatientFolder/DoseFolder/DICOMs
        patient_dirs = []
        for root, dirs, _ in os.walk(root_dir):
            # 找到包含 'dose' 或 'drf' 的子文件夹的父文件夹，即认定为病人文件夹
            has_dose_folder = any(['dose' in d.lower() or 'drf' in d.lower() for d in dirs])
            if has_dose_folder:
                patient_dirs.append(root)

        # 去重
        patient_dirs = list(set(patient_dirs))
        
        if not patient_dirs:
            print(f"[警告] 在 {root_dir} 未发现有效的病人文件夹结构。")
            continue

        for p_dir in tqdm(patient_dirs, desc=f"处理 {os.path.basename(root_dir)}"):
            process_patient(p_dir, config.OUTPUT_DIR, config.TARGET_SIZE)
            
    print(f"[*] 所有数据源清洗完毕，结果已汇总至: {config.OUTPUT_DIR}")

if __name__ == '__main__':
    main()
