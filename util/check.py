import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm


def check_raw_dicom(data_dir='../lung_pet_data'):
    print(f"正在扫描 {data_dir} 下的 DICOM 文件...")

    # 获取所有 .dcm 文件 (递归查找)
    dcm_files = glob.glob(os.path.join(data_dir, "**/*.dcm"), recursive=True)

    if not dcm_files:
        print("错误：未找到任何 .dcm 文件，请检查路径。")
        return

    print(f"找到 {len(dcm_files)} 个 DICOM 文件，开始检查像素值...\n")

    zero_files_count = 0
    # 记录出现问题的病例ID (文件夹名)
    problematic_patients = set()

    for f in tqdm(dcm_files):
        try:
            # 仅读取头部信息和像素数据，不强制加载整个文件以加快速度
            dcm = pydicom.dcmread(f, stop_before_pixels=False)

            # 获取像素数组
            if not hasattr(dcm, 'PixelData'):
                continue

            img = dcm.pixel_array

            # 检查最大值
            max_val = np.max(img)

            if max_val == 0:
                zero_files_count += 1
                patient_id = os.path.basename(os.path.dirname(f))
                problematic_patients.add(patient_id)

                # 打印详细信息（仅前10个，避免刷屏）
                if zero_files_count <= 10:
                    print(f"\n[发现全黑图像] {os.path.basename(f)}")
                    print(f"  - 路径: {f}")
                    print(f"  - Modality: {dcm.get('Modality', 'Unknown')}")
                    print(f"  - Series Description: {dcm.get('SeriesDescription', 'Unknown')}")
                    print(f"  - Instance Number: {dcm.get('InstanceNumber', 'Unknown')}")

        except Exception as e:
            # 忽略非图像文件或读取错误的警告，保持输出整洁
            pass

    print("-" * 50)
    print(f"检查完成。")
    print(f"全黑 DICOM 文件总数: {zero_files_count}")

    if len(problematic_patients) > 0:
        print(f"涉及的病例ID ({len(problematic_patients)} 个):")
        for pid in problematic_patients:
            print(f"  - {pid}")
        print("\n建议：这些全黑的切片通常是定位像(Scout)或无效层。")
        print("prepare_data.py 脚本中的 [DROP] 逻辑会自动过滤掉由这些切片组成的无效序列。")
    else:
        print("未发现全黑的原始 DICOM 文件。")


if __name__ == "__main__":
    check_raw_dicom()