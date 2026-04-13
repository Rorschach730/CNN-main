import os
import pydicom
import numpy as np
import napari
from magicgui import magicgui
from tkinter import filedialog, Tk
import sys

def load_patient_folder(folder_path):
    """读取特定文件夹下的 DICOM/IMA 序列"""
    try:
        # [核心修改]：同时放行 .dcm 和 .ima 扩展名 (忽略大小写)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(('.dcm', '.ima'))]
        if not files:
            return None, None

        slices = []
        for f in files:
            try:
                # pydicom 底层完全支持解析 .IMA 格式
                ds = pydicom.dcmread(f)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except:
                continue

        if not slices:
            return None, None

        # 排序：优先按 Z 轴坐标，没有则按实例编号
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if 'ImagePositionPatient' in x else int(
            getattr(x, 'InstanceNumber', 0)))

        # 提取像素数据
        volume = np.stack([s.pixel_array for s in slices])

        # 获取缩放比例 (Z, Y, X)
        ps = slices[0].PixelSpacing
        st = getattr(slices[0], 'SliceThickness', ps[0])
        spacing = (float(st), float(ps[0]), float(ps[1]))

        return volume, spacing
    except Exception as e:
        print(f"读取文件夹时出错: {e}")
        return None, None


def find_dicom_folders(root_dir):
    """递归查找所有包含 .dcm 或 .IMA 文件的文件夹"""
    dicom_folders = {}
    print("正在扫描目录，请稍候...")

    for root, dirs, files in os.walk(root_dir):
        # [核心修改]：文件夹判定逻辑同步放行 .IMA
        if any(f.lower().endswith(('.dcm', '.ima')) for f in files):
            rel_path = os.path.relpath(root, root_dir)
            display_name = rel_path if rel_path != "." else os.path.basename(root)
            dicom_folders[display_name] = root

    return dicom_folders


def main():
    root_tk = Tk()
    root_tk.withdraw()
    root_dir = filedialog.askdirectory(title="选择包含 DICOM/IMA 数据的总文件夹")
    root_tk.destroy()

    if not root_dir:
        print("未选择路径，程序退出。")
        return

    folder_map = find_dicom_folders(root_dir)

    if not folder_map:
        print(f"在 {root_dir} 及其子目录下未发现任何 DICOM 或 IMA 文件！")
        return

    # 创建查看器
    viewer = napari.Viewer(title=f"多中心 PET 浏览器 - {os.path.basename(root_dir)}")

    @magicgui(
        call_button="加载选中序列",
        layout="horizontal",
        folder_display={
            "choices": sorted(list(folder_map.keys())),
            "label": "选择序列:"
        }
    )
    def series_selector(folder_display):
        actual_path = folder_map[folder_display]
        print(f"正在尝试加载: {actual_path}")

        volume, spacing = load_patient_folder(actual_path)

        if volume is not None:
            viewer.layers.clear()
            viewer.add_image(
                volume,
                name=folder_display,
                scale=spacing,
                colormap='gray',
                rendering='mip'
            )
            viewer.reset_view()
            print(f"成功加载: {folder_display}")
        else:
            print("加载失败：该路径下未找到有效像素数据。")

    viewer.window.add_dock_widget(series_selector, area='bottom', name="序列选择器")

    print(f"\n扫描完成！共发现 {len(folder_map)} 个影像序列。")
    napari.run()

if __name__ == "__main__":
    main()