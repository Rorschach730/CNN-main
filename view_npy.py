import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random


def view_npy(file_path):
    if not os.path.exists(file_path):
        print(f"[!] 致命错误：文件 {file_path} 不存在。")
        return

    print(f"[*] 正在解构张量文件: {file_path}")
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[!] 加载失败: {e}")
        return

    noisy_vol = data.get('input')
    clean_vol = data.get('target')
    meta = data.get('metadata', {})

    if noisy_vol is None or clean_vol is None:
        print("[!] 数据结构不合法：缺失 'input' 或 'target' 键。")
        return

    d, h, w = noisy_vol.shape

    # [物理修复]: 解除中心锁定，引入 Z 轴全局随机抽取
    random_z = random.randint(0, d - 1)

    print(f"    - 物理维度: {noisy_vol.shape}")
    print(f"    - 元数据字典: {meta}")
    print(f"[*] 正在渲染 Z 轴随机切片 (Layer: {random_z}/{d - 1})...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # [物理修复]: 废除写死的 vmax=1.0，根据当前切片的张量峰值动态调整亮度映射
    vmax_clean = clean_vol[random_z].max()
    # 乘以 0.8 以轻微过曝的方式拉亮暗部细节
    vmax_clean = vmax_clean * 0.8 if vmax_clean > 1e-4 else 1.0

    # Ground Truth (Target)
    axes[0].imshow(clean_vol[random_z], cmap='gray', vmin=0, vmax=vmax_clean)
    axes[0].set_title(f"Ground Truth (Clean) - Slice {random_z}")
    axes[0].axis('off')

    # Input (Noisy)
    vmax_noisy = noisy_vol[random_z].max()
    vmax_noisy = vmax_noisy * 0.8 if vmax_noisy > 1e-4 else 1.0
    axes[1].imshow(noisy_vol[random_z], cmap='gray', vmin=0, vmax=vmax_noisy)

    drf_val = meta.get('drf', 'Unknown')
    drf_str = f"{drf_val:.1f}" if isinstance(drf_val, float) else drf_val
    axes[1].set_title(f"Input (3D TOF-OSEM, DRF={drf_str}) - Slice {random_z}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # 默认在 train 目录下抓取样本进行确诊
        import glob

        search_path = "./processed_data_3d_osem/train/*.npy"
        files = glob.glob(search_path)
        if not files:
            print(f"[!] 路径 {search_path} 下未找到任何 .npy 文件。请通过命令行传入绝对路径。")
            sys.exit(1)
        # 每次运行脚本时也随机抽取一个患者文件
        target_file = random.choice(files)

    view_npy(target_file)