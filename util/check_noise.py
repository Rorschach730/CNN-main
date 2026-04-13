import os
import random
import numpy as np
import matplotlib.pyplot as plt


def verify_physical_degradation(data_dir='../processed_data_sinogram_new/train'):
    """
    [物理特性抽检] 随机加载验证生成的低剂量 PET 数据
    """
    if not os.path.exists(data_dir):
        print(f"目录不存在: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if not files:
        print("未找到任何 .npy 数据文件。")
        return

    # 随机抽取一个病人的数据卷
    sample_file = random.choice(files)
    file_path = os.path.join(data_dir, sample_file)
    print(f"正在抽检物理废墟: {sample_file}")

    data = np.load(file_path, allow_pickle=True).item()
    img_noisy = data['input']
    img_clean = data['target']
    meta = data['metadata']

    # 抽取含有有效解剖结构的中间切片
    depth = img_clean.shape[0]
    slice_idx = depth // 2

    # 规避全黑切片
    while np.max(img_clean[slice_idx]) < 0.01 and slice_idx < depth - 1:
        slice_idx += 1

    clean_slice = img_clean[slice_idx]
    noisy_slice = img_noisy[slice_idx]

    # 计算绝对残差，放大物理退化的空间异质性
    residual = np.abs(noisy_slice - clean_slice)

    # 可视化渲染
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Patient: {meta['pid']} | Dose: {meta['dose']} | Slice: {slice_idx}/{depth}", fontsize=14)

    im0 = axes[0].imshow(clean_slice, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title("Ground Truth (Target)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(noisy_slice, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Physics Degraded (Input)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 残差图：使用专门的 cmap 凸显中心爆裂
    im2 = axes[2].imshow(residual, cmap='magma', vmin=0, vmax=residual.max() * 0.8)
    axes[2].set_title("Absolute Residual (Noise Map)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_physical_degradation()