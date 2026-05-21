import os
import re
import glob
import torch
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 解决中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 配置区域
# =====================================================================
ROOT_DIR = "I:/processed_data_trido/train"


# =====================================================================

def extract_info(filepath):
    filename = os.path.basename(filepath)
    z = int(m.group(1)) if (m := re.search(r'_Z(\d+)', filename)) else -1
    d = int(m.group(1)) if (m := re.search(r'_D(\d+)', filename)) else -1
    return z, d, filename


def load_slice_data(filepath):
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=True)
        # 兼容 [Part, Noise, Clean] 结构
        c_tensor = data[1].float()
        t_tensor = data[2].float()
        part_idx = int(data[0].mean().item())
        return t_tensor.squeeze().numpy(), c_tensor.squeeze().numpy(), part_idx
    except Exception as e:
        return None, None, -1


class PETWorkstation:
    def __init__(self, root_dir):
        self.patient_dirs = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.p_idx, self.d_idx, self.z_idx = 0, 0, 0
        self.cache, self.file_map, self.doses, self.zs = {}, {}, [], []

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(bottom=0.2)

        self.im1 = self.ax1.imshow(np.zeros((256, 256)), cmap='gray')
        self.im2 = self.ax2.imshow(np.zeros((256, 256)), cmap='gray')
        self.ax1.axis('off');
        self.ax2.axis('off')

        self.slider = Slider(plt.axes([0.25, 0.05, 0.6, 0.03]), 'Z轴层级', 0, 100, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_update)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.load_patient()

    def load_patient(self):
        p_dir = self.patient_dirs[self.p_idx]
        self.patient_id = os.path.basename(p_dir)
        pt_files = glob.glob(os.path.join(p_dir, "*.pt"))

        self.file_map, self.cache = {}, {}
        doses_set, zs_set = set(), set()
        for f in pt_files:
            z, d, _ = extract_info(f)
            if z == -1 or d == -1: continue
            doses_set.add(d);
            zs_set.add(z)
            if d not in self.file_map: self.file_map[d] = {}
            self.file_map[d][z] = f

        self.doses, self.zs = sorted(list(doses_set)), sorted(list(zs_set))
        if not self.zs: return

        # 强制重置索引到中间，防止越界
        self.z_idx = len(self.zs) // 2

        # 重置滑动条范围
        self.slider.valmax = len(self.zs) - 1
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.slider.set_val(self.z_idx)

        self.update_view()

    def update_view(self):
        if not self.zs or not self.doses: return

        # 边界保护
        self.z_idx = max(0, min(self.z_idx, len(self.zs) - 1))

        z_val = self.zs[self.z_idx]
        d_val = self.doses[self.d_idx % len(self.doses)]  # 剂量环形切换

        filepath = self.file_map.get(d_val, {}).get(z_val)

        if filepath:
            if filepath not in self.cache: self.cache[filepath] = load_slice_data(filepath)
            t_img, c_img, part = self.cache[filepath]

            if t_img is not None:
                self.im1.set_data(t_img);
                self.im2.set_data(c_img)
                self.im1.set_clim(0, t_img.max() * 0.8)
                self.im2.set_clim(0, c_img.max() * 0.8)

                self.ax1.set_title(f"Target (Full Dose) | Z={z_val}")
                self.ax2.set_title(f"Condition (1/{d_val} Dose)\n{os.path.basename(filepath)}")

        self.fig.canvas.draw_idle()

    def on_slider_update(self, val):
        self.z_idx = int(val)
        self.update_view()

    def on_scroll(self, event):
        """【修复 Bug】正确处理滚动层级"""
        if event.button == 'up':
            self.z_idx = min(self.z_idx + 1, len(self.zs) - 1)
        elif event.button == 'down':
            self.z_idx = max(self.z_idx - 1, 0)  # 这里原来是 - 0，已修复
        self.slider.set_val(self.z_idx)

    def on_key(self, event):
        if event.key in ['w', 'up']:
            self.d_idx = (self.d_idx + 1) % len(self.doses)
        elif event.key in ['s', 'down']:
            self.d_idx = (self.d_idx - 1) % len(self.doses)
        elif event.key in ['d', 'right']:
            self.p_idx = min(self.p_idx + 1, len(self.patient_dirs) - 1); self.load_patient(); return
        elif event.key in ['a', 'left']:
            self.p_idx = max(self.p_idx - 1, 0); self.load_patient(); return
        self.update_view()


if __name__ == '__main__':
    PETWorkstation(ROOT_DIR)
    plt.show()