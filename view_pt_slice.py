import os
import re
import glob
import torch
import numpy as np

import matplotlib

# 强行绕过 PyCharm 的静态拦截，呼出原生交互 GUI
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 解决中文字体乱码
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 配置区域：请填写包含多个病人文件夹的父目录
# =====================================================================
ROOT_DIR = "I:/processed_data_trido/train"


# =====================================================================

def extract_info(filepath):
    filename = os.path.basename(filepath)
    z = int(m.group(1)) if (m := re.search(r'_Z(\d+)', filename)) else -1
    d = int(m.group(1)) if (m := re.search(r'_D(\d+)', filename)) else -1
    part = int(m.group(1)) if (m := re.search(r'_Part(\d+)_', filename)) else -1
    return z, d, part, filename


def load_slice_data(filepath):
    """静默读取张量数据并修正映射关系"""
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=True)

        # 【核心修正】：由于你的数据是 [cond_crop, target_crop] 保存的
        # data[0] 是低剂量 (Condition), data[1] 是全剂量 (Target)
        if hasattr(data, 'dim') and data.dim() == 3:
            c_tensor = data[0]
            t_tensor = data[1]
        else:
            c_tensor = data[0]
            t_tensor = data[1]

        part = int(m.group(1)) if (m := re.search(r'_Part(\d+)_', filepath)) else -1

        return t_tensor.squeeze().numpy(), c_tensor.squeeze().numpy(), part
    except Exception:
        return None, None, -1


class PETWorkstation:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patient_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not self.patient_dirs:
            print(f"[错误] 在 {root_dir} 下未找到任何病人文件夹！")
            exit()

        self.p_idx = 0
        self.d_idx = 0
        self.z_idx = 0

        self.cache = {}
        self.file_map = {}
        self.doses = []
        self.zs = []
        self.patient_id = ""

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(bottom=0.2)

        self.im1 = self.ax1.imshow(np.zeros((256, 256)), cmap='gray')
        self.im2 = self.ax2.imshow(np.zeros((256, 256)), cmap='gray')
        self.ax1.axis('off')
        self.ax2.axis('off')

        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Z轴切片', 0, 100, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.load_patient()

    def load_patient(self):
        p_dir = self.patient_dirs[self.p_idx]
        self.patient_id = os.path.basename(p_dir)
        pt_files = glob.glob(os.path.join(p_dir, "*.pt"))

        self.file_map.clear()
        self.cache.clear()
        doses_set, zs_set = set(), set()

        for f in pt_files:
            z, d, part, fname = extract_info(f)
            if z == -1 or d == -1: continue
            doses_set.add(d);
            zs_set.add(z)
            if d not in self.file_map: self.file_map[d] = {}
            self.file_map[d][z] = f

        self.doses = sorted(list(doses_set))
        self.zs = sorted(list(zs_set))

        if not self.zs:
            self.ax1.set_title(f"病人 {self.patient_id} 无有效数据")
            self.fig.canvas.draw_idle()
            return

        self.d_idx = 0
        self.z_idx = len(self.zs) // 2

        self.slider.valmax = len(self.zs) - 1
        self.slider.ax.set_xlim(0, len(self.zs) - 1)
        self.slider.set_val(self.z_idx)

        self.update_view()

    def update_view(self):
        if not self.zs or not self.doses: return

        z_val = self.zs[self.z_idx]
        d_val = self.doses[self.d_idx]

        filepath = self.file_map.get(d_val, {}).get(z_val)

        if filepath:
            if filepath not in self.cache:
                self.cache[filepath] = load_slice_data(filepath)
            t_img, c_img, part = self.cache[filepath]

            if t_img is not None:
                self.im1.set_data(t_img);
                self.im2.set_data(c_img)
                self.im1.set_clim(vmin=0, vmax=t_img.max() * 0.8)
                self.im2.set_clim(vmin=0, vmax=c_img.max() * 0.8)

                p_name = {0: "脑部", 1: "胸部(心脏)", 2: "腹部/盆腔"}.get(part, f"类别{part}")

                self.fig.suptitle(f"当前病人: [ {self.patient_id} ]  ({self.p_idx + 1}/{len(self.patient_dirs)})\n"
                                  f"操作提示: [A/D] 切病人 | [W/S] 切剂量 | [鼠标滚轮] 切层级",
                                  fontsize=12, fontweight='bold', color='darkblue')

                self.ax1.set_title(f"Target (Full Dose 静态基准)\n部位: {p_name} | Z={z_val}")
                self.ax2.set_title(f"Condition (1/{d_val} Dose 动态输入)\n{os.path.basename(filepath)}")
        else:
            self.ax2.set_title(f"1/{d_val} 剂量在此层缺失数据")

        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        self.z_idx = int(val)
        self.update_view()

    def on_scroll(self, event):
        if event.button == 'up':
            self.z_idx = min(self.z_idx + 1, len(self.zs) - 1)
        elif event.button == 'down':
            self.z_idx = max(self.z_idx - 0, 0)
        self.slider.set_val(self.z_idx)

    def on_key(self, event):
        if event.key in ['up', 'w']:
            self.d_idx = (self.d_idx + 1) % len(self.doses)
            self.update_view()
        elif event.key in ['down', 's']:
            self.d_idx = (self.d_idx - 1) % len(self.doses)
            self.update_view()
        elif event.key in ['right', 'd']:
            self.p_idx = min(self.p_idx + 1, len(self.patient_dirs) - 1)
            self.load_patient()
        elif event.key in ['left', 'a']:
            self.p_idx = max(self.p_idx - 1, 0)
            self.load_patient()


if __name__ == '__main__':
    viewer = PETWorkstation(ROOT_DIR)
    plt.show()