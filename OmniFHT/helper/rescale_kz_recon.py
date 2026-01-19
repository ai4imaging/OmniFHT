import numpy as np
import torch
import mrcfile
import os
pth = '/home/yez/ODT/rebuttal_kz_ignoreDC/bubble1/'
# pth = "/home/yez/ODT/rebuttal/others/bubble1_output/"
# "/home/yez/ODT/rebuttal_kz_ignoreDC/bubble1/reconstruct(bubble1).mrc"
name = 'reconstruct(bubble1).mrc'
# name = 'eval_volume.mrc'
save_name = 'reconstruct(bubble1)_scale.mrc'

def read_mrc_volume(filename):
    with mrcfile.open(filename, mode='r') as mrc:
        # data 默认形状为 (Z, Y, X)
        data = mrc.data.copy()

        # 获取元数据（对于 3D Object 很重要）
        voxel_size = mrc.voxel_size  # 像素间距 (Angstrom 或 um)
        header = mrc.header  # 包含空间组、原点等信息

    return data


def save_mrcs(data, filename):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        # 将文件声明为 Image Stack (这会设置特定的 header 标志)
        mrc.set_image_stack()
        # (可选) 设置像素大小，例如 1.0 A/pixel
        mrc.voxel_size = 1.0
    print(f"Saved stack to {filename}")


volume = read_mrc_volume(os.path.join(pth, name))

lambda_nm = 0.5328  # 波长 (um)
RI_medium = 1.33  # 介质折射率 (n0)
pixel_size_um = 5.86 / 40  # 实际像元大小 (um)

# k0 是真空中的波数, k_m 是介质中的波数矢量模长
k0 = 2 * np.pi / lambda_nm
k_m = k0 * RI_medium
D=volume.shape[0]
pixel_size_k = (2 * np.pi) / (pixel_size_um * D)
norm_coef = D * D * D * pixel_size_k / D / D / (4 * np.pi * np.pi)
volume = volume * k_m



f = -volume * norm_coef * 4 * np.pi
deltaRI = (RI_medium * np.sqrt(
    1 + f * (lambda_nm / (RI_medium * 2 * np.pi)) ** 2)) - RI_medium

save_mrcs(deltaRI + 1.33, os.path.join(pth, save_name))


