import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import logging

# 引入你的项目路径
sys.path.append('/home/yez/ODT/cryodrgn_odt/cryodrgn')

import config as config_
import models as models  # 如果你使用的是 models_wobias.py，请修改这里为 import models_wobias as models
from lattice import Lattice
from mrc import MRCFile
import fft

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Load weights and evaluate volume")
    parser.add_argument(
        "--weights",
        type=str,
        # default="/home/yez/ODT/rebuttal_kz_ignoreDC/bubble1/weights.pkl",
        default="/home/yez/ODT/rebuttal_kz/fig2/m/weights.pkl",
        required=False,
        help="Path to the weights file (e.g., weights.100.pkl or weights.pkl)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        # default="/home/yez/ODT/rebuttal_kz_ignoreDC/bubble1/config.yaml",
        default = "/home/yez/ODT/rebuttal_kz/fig2/m/config.yaml",
        help="Path to the config.yaml file generated during training"
    )
    parser.add_argument(
        "-o", "--outfile",
        type=str,
        # default="/home/yez/ODT/rebuttal_kz_ignoreDC/bubble1/eval_volume.mrc",
        default="/home/yez/ODT/rebuttal_kz/fig2/m/eval_volume.mrc",
        help="Output filename for the .mrc volume"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "-D",
        type=int,
        default=None,
        help="Override box size D (optional)"
    )
    parser.add_argument(
        "--force_dc_zero",
        action="store_true",
        help="Manually set the DC component (center frequency) to 0 to remove background drift/artifacts."
    )
    return parser.parse_args()

from nibabel.viewers import OrthoSlicer3D
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    # 1. 加载配置
    logger.info(f"Loading config from {args.config}")
    cfg = config_.load(args.config)

    # 获取必要的参数
    lattice_args = cfg["lattice_args"]
    model_args = cfg["model_args"]

    D = args.D if args.D is not None else lattice_args["D"]
    extent = lattice_args["extent"]

    # 2. 初始化 Lattice
    logger.info(f"Initializing Lattice with D={D}, extent={extent}")
    lattice = Lattice(D, extent=extent, device=device)

    # 3. 构建模型结构
    logger.info("Building model...")
    # 处理激活函数映射
    activation_map = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}
    act_name = model_args.get("activation", "relu")  # 默认 relu
    activation = activation_map.get(act_name, nn.ReLU)

    model = models.get_decoder(
        in_dim=3,
        D=D,
        layers=model_args["qlayers"],  # 注意：这里用 qlayers 还是 layers 取决于你的 config 结构，通常 decoder 用 layers
        dim=model_args["qdim"],  # 同上，通常用 dim
        domain=model_args["domain"],
        enc_type=model_args["pe_type"],
        enc_dim=model_args["pe_dim"],
        activation=activation,
        feat_sigma=model_args["feat_sigma"]
    )
    model.to(device)

    # 4. 加载权重
    logger.info(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)

    # 处理 DataParallel 的 key (如果有 module. 前缀)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # 获取 normalization 参数
    norm = checkpoint.get("norm", [0.0, 1.0])  # 默认 mean=0, std=1
    logger.info(f"Using normalization: {norm}")

    # 5. 执行 Eval Volume
    logger.info("Evaluating volume...")

    # 如果你想在这里 Hack 强制 DC 为 0，我们需要手动调用内部逻辑
    # 因为直接调用 model.eval_volume 封装了所有步骤

    # if args.force_dc_zero:
    logger.warning("Applying FORCE DC ZERO patch...")
    # 为了强制置零 DC，我们需要 monkey patch 或者手动执行 eval_volume 的逻辑
    # 这里演示手动执行 PositionalDecoder 的逻辑 (假设你是 PositionalDecoder)
    if isinstance(model, models.PositionalDecoder):
        # 自定义 evaluation 逻辑
        vol = eval_volume_custom(model, lattice, D, lattice.circle_mask_radius, extent, norm, device)
    else:
        logger.warning("Model is not PositionalDecoder, using standard eval_volume (force-dc-zero might not work)")
        vol = model.eval_volume(lattice.coords, D, lattice.circle_mask_radius, extent, norm)
    # else:
    #     # 标准调用
    #     vol = model.eval_volume(lattice.coords, D, lattice.circle_mask_radius, extent, norm)

    # 6. 保存结果
    logger.info(f"Saving reconstruction to {args.outfile}")
    MRCFile.write(args.outfile, vol)
    logger.info("Done!")


def eval_volume_custom(model, lattice, D, R, extent, norm, device):
    """
    这是一个修改版的 eval_volume，允许强制去除 DC 分量
    """
    coords = lattice.coords
    vol_f = torch.zeros((D, D, D), dtype=torch.complex64)

    zval = None
    zdim = 0
    if zval is not None:
        zdim = len(zval)
        z = torch.tensor(zval, dtype=torch.float32, device=coords.device)

    # 逐层预测频谱
    for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
        x = coords + torch.tensor([0, 0, dz], device=coords.device)
        with torch.no_grad():
            y = model.forward(x)  # 预测频谱
            y = y.view(D, D)
        vol_f[i] = y

    # 反归一化
    print('norm', norm)
    vol_f = vol_f * norm[1] + norm[0]

    # # === 核心修改：强制 DC 为 0 ===
    # center = D // 2
    # print(f"Forcing DC component at index [{center}, {center}, {center}] to 0.")
    # vol_f[center, center, center] = 0 + 0j
    # ==========================

    # 掩模处理
    center_vec = torch.tensor([D // 2, D // 2, D // 2], dtype=torch.float32, device=device)
    z_idx, y_idx, x_idx = torch.meshgrid(
        torch.arange(D, dtype=torch.float32, device=device),
        torch.arange(D, dtype=torch.float32, device=device),
        torch.arange(D, dtype=torch.float32, device=device),
        indexing='ij'
    )
    distance_from_center = torch.sqrt(
        (x_idx - center_vec[0]) ** 2 + (y_idx - center_vec[1]) ** 2 + (z_idx - center_vec[2]) ** 2)
    mask = distance_from_center < R
    # OrthoSlicer3D(vol_f.cpu().numpy()).show()
    # vol_f = vol_f - 2.645 * 4#*256*256
    vol_f =vol_f #- 28/2
    # vol_f = vol_f - 0.0446*256
    print('constant shift', 2.645*4, 0.0446*256)
    vol_f = vol_f.to('cuda:0')
    mask = mask.to('cuda:0')
    vol_f_ = torch.where(mask, vol_f, torch.tensor(0, dtype=vol_f.dtype, device=device))
    print('vol f mean', vol_f_.mean())
    # 逆变换
    vol = fft.ihtn_center(
        vol_f_[0:-1, 0:-1, 0:-1]
    )
    const = vol[128, 128, 128]
    print('const', const)
    # vol_f =vol_f - const/10/2
    vol_f = vol_f - const*10/2
    # vol_f = vol_f - 0.0446*256
    print('constant shift', 2.645*4, 0.0446*256)
    vol_f = vol_f.to('cuda:0')
    mask = mask.to('cuda:0')
    vol_f = torch.where(mask, vol_f, torch.tensor(0, dtype=vol_f.dtype, device=device))
    print('vol f mean', vol_f.mean())
    # 逆变换
    vol = fft.ihtn_center(
        vol_f[0:-1, 0:-1, 0:-1]
    )
    # const =
    # OrthoSlicer3D(vol.cpu().numpy()).show()
    return vol#.abs()


if __name__ == "__main__":
    main()
