# OmniFHT: Pose-Free 3D Quantitative Phase Imaging of Flowing Cells

**OmniFHT** is a computational framework for **pose-free 3D quantitative phase imaging (QPI)** of flowing biological cells using **Fourier Holographic Tomography (FHT)**.  
It jointly estimates unknown **3D refractive-index (RI) distributions** and **cell poses** directly from 2D holographic projections, without requiring controlled rotation or synchronization hardware.

The codebase is **derived from and extends [cryoDRGN](https://github.com/ml-struct-bio/cryodrgn)** with physics-aware forward models, modified network architectures, and customized training loops for FHT.

---

## 1. Features

- **Physics-aware 3D reconstruction**  
  - Implements the **Fourier Diffraction Theorem** under the **Rytov (weak scattering) approximation**.  
  - Uses an implicit neural representation (INR) to model a continuous 3D scattering potential.

- **Pose-free joint inference**  
  - Simultaneous optimization of **3D volume**, **orientation (SO(3))**, and **in-plane translation**.  
  - Coarse-to-fine hierarchical pose search for robust initialization in complex rotational scenarios.

- **Flexible data handling**  
  - Supports standard **`.mrcs`** stacks of holographic projections.  
  - Test datasets are provided for quick end-to-end experiments.

- **Analysis utilities**  
  - Scripts for pose analysis, FSC computation, and bubble/RI distribution diagnostics.

---

## 2. Repository Structure

At the top level, the repository is organized as:

```text
OmniFHT/
├── OmniFHT/
│   ├── cryodrgn/              # Modified cryoDRGN core
│   │   ├── commands/          # Main CLI entry points (incl. abinit_homo.py)
│   │   └── ...                # Model, dataset, and training utilities
│   ├── analyze_pose_2D.py     # 2D pose analysis utilities
│   ├── analyze_pose_3D.py     # 3D pose / orientation analysis
│   ├── bubble_analyze.py      # Bubble / RI distribution diagnostics
│   ├── bubble_analyze_multi.py
│   ├── crop_center.py         # Cropping / centering helper
│   ├── fsc.py                 # FSC computation scripts
│   └── ...
├── test_data/                 # Example hologram stacks (.mrcs)
│   ├── 3_1.mrcs
│   ├── 4_1_64.mrcs
│   ├── 4_2_64.mrcs
│   ├── 4_3_64.mrcs
│   └── 7_1_64.mrcs
└── README.md
```

---

## 3. Installation

```
conda create -n omnifht python=3.9 -y
conda activate omnifht
pip install cryodrgn numpy scipy mrcfile tqdm matplotlib
git clone https://github.com/ai4imaging/OmniFHT.git
cd OmniFHT
```

---

## 4. Data Format

OmniFHT uses `.mrcs` stacks where each slice is a single holographic projection.

Example test data:

```
test_data/4_1_64.mrcs
```

---

## 5. Running Homogeneous Ab Initio Reconstruction

Example:

```
python OmniFHT/cryodrgn/commands/abinit_homo.py     test_data/4_1_64.mrcs     -b 4     -o runs/4_1_64_demo     -n 201     --checkpoint 100     --lr 0.01     --uninvert-data
```

For optical system settings, you can edit it in OmniFHT/cryodrgn/lattace.py line 29-31. 

```
lambda_ = 0.5328
RI_ = 1.33
pixel_size = 5.86 / 40
```

---

## 6. Outputs

```
runs/4_1_64_demo/
│── weights.pkl
│── vol.mrc
│── poses.pkl
│── run.log
└── checkpoints/
```

## 7. Citation

If you use OmniFHT:

*Pose-Free 3D Quantitative Phase Imaging of Flowing Cellular Populations*, Ye et al., 2025. (In preparation)

---

## 8. Support

Please open GitHub Issues for bugs, questions, or feature requests.
