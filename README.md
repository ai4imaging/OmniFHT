# OmniFHT

**OmniFHT** is a framework for **pose-free 3D quantitative phase imaging of flowing cells**, developed as part of the work described in:  

> *Pose-Free 3D Quantitative Phase Imaging of Flowing Cellular Populations*  

This repository is **derived from [cryoDRGN](https://github.com/ml-struct-bio/cryodrgn)**, but has been substantially modified to support the physics of **Fourier Holographic Tomography (FHT)** in flow cytometry.

---

## Overview

OmniFHT extends the cryoDRGN framework by modifying its **model**, **training loop**, and **lattice construction** to explicitly account for the physics of FHT under the weak scattering (Rytov) approximation.  

Key changes include:
- **Model modifications**: Variational autoencoder (VAE) architecture adapted for Fourier-domain implicit neural representations (INRs), enabling continuous 3D scattering potential modeling.  
- **Training procedure**: Iterative joint optimization of cell pose (rotation + in-plane translation) and 3D volume reconstruction using a coarse-to-fine hierarchical search.  
- **Lattice construction**: Incorporation of optical parameters such as wavelength, refractive index, and pixel size for proper Fourier-domain sampling.  

---

## Usage

The main entry point for homogeneous ab initio reconstruction is the **`abinit_homo`** command. This command performs joint pose inference and 3D reconstruction under FHT physics.

Inputs:
- `qpm.mrcs` : List of holographic projections.  

Outputs:
- `output_directory/weights.pkl` : Trained model weights.  
- `output_directory/vol.mrc` : Reconstructed 3D refractive index distribution.  
- `output_directory/poses.pkl` : Inferred poses (rotations and translations).  

---
