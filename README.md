# DiffGLE: Differentiable Coarse-Grained Dynamics

## Overview

This repository contains the implementation of **DiffGLE**, a differentiable coarse-grained dynamics framework based on the **Generalized Langevin Equation (GLE)**. The methodology leverages **Automatic Differentiation (AD)** and the **adjoint-state method** to accurately parameterize non-Markovian GLE models for coarse-grained fluids.

The code in this repository corresponds to our paper:

📄 **[DiffGLE: Differentiable Coarse-Grained Dynamics using Generalized Langevin Equation](https://arxiv.org/abs/2410.08424)**  
👨‍🔬 *Authors: Jinu Jeong*, Ishan Nadkarni

> 🔬 **This repository contains the demo code for the APS March Meeting 2025.**

## Features

- **End-to-End Differentiable Simulation**: Differentiable coarse-grained molecular dynamics (CGMD) framework.
- **GLE Parameterization**: Colored-noise ansatz for memory kernels.
- **Adjoint-State Optimization**: Efficient optimization of CG models with memory kernel and colored thermal noise.
- **Validation on Complex Fluids**: Demonstrated on H₂O, CO₂, and confined H₂O.

## Requirements

Install dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

## Quick start

Preprocessed inputs and CG potentials should be under `./Data/`.

```bash
python main.py -s H2O
python main.py -s CO2
python main_confined_H2O.py
```

Run bulk H₂O before confined H₂O (transfer learning uses `Result/H2O/model_state_dict.pth`). Outputs go to `./Result/<system>/`.

**Additional options:** `--device` (default `cuda:0`), `--iterations` / `-n` (default `300`), `--output-dir`, and `--show` (interactive plots; off by default). For confined H₂O: `--pretrained`, `--no-pretrained`.

## Advanced: download data and preprocessing

Use this only if you want to regenerate inputs from the full all-atom trajectories (large files, not required for the default demo).

### 1. Download the dataset

Download from [uofi.box.com](https://uofi.box.com/s/gruyslzav75ibbg0qjlh877f37le78c4) and unpack under `./Data/`:

```
Data/
├── CO2/
│   ├── AA/pos_COM.npy, vel_COM.npy
│   └── CG/CG_CG.pot
├── H2O/
│   ├── AA/pos_COM.npy, vel_COM.npy
│   └── CG/CG_CG.pot
└── Confined_H2O/
    ├── AA/pos_COM.npy, vel_COM.npy, z.npy
    └── CG/CG1_CG1.pot, CG1_GR1.pot
```

### 2. Preprocess all-atom trajectories

Run once to compute reference observables and initial conditions. Outputs are saved in each `Data/<system>/` folder.

```bash
python preprocess.py                  # all systems (default: cpu)
python preprocess.py --system H2O     # one system
python preprocess.py --device cuda:0  # optional GPU (auto-falls back to cpu for large trajectories)
```

**Bulk (`CO2`, `H2O`)** writes: `pos0.npy`, `vel0.npy`, `r_aa.npy`, `rdf_aa.npy`, `msd_aa.npy`, `vacf_aa.npy`, `box.npy`

**Confined (`Confined_H2O`)** writes: `z.npy`, `pos0.npy`, `vel0.npy`, `z_aa.npy`, `density_aa.npy`, `msd_aa.npy`, `vacf_aa.npy`, `vacf_aa_gt.npy`, `box.npy`

## Outputs

Results are written to `./Result/<system>/`:

- `model_state_dict.pth` — trained GLE kernel
- `plot_dict.pkl` — optimization diagnostics
- `optimization_iter_*.png` — per-iteration VACF / filter plots
- `<system>.xyz` — production trajectory

## Repository structure

| File | Description |
|------|-------------|
| `main.py` | Bulk CO₂ / H₂O demo |
| `main_confined_H2O.py` | Confined H₂O demo with transfer learning |
| `preprocess.py` | AA trajectory preprocessing |
| `force.py`, `utility.py`, `integrator.py` | Core simulation and analysis modules |
| `CO2.py`, `H2O.py` | Legacy single-system scripts (superseded by `main.py`) |

## Additional resources

🔍 Other works from our research group: [MultiNano Group](https://github.com/multinanogroup)

## License & copyright

© 2026 Jinu Jeong and Ishan Nadkarni. All rights reserved.

This project is licensed under the MIT License. See the LICENSE file for details.

