# DiffGLE

This repository contains the code and compact result data for DiffGLE, a differentiable generalized Langevin equation (GLE) approach for correcting coarse-grained molecular dynamics. The conservative coarse-grained force is kept fixed, while a trainable colored-noise filter defines the fluctuating force and the friction memory through the fluctuation-dissipation theorem.

This `v2` branch is the reproducibility branch for the revised manuscript. It is organized to regenerate the plotted results without storing large raw all-atom trajectories or personal run logs.

## What Is Included

- `main.py`, `main_confined_H2O.py`, `force.py`, `utility.py`, `preprocess.py`:
  differentiable CG/GLE simulation code for the molecular-fluid examples.
- `Data/`:
  compact preprocessed inputs for the H2O, CO2, and confined-H2O demos, including COM initial conditions, reference observables, boxes, and fixed tabulated CG potentials.
- `paper_data/`:
  compact saved optimization histories, production observables, gradient-audit arrays, and supplementary source data used to generate the manuscript figures.
- `scripts/`:
  figure-reproduction scripts for the main results, gradient-mechanics diagnostics, FDT closure, and CG pair potentials.

Generated output is written to `figures/generated/` or `Result/`; both are intentionally ignored by git.

## Installation

For reproducing the paper figures from the bundled compact arrays:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-figures.txt
```

For rerunning the differentiable MD demos, install the full dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your machine before the full run if you need a specific CUDA version. The original molecular-fluid training was run on GPU, but the figure scripts are CPU-only.

## Reproduce Manuscript Figures

Regenerate the compact main-results figure:

```bash
python scripts/make_paper_figures.py
```

Regenerate the gradient-mechanics diagnostic:

```bash
python scripts/plot_gradient_mechanics.py
```

Regenerate supplementary FDT and conservative-potential figures:

```bash
python scripts/plot_fdt_closure.py
python scripts/plot_pair_potentials.py
```

The generated files are written under `figures/generated/`.

## Rerun Molecular-Fluid Training Demos

The compact preprocessed data under `Data/` are sufficient for the default H2O and CO2 demos:

```bash
python main.py --system CO2 --device cuda:0 --iterations 300
python main.py --system H2O --device cuda:0 --iterations 300
```

Use `--device cpu` for small smoke tests. CPU training is much slower and is intended only for debugging.

Outputs are written to `Result/<system>/`.

## Data Notes

The repository includes compact observables and saved optimization histories, not the large raw atomistic trajectories. For the molecular fluids, the CG mapping is one center-of-mass bead per molecule. The fixed conservative interactions are the tabulated CG-CG pair potentials in `Data/H2O/CG/CG_CG.pot` and `Data/CO2/CG/CG_CG.pot`.

The star-polymer no-force benchmark in `paper_data/optimization_histories/` is stored as compact curves and learned filter/kernel snapshots. It is included for figure reproduction and does not require the exploratory polymer run directories used during development.

## Repository Hygiene

This branch intentionally excludes:

- raw all-atom trajectories,
- cluster job files,
- personal logs and notebooks,
- generated per-iteration PNG dumps,
- local virtual environments and caches.

## Citation

If you use this code, please cite the DiffGLE manuscript associated with this repository. A formal citation entry will be added when the revised manuscript record is finalized.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
