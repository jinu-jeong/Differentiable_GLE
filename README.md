# DiffGLE

This repository contains the code and compact result data for DiffGLE, a differentiable generalized Langevin equation (GLE) approach for correcting coarse-grained molecular dynamics. The conservative coarse-grained force is kept fixed, while a trainable colored-noise filter defines the fluctuating force and the friction memory through the fluctuation-dissipation theorem.

This `v2` branch is the reproducibility branch for the revised manuscript. It is scoped to the systems shown in the paper:

- bulk CO2,
- bulk H2O,
- a star-polymer no-force memory benchmark.

Large raw all-atom trajectories, exploratory cluster run folders, generated image dumps, and personal logs are intentionally not included.

## What Is Included

- `main.py`, `force.py`, `utility.py`, `preprocess.py`: differentiable CG/GLE code for the bulk molecular-fluid examples.
- `Data/CO2/` and `Data/H2O/`: compact preprocessed inputs for the one-bead-per-molecule fluid demos, including COM initial conditions, reference observables, simulation boxes, and fixed tabulated CG pair potentials.
- `paper_data/`: compact saved optimization histories, production observables, gradient-audit arrays, and supplementary source data used to regenerate the manuscript figures.
- `scripts/`: figure-reproduction scripts for the main results, gradient-mechanics diagnostics, FDT closure, and CG pair potentials.

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

Install the PyTorch build appropriate for your machine before the full run if you need a specific CUDA version. The original molecular-fluid training runs used GPU acceleration, but all figure scripts are CPU-only.

## Reproduce Manuscript Figures

Regenerate the compact main-results figure for CO2, H2O, and the star-polymer no-force benchmark:

```bash
python scripts/make_paper_figures.py
```

Regenerate the CO2 gradient-mechanics diagnostic:

```bash
python scripts/plot_gradient_mechanics.py
```

Regenerate supplementary FDT-closure and conservative-potential figures:

```bash
python scripts/plot_fdt_closure.py
python scripts/plot_pair_potentials.py
```

The generated files are written under `figures/generated/`.

## Paper Run Settings

The compact arrays in `paper_data/` are the recommended way to reproduce the plotted manuscript results exactly. The corresponding settings are stored next to each saved history:

| system | paper-data folder | key setting summary |
|---|---|---|
| CO2 | `paper_data/optimization_histories/co2_symdetach_lr3e5_iter372/` | free colored-noise filter, fixed CO2 CG-CG potential, adjoint-assisted optimization, SGD, selected iteration 372 |
| H2O | `paper_data/optimization_histories/h2o_original_fromscratch/` | free colored-noise filter, fixed H2O CG-CG potential, adjoint-assisted optimization, Adam learning-rate schedule, 300 iterations |
| star polymer | `paper_data/optimization_histories/star_polymer_noforce_fromscratch_support700/` and `paper_data/optimization_histories/star_polymer_noforce_continuation_support700/` | no conservative force, fixed-noise direct optimization, 64 replicas, 7 particles per replica, long-support free filter |

See `paper_data/optimization_histories/optimization_history_manifest.md` for the exact hyperparameters and manuscript-facing metrics.

## Rerun Bulk-Fluid Training Demos

The compact preprocessed data under `Data/` are sufficient for the default H2O and CO2 differentiable MD demos:

```bash
python main.py --system CO2 --device cuda:0 --iterations 300
python main.py --system H2O --device cuda:0 --iterations 300
```

Use `--device cpu` for small smoke tests. CPU training is much slower and is intended only for debugging.

Outputs are written to `Result/<system>/`.

## Data Notes

For the molecular fluids, the CG mapping is one center-of-mass bead per molecule. The fixed conservative interactions are the tabulated CG-CG pair potentials in `Data/H2O/CG/CG_CG.pot` and `Data/CO2/CG/CG_CG.pot`. The force column from each table is interpolated during simulation and is not trained during DiffGLE optimization.

The star-polymer no-force benchmark is stored as compact target curves and learned filter/kernel snapshots in `paper_data/optimization_histories/`. It is included for manuscript figure reproduction and for documenting the final no-force memory-correction result.

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
