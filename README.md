# Differentiable GLE with memory fields

Coarse-grained (CG) molecular dynamics with a **generalized Langevin equation (GLE)** whose memory
kernel is learned **top-down**, by differentiating the CG simulation so that its dynamics match
all-atom references.

This branch (`memory_field`) adds **memory fields**: a memory kernel that depends on a state variable
of the bead, here the distance $d$ to a wall, $\kappa(t, d)$. The kernel is a Prony approximation, a
sum of damped-oscillator modes realized by auxiliary variables (Markovian embedding), and the four
parameters of every mode are functions of $d$. Because every mode has a non-negative spectrum for any
parameter values, the fluctuation–dissipation theorem holds for any $d$-dependence, and the exact
GLE-BAOAB O step keeps the temperature exact also when beads move between distances. A
position-independent kernel (bulk fluids) is the uniform special case.

- **Bulk water and CO₂:** 16 parameters (4 modes) trained with exact gradients through the simulation
  (standard autograd, memory per step set by the number of modes, not by the memory length).
- **Water between graphene sheets:** a memory field learned in an 80 Å channel reproduces the
  layer-resolved velocity autocorrelation functions (VACFs), transfers without retraining to
  20–40 Å channels, and predicts the residence time of water in the contact layer, which it was not
  trained on.

The previous version, **DiffGLE** ([arXiv:2410.08424](https://arxiv.org/abs/2410.08424)), with a
learnable noise filter trained by an adjoint method, is on the `v2` branch; its pipeline is still
available here as `main.py`.

## Install

```bash
pip install -r requirements.txt          # Python 3.10+, PyTorch with CUDA recommended
```

The all-atom simulations of confined water use LAMMPS through its Python module (GPU package).
Data and outputs go to `Data/` and `Result/` (not tracked).

## Bulk water and CO₂

Inputs: `Data/H2O`, `Data/CO2`. The preprocessed initial states, reference observables and CG pair potentials
are included in the repository; the all-atom trajectories (see *Data* below) are needed only to rerun
`preprocess.py`.

```bash
python preprocess.py --system H2O                    # optional: recompute them from the all-atom trajectory
python main.py -s H2O                                # DiffGLE filter (initial kernel of the modes)
python -m experiments.train_osc -s H2O               # 4 modes, GLE-BAOAB, exact gradients -> Result/H2O_osc_baoab
```

Reproduction scripts for the bulk comparisons are in `experiments/bulk/` (training repeats, filter with
exact gradients, mode-count scan, measured BPTT memory, production evaluation); results go to
`Result/bulk/`.

## Water between graphene sheets

**1. All-atom references** (flexible SPC/E water between frozen graphene sheets $H$ apart; same force
field and lateral box as `Data/Confined_H2O`):

```bash
cd aa/confined_graphene
python build_confined.py 80 2123 H80/start.data      # H = 10 A: build_h10.py
python run_confined.py --data H80/start.data --out H80 --equil 200000 --prod 100000
python make_dataset.py 80                            # -> Data/Confined_H2O_v2_H80 (CG system) + targets/
python make_dataset.py 80 --slab-frame               # -> targets_slab/ (VACFs in the frame of the water slab)
```

The water slab slides as a whole on graphene and its centre-of-mass velocity is not equilibrated in
100 ps, so the training targets are computed in the frame of the slab (`--slab-frame`).

**2. Train a memory field** on the 80 Å channel (starts from the bulk-water kernel
`Result/H2O_osc_baoab`):

```bash
python experiments/confined/train.py --system Confined_H2O_v2_H80 --wall-gap 80 \
    --model knot --groups 2 --target fine --smooth 3 -n 300 --avg-last 50 --prod 50000 \
    --out Result/memory_field_H80
```

`--model knot`: every mode parameter is a piecewise-linear curve in $d$ (20 knots, 2.5–20 Å),
separately parallel and perpendicular to the walls. Comparison models: `--model bins --bins 8 --groups 2`
(one kernel per layer bin) and `--bins 1 --groups 1` (one kernel for all beads).

**3. Transfer to a narrower channel** without retraining (`-n 0`: production only):

```bash
python experiments/confined/train.py --system Confined_H2O_v2_H20 --wall-gap 20 \
    --model knot --groups 2 --target fine -n 0 --prod 50000 \
    --load-state Result/memory_field_H80/thermostat_state_dict.pth --out Result/memory_field_H80_in_H20
# add --superpose to add the effects of both walls
```

**4. Residence in the contact layer** (not a training target):

```bash
python experiments/confined/residence.py aa 20                                  # all-atom
python experiments/confined/residence.py cg 20 Result/memory_field_H80_in_H20   # CG (z_traj.npy)
```

## Data

Bulk and original confined inputs: download from
[uofi.box.com](https://uofi.box.com/s/gruyslzav75ibbg0qjlh877f37le78c4) and unpack under `./Data/`:

```
Data/
├── CO2/            AA/pos_COM.npy, vel_COM.npy;      CG/CG_CG.pot
├── H2O/            AA/pos_COM.npy, vel_COM.npy;      CG/CG_CG.pot
└── Confined_H2O/   AA/pos_COM.npy, vel_COM.npy, z.npy; CG/CG1_CG1.pot, CG1_GR1.pot
```

`python preprocess.py` computes the reference observables and initial states in each `Data/<system>/`.
The confined systems `Confined_H2O_v2_H<H>` are made by `aa/confined_graphene/make_dataset.py` (step 1
above); `experiments/confined/aa_targets.py` computes the same targets for the original 80 Å trajectory
`Data/Confined_H2O`, which serves as a cross-check.

## Repository structure

```
diffgle/                    library
  constants.py              units, temperature, paths
  systems.py                SYSTEMS settings table + load_system()
  potentials.py             tabulated CG pair forces, neighbour list
  thermostats/
    damped_osc.py           damped-oscillator modes (Markovian embedding, exact FDT)
    conv.py                 DiffGLE free filter (GLE_TS) and its exact adjoint (ConvGLE)
    langevin.py             Markovian Langevin
  memory_field.py           memory field kappa(t, d): MemoryField, MemoryFieldSuperposed, BinnedDampedOsc,
                            tabulated exact O step, FrozenWallBAOAB, layer-conditioned VACF
  integrators.py            BAOAB (+ exact GLE O step), VV + Euler (DiffGLE)
  observables.py            VACF, MSD, RDF, density, temperature
experiments/
  train_osc.py              bulk: damped-oscillator modes + GLE-BAOAB + exact gradients
  eval_osc.py, check_*.py, ablation_*.py, diagnose_paper_adjoint.py   bulk: gradient, loss and integrator checks
  bulk/                     bulk: reproduction runs (repeats, mode scan, memory, production evaluation)
  confined/
    train.py                memory field (and comparison kernels) for confined water
    aa_targets.py           all-atom density and layer-conditioned VACF targets
    residence.py            contact-layer residence and layer exchange
aa/confined_graphene/       all-atom runs: build, run (LAMMPS), make CG system and targets
tests/                      python -m pytest tests (memory field, damped-oscillator kernel, BAOAB, adjoint)
main.py, preprocess.py      DiffGLE pipeline and preprocessing
```

## Tests

```bash
python -m pytest tests
```

`tests/test_memory_field.py` checks that every tabulated O-step map keeps $\mathcal N(0, k_BT/m\,I)$
exactly, that beads moving between distances stay at the target temperature, the knot curves, the
two-wall superposition and the layer-conditioned VACF.

## License

© 2026 Jinu Jeong. This project is licensed under the MIT License. See the LICENSE file for details.
