# Differentiable GLE with memory fields

**Learning state-dependent memory kernels for coarse-grained molecular dynamics**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Coarse-grained (CG) models remove degrees of freedom, and the removed degrees of freedom leave behind
friction and noise with memory. A generalized Langevin equation (GLE) carries them through a memory
kernel. Usually every bead gets the same kernel, as in a bulk fluid, but near interfaces and in
confinement the memory of a molecule depends on where it is.

This repository learns a **memory field**: a memory kernel that depends on a state variable of the
bead, $\kappa(t,\zeta)$, here the distance $d$ to a wall. The field is learned **top-down**: the CG
simulation is differentiated so that its dynamics match all-atom references.

- **Any state dependence, exact thermodynamics.** The kernel is a sum of damped-oscillator modes whose
  parameters are functions of the state. Every mode satisfies the fluctuation–dissipation theorem (FDT)
  for any parameter values, so the FDT and the temperature stay exact for any state dependence, also
  when beads move between states.
- **Exact gradients at low memory cost.** The memory is carried by a few auxiliary variables instead of
  a velocity history, so standard autograd differentiates through the whole simulation window with a
  memory cost set by the number of modes.
- **Bulk fluids are the uniform case.** For bulk water and CO₂ the field is a single kernel with
  4 modes (16 parameters).
- **Water between graphene sheets.** A memory field learned in an 80 Å channel reproduces the
  layer-resolved velocity autocorrelation functions (VACFs), transfers without retraining to channels
  20–40 Å wide, and predicts how long water stays next to the wall, which it was not trained on.

> The earlier version, **DiffGLE** (learnable colored-noise filter trained with an adjoint method;
> [Europhys. Lett.](https://doi.org/10.1209/0295-5075/ae9c6e),
> [arXiv:2410.08424](https://arxiv.org/abs/2410.08424)), with its paper data and figure scripts, is on
> the [`v2` branch](https://github.com/jinu-jeong/Differentiable_GLE/tree/v2).

## Core idea

Each CG bead of mass $m$ obeys

```math
\dot{\mathbf q}=\mathbf v,\qquad
m\dot{\mathbf v}=\mathbf F(\mathbf q)-m\int_0^t\kappa\bigl(t-s,\zeta\bigr)\,\mathbf v(s)\,ds+\mathbf R(t),
```

with a fixed conservative CG force $\mathbf F$. The memory field is a Prony approximation, a sum of $M$
damped-oscillator modes whose four parameters depend on the state $\zeta$ of the bead:

```math
\kappa(t,\zeta)=\sum_{k=1}^{M}a_k(\zeta)\,\bigl[e^{-A_k(\zeta)\,t}\bigr]_{11},\qquad
A_k=\begin{pmatrix}\gamma_{1,k}&\omega_k\\-\omega_k&\gamma_{2,k}\end{pmatrix},\qquad
a_k,\gamma_{1,k},\gamma_{2,k}>0 .
```

Each mode is realized by two auxiliary variables per degree of freedom (Markovian embedding). The
symmetric part of $A_k$ is positive definite for any parameter values, so every slice
$\kappa(\cdot,\zeta)$ has a non-negative spectrum and the noise matches the friction. For a wall, each
parameter is a piecewise-linear curve in $d$ through 20 knots (2.5–20 Å), separately parallel and
perpendicular to the wall. The velocity and auxiliary variables are propagated exactly over each time
step (GLE-BAOAB); every one-step map preserves $\mathcal N(0,k_BT/m\,I)$, so the integrator keeps the
temperature exact whatever the state dependence.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build for your CUDA version first if you need a specific one. The all-atom
simulations of confined water use LAMMPS through its Python module (GPU package). Inputs are in
`data/` (see [Force fields and data](#force-fields-and-data)) and every run writes to its own folder in
`results/`.

## Bulk water and CO₂

The processed inputs `data/processed/H2O` and `data/processed/CO2` (initial states, reference
observables, CG pair potentials) are included, so these runs need no download:

```bash
python main.py -s H2O                          # DiffGLE filter, used as the initial kernel of the modes
python -m experiments.train_osc -s H2O         # 4 modes, GLE-BAOAB, exact gradients -> results/H2O_osc_baoab
```

Without `results/H2O` the modes start from a generic kernel. `experiments/bulk/` holds the reproduction
runs for the bulk comparisons (training repeats, filter trained with exact gradients, mode-count scan,
measured backpropagation memory, production evaluation); results go to `results/bulk/`.

## Water between graphene sheets

**1. All-atom references.** Flexible SPC/E water between frozen graphene sheets $H$ apart, with the force
field in [forcefields/](forcefields/) (same for every $H$; $H$ = 20, 30, 40, 80 Å with 447, 726, 1006, 2123
molecules). The system is called `graphene_water_H<H>`:

```bash
A=experiments/confined/allatom
R=data/raw/graphene_water_H80/aa
mkdir -p $R
python $A/build_confined.py 80 2123 $R/start.data
python $A/run_confined.py --data $R/start.data --out $R --equil 200000 --prod 100000
python $A/make_dataset.py 80                          # -> data/processed/graphene_water_H80: CG system, targets/
python $A/make_dataset.py 80 --slab-frame             # -> targets_slab/ (training targets)
```

On graphene the water slab slides as a whole, and its centre-of-mass velocity is not equilibrated in
100 ps. The training targets are therefore computed in the frame of the slab (`--slab-frame`).

**2. Train a memory field** on the 80 Å channel, starting from the bulk-water kernel:

```bash
python experiments/confined/train.py --system graphene_water_H80 --wall-gap 80 \
    --model knot --groups 2 --target fine --smooth 3 -n 300 --avg-last 50 --prod 50000 \
    --out results/memory_field_H80
```

`--smooth` weights a penalty on the curvature of the curves in $d$. Comparison models:
`--model bins --bins 8 --groups 2` (one kernel per water layer) and `--bins 1 --groups 1` (one kernel
for all beads).

**3. Transfer to a narrower channel** without retraining (`-n 0`: production only):

```bash
python experiments/confined/train.py --system graphene_water_H20 --wall-gap 20 \
    --model knot --groups 2 --target fine -n 0 --prod 50000 \
    --load-state results/memory_field_H80/thermostat_state_dict.pth --out results/memory_field_H80_in_H20
```

Add `--superpose` to add the effects of both walls.

**4. Residence in the contact layer**, an observable the field is not trained on:

```bash
python experiments/confined/residence.py aa 20                                  # all-atom
python experiments/confined/residence.py cg 20 results/memory_field_H80_in_H20  # CG
```

## Force fields and data

All potentials are in [forcefields/](forcefields/): the all-atom force field of water between graphene sheets
(LAMMPS include and graphene coordinates) and the CG pair potentials of every system, kept fixed while the
memory is learned. The all-atom trajectories themselves are not in the repository: run the MD (step 1
above) or, for bulk water and CO₂, download them.

```
data/
├── raw/          all-atom trajectories (simulated or downloaded); read only
│   ├── H2O/aa/, CO2/aa/
│   └── graphene_water_H{20,30,40,80}/aa/    output of experiments/confined/allatom/run_confined.py
└── processed/    made from raw/ and forcefields/ by the scripts; read by diffgle.systems.load_system
    ├── H2O/, CO2/                           included in the repository
    └── graphene_water_H<H>/                 CG system, targets/, targets_slab/
```

Nothing writes to `data/raw/`, and everything in `data/processed/` can be rebuilt: `preprocess.py` for
`H2O` and `CO2`, `experiments/confined/allatom/make_dataset.py` for `graphene_water_H<H>`. Bulk
trajectories: [uofi.box.com](https://uofi.box.com/s/gruyslzav75ibbg0qjlh877f37le78c4) (unpack `H2O/AA` as
`data/raw/H2O/aa/`, same for `CO2`). Details in [data/README.md](data/README.md) and
[forcefields/README.md](forcefields/README.md).

## Repository structure

| path | contents |
|---|---|
| `diffgle/memory_field.py` | memory field and its two-wall superposition, binned comparison kernels, tabulated exact O step, BAOAB with frozen walls, layer-resolved VACF |
| `diffgle/thermostats/` | damped-oscillator modes (`damped_osc.py`), DiffGLE filter and its exact adjoint (`conv.py`), Langevin |
| `diffgle/` | units, systems table, tabulated CG forces, integrators, observables |
| `experiments/confined/` | memory-field training, all-atom targets, contact-layer residence |
| `experiments/train_osc.py`, `experiments/bulk/` | bulk training with modes and exact gradients; reproduction runs |
| `experiments/` (other) | gradient, loss and integrator checks for the bulk models |
| `experiments/confined/allatom/` | all-atom runs of water between graphene sheets: build, run, make the CG system and targets |
| `forcefields/` | all-atom force field of water between graphene; CG pair potentials of every system |
| `data/`, `results/` | inputs (raw, processed) and run outputs; see [Force fields and data](#force-fields-and-data) |
| `tests/` | memory field, damped-oscillator kernel, BAOAB, adjoint |
| `main.py`, `preprocess.py` | DiffGLE pipeline and preprocessing |

## Tests

```bash
python -m pytest tests
```

`tests/test_memory_field.py` checks that every tabulated one-step map preserves
$\mathcal N(0,k_BT/m\,I)$ exactly, that beads moving between distances stay at the target temperature,
the knot curves, the two-wall superposition and the layer-resolved VACF.

## Reference

For the earlier DiffGLE method, please cite:

J. Jeong, I. Nadkarni, "DiffGLE: Differentiable Coarse-Grained Dynamics using Generalized Langevin
Equation," *Europhysics Letters*. https://doi.org/10.1209/0295-5075/ae9c6e

## License

© 2026 Jinu Jeong. Released under the MIT License; see [LICENSE](LICENSE).
