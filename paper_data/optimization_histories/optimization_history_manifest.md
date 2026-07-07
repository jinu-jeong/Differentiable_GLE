# Optimization History Manifest

This directory stores the compact optimization histories and source arrays used to regenerate the revised DiffGLE manuscript figures. It intentionally contains processed arrays and small metadata files only; exploratory run folders, raw trajectories, scheduler files, and generated per-iteration image dumps are not included.

## Included Systems

| system | folder | role in manuscript | main metric |
|---|---|---|---|
| CO2 | `co2_symdetach_lr3e5_iter372/` | selected bulk-fluid DiffGLE result | production VACF RMSE `0.0069809016`; RDF RMSE `0.0101636656` |
| H2O | `h2o_original_fromscratch/` | original bulk-water optimization history used for filter, kernel, and loss evolution | production VACF RMSE `0.0218090694`; RDF RMSE `0.0362875998` |
| star polymer no-force | `star_polymer_noforce_fromscratch_support700/` | from-scratch no-force memory benchmark history | DiffGLE VACF RMSE `0.0286259692`; Markov RMSE `0.0760488839` |
| star polymer no-force | `star_polymer_noforce_continuation_support700/` | final no-force memory benchmark used in the manuscript figure | DiffGLE VACF RMSE `0.0197935887`; Markov RMSE `0.0760488839` |

## CO2 Selected Result

Folder: `co2_symdetach_lr3e5_iter372/`

Purpose: reproduce the CO2 panels in the main manuscript figure.

Settings summary:

- conservative force: fixed tabulated CO2 CG-CG pair potential,
- filter parameterization: free colored-noise filter,
- optimization route: adjoint-assisted trajectory differentiation,
- optimizer: SGD,
- original simple-start stage: learning rate `1e-3`, VACF loss window `0:500`,
- selected continuation endpoint: iteration `372`,
- selected checkpoint evaluation: full 1000-sample VACF window.

Files:

- `history_arrays.npz`: saved original optimization trajectory plus selected endpoint arrays,
- `iteration_372.npz`: selected filter/kernel/VACF arrays,
- `settings_summary.json`: compact settings and metric summary.

## H2O Original From-Scratch Result

Folder: `h2o_original_fromscratch/`

Purpose: reproduce the H2O optimization panels in the main manuscript figure. The production observable overlay used by the figure script is stored in `paper_data/source_data/h2o_ibi20_original_final_iter299_observables.npz`.

Settings summary:

- conservative force: fixed tabulated H2O CG-CG pair potential,
- filter parameterization: free colored-noise filter,
- optimization route: adjoint-assisted trajectory differentiation,
- optimizer: Adam,
- learning-rate schedule: `3e-4` from iteration `0`, `1e-4` from iteration `100`, `1e-5` from iteration `200`,
- total iterations: `300`.

Files:

- `history_arrays.npz`: filter, memory-kernel, VACF, and loss evolution arrays,
- `optimization_metrics.csv`: iteration-wise scalar metrics,
- `settings_summary.json`: compact settings and metric summary.

## Star-Polymer No-Force Benchmark

Folders:

- `star_polymer_noforce_fromscratch_support700/`,
- `star_polymer_noforce_continuation_support700/`.

Purpose: reproduce the star-polymer no-force memory-benchmark panels in the main manuscript figure. This benchmark isolates the memory-learning problem by omitting the conservative force.

Settings summary:

- conservative force: none,
- target: normalized reference VACF,
- replicas: `64`,
- particles per replica: `7`,
- mass: `11.0`,
- internal time step: `0.005`,
- samples per trajectory: `1000`,
- burn-in steps: `1000`,
- filter parameterization: free filter,
- filter length: `1000`,
- taper/support: `650 -> 700`,
- optimizer: Adam,
- learning rate: `0.003`,
- gradient mode: fixed-noise direct optimization.

Files in each folder:

- `kernel_snapshots.npz`: saved filter and FDT memory-kernel snapshots,
- `optimization_metrics.csv`: iteration-wise scalar metrics,
- `star_polymer_diffgle_noforce_curves.npz`: target, DiffGLE, and Markov baseline VACF curves,
- `star_polymer_diffgle_noforce_summary.json`: compact settings and final metrics,
- `vacf_evolution_replay.npz`: replayed VACF traces for saved filter snapshots.

## Figure Scripts

The main paper figure can be regenerated with:

```bash
python scripts/make_paper_figures.py
```

Additional diagnostics can be regenerated with:

```bash
python scripts/plot_gradient_mechanics.py
python scripts/plot_fdt_closure.py
python scripts/plot_pair_potentials.py
```
