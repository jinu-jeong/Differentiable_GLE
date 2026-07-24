# Paper Data Manifest

This directory contains compact data needed to regenerate the figures in the revised DiffGLE manuscript. It intentionally stores processed arrays and source curves, not large raw trajectories.

## `optimization_histories/`

- `co2_symdetach_lr3e5_iter372/`:
  selected CO2 DiffGLE checkpoint and saved optimization history used in the main CO2 result.
- `h2o_original_fromscratch/`:
  original H2O adjoint optimization history used for filter/kernel/loss evolution panels.
- `star_polymer_noforce_fromscratch_support700/`:
  from-scratch star-polymer no-force optimization snapshots.
- `star_polymer_noforce_continuation_support700/`:
  final star-polymer no-force continuation result used for the manuscript figure.

The top-level `optimization_history_manifest.md` records the metrics associated with these histories.

## `source_data/`

Production observables used for figure overlays:

- `co2_symdetach_iter372_observables.npz`
- `h2o_ibi20_original_final_iter299_observables.npz`

## `gradient_mechanics/`

Saved CO2 direct-vs-adjoint gradient audit arrays and memory-use summary used by `scripts/plot_gradient_mechanics.py`.

## `supplementary/`

Source curves for supplementary FDT-closure and fixed-pair-potential figures.
