# Optimization-History Figure Manifest

Date prepared: 2026-06-27

This folder collects manuscript provenance for the best accuracy plots and the
from-simple-initialization optimization histories. It is separate from the
simulation code and contains only copied or compacted result artifacts.

## Folder Map

| Folder | Role | Continuation? | Main result |
|---|---|---:|---|
| `co2_fromscratch_sgd_windowdetached/` | CO2 paper-quality from-scratch optimization history | No | production VACF RMSE `0.0142938755`, RDF RMSE `0.0116967017` |
| `co2_symdetach_lr3e5_iter372/` | CO2 selected symmetric-detach checkpoint, staged for manuscript figure | Yes | production VACF RMSE `0.0069809016`, RDF RMSE `0.0101636656` |
| `h2o_original_fromscratch/` | Original H2O from-scratch history used for clean simple-initialization provenance | No | production VACF RMSE `0.0218090694`, RDF RMSE `0.0362875998` |
| `h2o_ibi20_fromscratch/` | H2O IBI20 conservative-potential from-scratch diagnostic | No | final training VACF RMSE `0.0138259111`; production transfer was not the best |
| `star_polymer_noforce_fromscratch_support700/` | Star-polymer no-force support-700 history from simple initial filter | No | DiffGLE VACF RMSE `0.0286259692`, Markov `0.0760488839` |
| `star_polymer_noforce_continuation_support700/` | Star-polymer no-force best continuation from the support-700 checkpoint | Yes | DiffGLE VACF RMSE `0.0197935887`, Markov `0.0760488839` |

## CO2 From Scratch

Source:

- Main run data: `code/Result/gpu_reproduction/20260613_gpu_full/CO2`
- Logbook plots/settings: `code/Result/gpu_reproduction/20260617_sgd_windowdetached_2orig_summary/co2_logbook_plots`
- Production validation: `code/Result/gpu_reproduction/20260614_original_kernel_production_sampling/CO2_free_original_50k_x5`

Settings:

- `FILTER_PARAM=free`
- `ITERATIONS=300`
- `OPTIMIZER=SGD`
- `LEARNING_RATE=1e-3`
- `ODE_BACKPROP_MODE=adjoint`
- `TRAINING_VACF_GRADIENT_MODE=window_detached`
- `TRAINING_VACF_ORIGINS=2`
- `VACF_LOSS_WINDOW=0:500`
- `GRAD_CLIP_NORM=100`
- `MAX_KERNEL_L2=0.5`
- Conservative potential: `Data/CO2/CG/CG_CG.pot`

Staged files:

- `co2_sgd_windowdetached_saved_vacf_filter_kernel_evolution.png`
- `co2_sgd_windowdetached_kernel_evolution_iter000_299.png`
- `co2_sgd_windowdetached_loss_evolution_iter000_299.png`
- `co2_sgd_windowdetached_optimization_settings.txt`
- `co2_fromscratch_filter_memory_vacf_loss_regenerated.png`
- `history_arrays.npz`
- `optimization_metrics.csv`
- `settings_summary.json`
- `co2_production_observables_50kx5.png`
- `co2_production_metrics_50kx5.csv`

## CO2 Symmetric-Detach Selected Checkpoint

Source:

- Original simple-start history: `code/Result/gpu_reproduction/20260613_gpu_full/CO2`
- Selected continuation checkpoint: `code/Result/gpu_reproduction/20260614_co2_symdetach_highlr_long/CO2_free20_symdetach_lr3e-5_clip500_iter300-599/checkpoints/checkpoint_iter_372.pth`
- Selected iteration arrays: `code/Result/gpu_reproduction/20260614_co2_symdetach_highlr_long/CO2_free20_symdetach_lr3e-5_clip500_iter300-599/iteration_data/iteration_372.npz`
- Production validation: `code/Result/gpu_reproduction/20260614_co2_symdetach_highlr_iter372_quickprod/CO2_symdetach_lr3e-5_iter372_10k_x1`

Settings:

- Original optimization: SGD, adjoint, window-detached, `LEARNING_RATE=1e-3`, `TRAINING_VACF_ORIGINS=2`, `VACF_LOSS_WINDOW=0:500`.
- Continuation: SGD, adjoint, `TRAINING_VACF_GRADIENT_MODE=symmetric_detach`, `LEARNING_RATE=3e-5`, `GRAD_CLIP_NORM=500`, `TRAINING_VACF_ORIGINS=20`.
- Selected checkpoint: iter `372`.
- Selected training full-VACF RMSE: `0.0083357533`.
- Production validation: warmup `3000`, production `10000`, replicas `1`, VACF origins total `901`.
- Production metrics: VACF RMSE `0.0069809016`, RDF RMSE `0.0101636656`.

Staged files:

- `iteration_372.npz`
- `CO2_symdetach_lr3e-5_iter372_training_vacf.png`
- `co2_symdetach_iter372_quickprod_metrics.csv`
- `history_arrays.npz`
- `settings_summary.json`

Note: only the selected continuation endpoint was preserved locally, so
`history_arrays.npz` contains the dense original `0-299` history plus the
selected `iter372` endpoint. It is intentionally not a dense `300-372`
continuation trajectory.

## H2O Original From Scratch

Source:

- Main run data: `code/Result/gpu_reproduction/20260613_gpu_full/H2O`
- Production validation: `code/Result/gpu_reproduction/20260614_original_kernel_production_sampling/H2O_free_original_50k_x5`

Settings:

- `FILTER_PARAM=free`
- `ITERATIONS=300`
- `OPTIMIZER=Adam`
- LR schedule: `3e-4` at iter `0`, `1e-4` at iter `100`, `1e-5` at iter `200`
- `TRAINING_VACF_GRADIENT_MODE=window_detached`
- Conservative potential: original packaged H2O CG potential

Staged files:

- `H2O_memory_kernel_evolution_lines.png`
- `h2o_original_fromscratch_filter_memory_vacf_loss.png`
- `history_arrays.npz`
- `optimization_metrics.csv`
- `settings_summary.json`
- `h2o_original_production_observables_50kx5.png`
- `h2o_original_production_metrics_50kx5.csv`

Publication-style mirror figure:

- `../assembled_figures/h2o/h2o_optimization_publication.png`
- `../assembled_figures/h2o/h2o_optimization_publication.pdf`
- Uses the original from-scratch adjoint history for the optimization panels and the IBI20-corrected H2O production observables for the VACF/RDF validation inset.

Note: this is the clean from-scratch history. The paper-facing best H2O
accuracy remains the later IBI20/smooth-target branch in
`../best_results_manifest.md`.

## H2O IBI20 From Scratch

Source:

- Main run data: `code/Result/gpu_reproduction/20260615_h2o_ibi20_original_objective/H2O_ibi20_original_2origin_free_defaultsched_fromscratch`

Settings:

- `FILTER_PARAM=free`
- `ITERATIONS=300`
- `OPTIMIZER=Adam`
- LR schedule: `3e-4` at iter `0`, `1e-4` at iter `100`, `1e-5` at iter `200`
- `TRAINING_VACF_GRADIENT_MODE=window_detached`
- `TRAINING_VACF_ORIGINS=2`
- Conservative potential: `IBI20` H2O `potential_iter_020.pot`

Staged files:

- `h2o_ibi20_fromscratch_filter_memory_vacf_loss.png`
- `h2o_ibi20_original_objective_vs_original.png`
- `h2o_ibi20_iter263_264_299_prod_compare.png`
- `history_arrays.npz`
- `optimization_metrics.csv`
- `settings_summary.json`

Note: this run reached a strong training RMSE but did not become the best
production-transfer H2O result.

## Star Polymer No-Force

From-scratch source:

- Cluster source: `polymer_extension/runs/star_diffgle_noforce_free1000_taper650_700_adam_fullmem_64r7p_lr003_init008_20260623_1`
- Staged folder: `star_polymer_noforce_fromscratch_support700/`

From-scratch settings:

- Target: Li 2017 TorchMD reference VACF
- `replicas=64`
- `particles_per_replica=7`
- `mass=11.0`
- `dt=0.005`
- `samples=1000`
- `burn_in_steps=1000`
- `FILTER_PARAM=free`
- `FILTER_LENGTH=1000`
- Taper/support: `650 -> 700`
- `OPTIMIZER=Adam`
- `LEARNING_RATE=0.003`
- `FILTER_GRADIENT_MODE=full_memory`
- `VACF_GRADIENT_MODE=full`
- `ITERATIONS=500`
- Best iteration: `499`
- DiffGLE VACF RMSE: `0.0286259692`
- Markov gamma VACF RMSE: `0.0760488839`

Continuation source:

- Cluster source: `polymer_extension/runs/star_diffgle_noforce_free1000_taper650_700_adam_fullmem_64r7p_lr003_continue1000_from185999_20260624_1`
- Staged folder: `star_polymer_noforce_continuation_support700/`

Continuation settings/result:

- Initialized from the from-scratch support-700 checkpoint.
- Same `64 x 7`, `dt=0.005`, `FILTER_LENGTH=1000`, taper `650 -> 700`, Adam `lr=0.003`.
- `ITERATIONS=1000`
- Best iteration: `991`
- DiffGLE VACF RMSE: `0.0197935887`
- Markov gamma VACF RMSE: `0.0760488839`

Staged files in each star-polymer folder:

- `kernel_snapshots.npz`
- `optimization_metrics.csv`
- `star_polymer_diffgle_noforce_curves.npz`
- `star_polymer_diffgle_noforce_summary.json`
- `star_polymer_diffgle_noforce_training.png`
- `train_stdout.json`

Additional from-scratch panels:

- `support700_filter_memory_vacf_evolution.png`
- `support700_live_vacf_rmse_progression.png`
- `vacf_evolution_replay.npz`
  - Generated locally with `../scripts/replay_star_polymer_noforce_vacf_evolution.py`.
  - Replays the 22 saved from-scratch support-700 filter snapshots with the recorded `64 x 7` no-force setup.

Additional continuation panel:

- `continue1000_filter_memory_vacf_loss_evolution.png`
- `vacf_evolution_replay.npz`
  - Generated locally with `../scripts/replay_star_polymer_noforce_vacf_evolution.py`.
  - Replays the 22 saved support-700 continuation filter snapshots with the recorded `64 x 7` no-force setup.
  - This file supplies the faint VACF-evolution traces in the assembled manuscript figure; the original staged continuation archive only stored final VACF curves plus scalar RMSE/loss history.

Manuscript-facing assembled figure:

- `../assembled_figures/star_polymer/star_polymer_noforce_publication.png`
- `../assembled_figures/star_polymer/star_polymer_noforce_publication.pdf`
- Clean four-panel figure matching the H2O/CO2 style: VACF comparison with replayed faint evolution traces, filter evolution, memory-kernel evolution, and training-loss curve. It uses the full selected support-700 chain: from-scratch iterations `0-499`, then continuation iterations offset to global iterations `500-1499`. RDF inset omitted because this no-force polymer benchmark has no structural RDF target.

## Compact History Array Schema

For fluid folders, `history_arrays.npz` contains:

- `iterations`
- `vacf_gle`
- `vacf_aa`
- `filters`
- `memories`
- `metrics_iteration`
- `metrics_loss`
- `metrics_vacf_rmse`
- `metrics_learning_rate`
- `metrics_memory_l2`
- `metrics_filter_l2`

These files are intended for consistent manuscript replotting without scanning
hundreds of per-iteration `.npz` files.
