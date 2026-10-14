# Data

Two kinds of data, kept apart:

- `raw/` — all-atom trajectories. **No script writes here** except the all-atom MD itself, which writes its
  own `aa/` folder. Not part of the repository: download them (bulk) or run the MD (water–graphene).
- `processed/` — everything the CG simulations read, made from `raw/` and `../forcefields/` by the scripts
  below. It can be deleted and rebuilt. `diffgle.systems.load_system(name)` reads `processed/<name>/`.

Force fields and CG potentials are in [../forcefields/](../forcefields/). Run outputs (trained models,
analyses) go to `../results/<run>/`, never into `data/`.

## Systems

| system | what | raw | processed by |
|---|---|---|---|
| `H2O` | bulk SPC/E water, 2141 molecules | `raw/H2O/aa/` (download) | `preprocess.py --system H2O` (included in the repository) |
| `CO2` | bulk CO₂, 1700 molecules | `raw/CO2/aa/` (download) | `preprocess.py --system CO2` (included in the repository) |
| `graphene_water_H<H>` | flexible SPC/E water between frozen graphene sheets H = 20, 30, 40, 80 Å apart; molecular centres of mass every 1 fs, 100 ps | `raw/graphene_water_H<H>/aa/` (run `experiments/confined/allatom/`) | `experiments/confined/allatom/make_dataset.py <H>` and `--slab-frame` |

Names used before 2026-10-13 (`Confined_H2O_v2_H<H>`) still load through `diffgle.systems.ALIASES`.

## Layout

```
raw/
  H2O/aa/pos_COM.npy, vel_COM.npy
  CO2/aa/pos_COM.npy, vel_COM.npy
  graphene_water_H<H>/aa/      start.data, com.npy, vcm.npy (frames, molecules, 3; A, A/fs),
                               thermo.npz, meta.json, final.data, log.lammps
processed/<system>/
  pos0.npy, vel0.npy, box.npy, vacf_aa.npy, msd_aa.npy, CG/*.pot
  bulk:            r_aa.npy, rdf_aa.npy
  water-graphene:  z.npy, z_aa.npy, density_aa.npy,
                   targets/       aa_density.npz, aa_layer_vacf{,_fine,_noise}.npz   (lab frame)
                   targets_slab/  the same in the frame of the water slab (training targets)
```

## Getting the raw data

- `H2O`, `CO2`: [uofi.box.com](https://uofi.box.com/s/gruyslzav75ibbg0qjlh877f37le78c4); unpack `H2O/AA`
  as `raw/H2O/aa` (same for `CO2`). Only needed to rerun `preprocess.py`; the processed inputs are included.
- `graphene_water_H<H>`: run the all-atom MD with the force field in `../forcefields/allatom/graphene_water/`
  (commands in [../forcefields/README.md](../forcefields/README.md)), then

```bash
for H in 20 30 40 80; do
  python experiments/confined/allatom/make_dataset.py $H
  python experiments/confined/allatom/make_dataset.py $H --slab-frame
done
```
