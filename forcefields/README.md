# Force fields

Everything needed to regenerate the data from scratch: the all-atom force field to run the reference MD,
and the coarse-grained (CG) pair potentials, which are kept fixed while the memory is learned.
Units: LAMMPS `real` (Å, fs, kcal/mol, amu, e).

## All-atom: water between graphene sheets

`allatom/graphene_water/`

| file | contents |
|---|---|
| `forcefield.lmp` | LAMMPS include (after `read_data`): flexible SPC/E water and graphene, long-range electrostatics with the slab correction |
| `graphene_sheets.xyz` | the 2 × 336 carbon atoms of two graphene sheets at z = 0 and z = 80 Å; the builder moves the upper sheet to z = H |

| | parameters |
|---|---|
| water | SPC/E charges, H +0.4236 e, O −0.8472 e; O Lennard-Jones ε = 0.15535 kcal/mol, σ = 3.166 Å, no LJ on H |
| | harmonic O–H bond, 1000 kcal mol⁻¹ Å⁻², 1.0 Å; harmonic H–O–H angle, 100 kcal mol⁻¹ rad⁻², 109.47° |
| carbon | Lennard-Jones ε = 0.069863 kcal/mol, σ = 3.566 Å; frozen (not integrated) |
| water–carbon | arithmetic mixing (C–O: ε = 0.104 kcal/mol, σ = 3.366 Å) |
| cut-offs | LJ and real-space Coulomb 11 Å, tail correction; PPPM, accuracy 10⁻⁴, `kspace_modify slab 3` |
| box | x [−25.174493, 4.29808] Å, y [−24.815001, 4.963] Å (periodic); z [−10, H + 30] Å (fixed) |
| run | minimisation with the graphene held; Nosé–Hoover NVT on water, 300 K, 100 fs; time step 1 fs |

Water numbers: H = 20, 30, 40, 80 Å with 447, 726, 1006, 2123 molecules (the same density at the channel
centre). To make a reference trajectory:

```bash
A=experiments/confined/allatom
R=data/raw/graphene_water_H80/aa
mkdir -p $R
python $A/build_confined.py 80 2123 $R/start.data
python $A/run_confined.py --data $R/start.data --out $R --equil 200000 --prod 100000   # LAMMPS Python module
python $A/make_dataset.py 80 && python $A/make_dataset.py 80 --slab-frame
```

## Coarse-grained pair potentials

`cg/` — tabulated potentials in VOTCA format (index, r in Å, U in kcal/mol, F in kcal mol⁻¹ Å⁻¹), one
bead per molecule at its centre of mass. They were obtained by relative-entropy minimization with VOTCA
and are kept fixed.

| file | system | pair | bead mass |
|---|---|---|---|
| `cg/H2O/CG_CG.pot` | bulk water (`H2O`) | water–water | 18.0 amu |
| `cg/CO2/CG_CG.pot` | bulk CO₂ (`CO2`) | CO₂–CO₂ | 44.1 amu |
| `cg/graphene_water/CG1_CG1.pot` | water between graphene (`graphene_water_H<H>`) | water–water | 18.01528 amu |
| `cg/graphene_water/CG1_GR1.pot` | water between graphene | water–carbon (frozen wall beads) | — |

In the simulations, forces are evaluated for pairs within 6 Å. `preprocess.py` (bulk) and
`experiments/confined/allatom/make_dataset.py` (water–graphene) copy these files into
`data/processed/<system>/CG/`.

The all-atom references of the bulk systems are those of DiffGLE (SPC/E water, CO₂); their trajectories
are available as described in [../data/README.md](../data/README.md).
