"""All-atom MD of flexible SPC/E water between graphene sheets (LAMMPS through its Python module).

Force field: forcefields/allatom/graphene_water/forcefield.lmp (flexible SPC/E water, graphene LJ, PPPM with
the slab correction, boundary p p f). Graphene is not integrated; Nose-Hoover NVT 300 K / 100 fs on water, 1 fs.
Steps: minimisation with the graphene held, equilibration, production. During production the
centre-of-mass position (unwrapped) and velocity of every water molecule are stored every step (float32,
A and A/fs) from compute com/chunk and vcm/chunk.

    # python with the LAMMPS module (GPU package)
    python experiments/confined/allatom/run_confined.py --data data/raw/graphene_water_H20/aa/start.data \
        --out data/raw/graphene_water_H20/aa --equil 200000 --prod 100000
outputs in --out: com.npy, vcm.npy (prod+1, N, 3), thermo.npz, meta.json, final.data, log.lammps
"""
import argparse
import json
import os
import time

import numpy as np
import lammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FORCEFIELD = os.path.join(ROOT, "forcefields", "allatom", "graphene_water", "forcefield.lmp")

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--equil", type=int, default=200000)
p.add_argument("--prod", type=int, default=100000)
p.add_argument("--seed", type=int, default=123456789)
p.add_argument("--gpu", default="-sf gpu -pk gpu 1 neigh no")   # neighbour lists on the CPU: allows neigh_modify exclude
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

lmp = lammps.lammps(cmdargs=a.gpu.split() + ["-screen", "none", "-log", os.path.join(a.out, "log.lammps")])
lmp.commands_string(f"""
units           real
atom_style      full
boundary        p p f
read_data       {os.path.abspath(a.data)}
include         {FORCEFIELD}
timestep        1
thermo_style    custom step temp pe etotal press
thermo          1000
""")
t0 = time.time()
# minimisation with the graphene held in place
lmp.commands_string("""
fix hold graphene setforce 0.0 0.0 0.0
minimize 1.0e-4 1.0e-6 2000 20000
unfix hold
reset_timestep 0
""")
lmp.command(f"velocity water create 300.0 {a.seed} rot yes mom yes dist gaussian")
lmp.command("fix 1 water nvt temp 300.00 300.00 100")
lmp.command("compute twat water temp")
lmp.command("thermo_modify temp twat")
lmp.command(f"run {a.equil}")
t_eq = time.time() - t0

lmp.commands_string("""
compute mol water chunk/atom molecule
compute com water com/chunk mol
compute vcm water vcm/chunk mol
""")
nmol = int(lmp.extract_compute("mol", 0, 0) or 0) or int((lmp.get_natoms() - 672) // 3)
com = np.empty((a.prod + 1, nmol, 3), np.float32)
vcm = np.empty((a.prod + 1, nmol, 3), np.float32)
thermo = {"step": [], "T_water": [], "pe": []}
step0 = lmp.extract_global("ntimestep")


def record(_=None):
    k = lmp.extract_global("ntimestep") - step0
    com[k] = lmp.numpy.extract_compute("com", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY)
    vcm[k] = lmp.numpy.extract_compute("vcm", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY)
    if k % 100 == 0:
        thermo["step"].append(k)
        thermo["T_water"].append(lmp.get_thermo("temp"))
        thermo["pe"].append(lmp.get_thermo("pe"))


import __main__  # noqa: E402
__main__.cg_record = record
lmp.command("fix rec all python/invoke 1 end_of_step cg_record")
lmp.command("run 0")
record()                                         # step 0 (end_of_step does not fire in run 0)
lmp.command(f"run {a.prod}")
lmp.command(f"write_data {os.path.join(os.path.abspath(a.out), 'final.data')}")

np.save(os.path.join(a.out, "com.npy"), com)
np.save(os.path.join(a.out, "vcm.npy"), vcm)
np.savez(os.path.join(a.out, "thermo.npz"), **{k: np.array(v) for k, v in thermo.items()})
box = lmp.extract_box()
json.dump(dict(vars(a), nmol=nmol, box_lo=list(box[0]), box_hi=list(box[1]), dt_fs=1.0,
               units="real: A, fs, kcal/mol; com in A (unwrapped), vcm in A/fs",
               seconds_equil=t_eq, seconds_total=time.time() - t0, lammps_version=lmp.version()),
          open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"done {a.out}: N={nmol}, equil {t_eq:.0f} s, total {time.time() - t0:.0f} s, "
      f"<T_water> {np.mean(thermo['T_water']):.1f} K", flush=True)
lmp.close()
