"""All-atom MD of flexible SPC/E water between graphene sheets (LAMMPS through its Python module).

Force field and settings copied from the input of the original 80 A system (pair lj/cut/coul/long 11 A,
SPC/E charges and O LJ, graphene C LJ 0.069863 kcal/mol / 3.566 A, arithmetic mixing, tail correction,
PPPM 1e-4 with slab 3, boundary p p f, harmonic O-H (1000, 1.0 A) and H-O-H (100, 109.47 deg), special_bonds
0 0 0.5, graphene-graphene neighbours excluded, graphene not integrated, Nose-Hoover NVT 300 K / 100 fs on
water, 1 fs). Steps: minimisation with the graphene held, equilibration, production. During production the
centre-of-mass position (unwrapped) and velocity of every water molecule are stored every step (float32,
A and A/fs) from compute com/chunk and vcm/chunk.

    python run_confined.py --data H20/start.data --out H20 \     # python with the LAMMPS module (GPU package)
        --equil 200000 --prod 100000
outputs in --out: com.npy, vcm.npy (prod+1, N, 3), thermo.npz, meta.json, final.data, log.lammps
"""
import argparse
import json
import os
import time

import numpy as np
import lammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY

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
set type 1 charge 0.4236
set type 2 charge -0.8472
group graphene type 3
group water type 1 2
pair_style      lj/cut/coul/long 11.0 11.0
pair_coeff      1 1 0.00 0.00
pair_coeff      2 2 0.15535 3.166
pair_coeff      3 3 0.069863 3.5660
pair_modify     mix arithmetic
kspace_style    pppm 1.0e-4
kspace_modify   slab 3
dielectric      1.0
pair_modify     tail yes
bond_style      harmonic
angle_style     harmonic
bond_coeff      1 1000.00 1.000
angle_coeff     1 100.0 109.47
special_bonds   lj/coul 0.0 0.0 0.5
neighbor        2.0 bin
timestep        1
neigh_modify    delay 0 every 1 check yes
neigh_modify    exclude group graphene graphene
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
