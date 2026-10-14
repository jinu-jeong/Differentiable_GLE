"""LAMMPS data file: flexible SPC/E water between two graphene sheets a distance H apart.

Force field forcefields/allatom/graphene_water/forcefield.lmp. Graphene: forcefields/allatom/graphene_water/
graphene_sheets.xyz (lower sheet at z = 0, upper sheet moved from z = 80 to z = H); box x [-25.174493, 4.29808],
y [-24.815001, 4.963], z [-10, H + 30] (non-periodic in z). Atom types 1 H, 2 O, 3 C; bonds/angles of type 1
(O-H 1.0 A, H-O-H 109.47 deg, the equilibrium values of the input). Graphene atoms have molecule ID 0, so that
compute chunk/atom molecule gives one chunk per water molecule.
Water: N molecules on a lattice filling 3.0 < z < H - 3.0 with random orientations; each orientation is redrawn
until no H/O pair with any placed molecule is closer than 1.7 A, and the script stops with an error if 200 draws
fail. (Before 2026-10-13 only the last 400 molecules were checked and the last draw was kept after 200 failures:
the H30-H80 start files had pairs down to 1.0 A, removed by the minimisation and 200 ps equilibration of
run_confined.py.) Water numbers used: H = 20, 30, 40, 80 A with N = 447, 726, 1006, 2123.

    python experiments/confined/allatom/build_confined.py H N data/raw/graphene_water_H<H>/aa/start.data
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GRAPHENE = os.path.join(ROOT, "forcefields", "allatom", "graphene_water", "graphene_sheets.xyz")

H, N, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
XLO, XHI, YLO, YHI = -25.174493, 4.29808, -24.815001, 4.963
LX, LY = XHI - XLO, YHI - YLO
ZLO, ZHI = -10.0, H + 30.0
mO, mH, mC = 15.9994, 1.00794, 12.0107
qO, qH = -0.8472, 0.4236
rOH, theta = 1.0, np.deg2rad(109.47)
rng = np.random.default_rng(int(H * 1000) + N)

# graphene sheets at z = 0 and 80 A; the upper one is moved to z = H
C = np.loadtxt(GRAPHENE, skiprows=2, usecols=(1, 2, 3))
assert np.allclose(np.unique(np.round(C[:, 2], 3)), [0.0, 80.0])
C[C[:, 2] > 40, 2] = H

# water lattice
zmin, zmax = 3.0, H - 3.0
vol = LX * LY * (zmax - zmin)
a = (vol / N) ** (1 / 3)
nx, ny, nz = [max(1, int(np.floor(L / a))) for L in (LX, LY, zmax - zmin)]
while nx * ny * nz < N:                       # grow the smallest-spacing direction until the lattice holds N
    s = np.array([LX / nx, LY / ny, (zmax - zmin) / nz])
    k = int(np.argmax(s))
    if k == 0: nx += 1
    elif k == 1: ny += 1
    else: nz += 1
g = np.stack(np.meshgrid((np.arange(nx) + 0.5) / nx * LX + XLO, (np.arange(ny) + 0.5) / ny * LY + YLO,
                         zmin + (np.arange(nz) + 0.5) / nz * (zmax - zmin), indexing="ij"), -1).reshape(-1, 3)
sites = g[rng.choice(len(g), N, replace=False)]
h1 = rOH * np.array([np.sin(theta / 2), 0, np.cos(theta / 2)])
h2 = rOH * np.array([-np.sin(theta / 2), 0, np.cos(theta / 2)])
shift = mH * (h1 + h2) / (mO + 2 * mH)


def rot():
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z_ = q
    return np.array([[1 - 2 * (y * y + z_ * z_), 2 * (x * y - z_ * w), 2 * (x * z_ + y * w)],
                     [2 * (x * y + z_ * w), 1 - 2 * (x * x + z_ * z_), 2 * (y * z_ - x * w)],
                     [2 * (x * z_ - y * w), 2 * (y * z_ + x * w), 1 - 2 * (x * x + y * y)]])


box = np.array([LX, LY, 1e9])
mols = []
for c in sites:
    for _ in range(200):
        R = rot()
        o = c - R @ shift
        m = np.array([o, o + R @ h1, o + R @ h2])
        if not mols:
            break
        prev = np.concatenate(mols)            # all placed molecules
        dd = m[:, None] - prev[None]
        dd -= np.round(dd / box) * box
        if np.sqrt((dd ** 2).sum(-1)).min() > 1.7:
            break
    else:                                      # no orientation found: refuse instead of writing an overlap
        raise RuntimeError(f"cannot place a molecule at {c} without H/O pairs closer than 1.7 A "
                           f"(lattice too dense for H = {H} A, N = {N})")
    mols.append(m)
W = np.array(mols)                              # (N, 3, 3): O, H, H

lines = [f"Flexible SPC/E water ({N}) between graphene sheets at z = 0 and {H:g} A\n",
         f"{3 * N + len(C)} atoms", "3 atom types", f"{2 * N} bonds", "1 bond types", f"{N} angles",
         "1 angle types", "", f"{XLO} {XHI} xlo xhi", f"{YLO} {YHI} ylo yhi", f"{ZLO} {ZHI} zlo zhi", "",
         "Masses", "", f"1 {mH}", f"2 {mO}", f"3 {mC}", "", "Atoms # full", ""]
aid = 0
for i, m in enumerate(W):
    for j, (typ, q) in enumerate(((2, qO), (1, qH), (1, qH))):
        aid += 1
        lines.append(f"{aid} {i + 1} {typ} {q} {m[j, 0]:.6f} {m[j, 1]:.6f} {m[j, 2]:.6f}")
for c in C:
    aid += 1
    lines.append(f"{aid} 0 3 0.0 {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
lines += ["", "Bonds", ""]
for i in range(N):
    o = 3 * i + 1
    lines += [f"{2 * i + 1} 1 {o} {o + 1}", f"{2 * i + 2} 1 {o} {o + 2}"]
lines += ["", "Angles", ""]
for i in range(N):
    o = 3 * i + 1
    lines.append(f"{i + 1} 1 {o + 1} {o} {o + 2}")
open(out, "w").write("\n".join(lines) + "\n")
print(f"H={H} N={N} lattice {nx}x{ny}x{nz} spacing {LX / nx:.2f} {LY / ny:.2f} {(zmax - zmin) / nz:.2f} A -> {out}")
