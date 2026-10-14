"""H = 10 start file. Same as aa/confined_graphene/build_confined.py (same graphene, box, water geometry, charges, masses,
topology, 3.0 < z < H - 3.0, 1.7 A H/O criterion) except for the placement, because the shared builder fails at
H = 10: its lattice there is 10x10x2 with only 2.0 A between the two layers, and 96 intermolecular H/O pairs stay
closer than 1.7 A (min 0.50 A; its rejection loop gives up after 200 orientations and keeps the last one).
Here: two layers (z = 4 and 6 A, as in the shared lattice) with the upper layer shifted by half a lattice spacing
in x and y (nearest inter-layer site distance 2.9 A), the overlap test against ALL placed molecules, up to 2000
orientations per site, and if a site still fails it is dropped and another free site is used.

    python aa/confined_graphene/build_h10.py 10 170 aa/confined_graphene/H10/start.data
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from preprocess import load_atom_types  # noqa: E402

H, N, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
XLO, XHI, YLO, YHI = -25.174493, 4.29808, -24.815001, 4.963
LX, LY = XHI - XLO, YHI - YLO
ZLO, ZHI = -10.0, H + 30.0
mO, mH, mC = 15.9994, 1.00794, 12.0107
qO, qH = -0.8472, 0.4236
rOH, theta = 1.0, np.deg2rad(109.47)
rng = np.random.default_rng(int(H * 1000) + N)

z = load_atom_types(os.path.join(ROOT, "Data", "Confined_H2O", "AA", "z.npy"))
X0 = np.asarray(np.load(os.path.join(ROOT, "Data", "Confined_H2O", "AA", "pos_COM.npy"), mmap_mode="r")[0])
C = X0[z == 5].copy()
assert np.allclose(np.unique(np.round(C[:, 2], 3)), [0.0, 80.0])
C[C[:, 2] > 40, 2] = H

zmin, zmax = 3.0, H - 3.0
nx, ny, nz = 10, 10, 2
sx, sy = LX / nx, LY / ny
g = []
for k in range(nz):
    off = 0.5 * k
    zz = zmin + (k + 0.5) / nz * (zmax - zmin)
    for i in range(nx):
        for j in range(ny):
            g.append([XLO + (i + 0.5 + off) * sx, YLO + (j + 0.5 + off) * sy, zz])
g = np.array(g)
order = rng.permutation(len(g))
h1 = rOH * np.array([np.sin(theta / 2), 0, np.cos(theta / 2)])
h2 = rOH * np.array([-np.sin(theta / 2), 0, np.cos(theta / 2)])
shift = mH * (h1 + h2) / (mO + 2 * mH)


def rot():
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z_ = q
    return np.array([[1 - 2 * (y * y + z_ * z_), 2 * (x * y - z_ * w), 2 * (x * z_ + y * w)],
                     [2 * (x * y + z_ * w), 1 - 2 * (x * x + z_ * z_), 2 * (y * z_ - x * w)],
                     [2 * (x * z_ + y * w), 2 * (y * z_ + x * w), 1 - 2 * (x * x + y * y)]])


box = np.array([LX, LY, 1e9])
mols, dropped = [], 0
for s in order:
    if len(mols) == N:
        break
    c = g[s]
    for _ in range(2000):
        R = rot()
        o = c - R @ shift
        m = np.array([o, o + R @ h1, o + R @ h2])
        if not mols:
            ok = True
            break
        prev = np.concatenate(mols)
        dd = m[:, None] - prev[None]
        dd -= np.round(dd / box) * box
        ok = np.sqrt((dd ** 2).sum(-1)).min() > 1.7
        if ok:
            break
    if ok:
        mols.append(m)
    else:
        dropped += 1
assert len(mols) == N, f"only {len(mols)} placed"
W = np.array(mols)

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
print(f"H={H} N={N} staggered lattice {nx}x{ny}x{nz} spacing {sx:.2f} {sy:.2f} {(zmax - zmin) / nz:.2f} A, "
      f"sites dropped {dropped} -> {out}")
