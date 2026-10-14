"""Turn an aa/confined_graphene/H<H> all-atom run into (1) a CG system directory and (2) all-atom training targets.

(1) Data/Confined_H2O_v2_H<H>/: pos0.npy, vel0.npy (walls first, then water molecules; last production frame,
    velocities in A/fs as in Data/Confined_H2O), z.npy (types 5 wall / 7 water), box.npy, CG/ (copy of the CG
    potentials of Data/Confined_H2O), vacf_aa.npy, msd_aa.npy (all water, as load_system expects),
    z_aa.npy / density_aa.npy (density along z).
(2) Data/Confined_H2O_v2_H<H>/targets/: aa_density.npz, aa_layer_vacf.npz (8 bins, + aa_layer_vacf_noise.npz)
    and aa_layer_vacf_fine.npz (22 bins with errors), with the functions of experiments/confined/aa_targets.py and
    the wall distance d = min(z, H - z).

    python make_dataset.py 80
    python make_dataset.py 80 --slab-frame    # targets only, velocities relative to the water slab (see below)

--slab-frame: the water slab as a whole slides along the frictionless graphene; its centre-of-mass velocity
(x, y) decorrelates over ~10-20 ps and in a 100 ps run is not equilibrated (its energy was 0.3-3.7 times kT/2
per component depending on the channel), which adds a nearly constant, run-dependent offset to every molecular
VACF. With --slab-frame the slab's (x, y) centre-of-mass velocity is subtracted at every frame before the layer
VACFs are computed; the targets go to Data/Confined_H2O_v2_H<H>/targets_slab/ (the default training targets of
experiments/confined/train.py) and nothing else is written.
"""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import experiments.confined.aa_targets as ca  # noqa: E402

H = float(sys.argv[1])
SLAB = "--slab-frame" in sys.argv
tag = f"H{H:g}"
run = os.path.join(os.path.dirname(os.path.abspath(__file__)), tag)
sysdir = os.path.join(ROOT, "Data", f"Confined_H2O_v2_{tag}")
tgt = os.path.join(sysdir, "targets_slab" if SLAB else "targets")
if not SLAB:
    os.makedirs(os.path.join(sysdir, "CG"), exist_ok=True)
os.makedirs(tgt, exist_ok=True)

x = np.load(os.path.join(run, "com.npy"), mmap_mode="r")      # (T, N, 3) A, unwrapped
v = np.load(os.path.join(run, "vcm.npy"), mmap_mode="r")      # A/fs
T, N, _ = x.shape
zz = np.ascontiguousarray(x[:, :, 2], dtype=np.float32)
vv = np.array(v, dtype=np.float32)          # a writable copy (the .npy is memory-mapped)
print(f"{tag}: {T} frames, {N} molecules, z range {zz.min():.2f}-{zz.max():.2f} A", flush=True)
if SLAB:
    Vs = vv[:, :, :2].mean(1, keepdims=True)
    print(f"slab COM velocity: <V^2>/<v^2> = {(Vs ** 2).mean() / vv[:, :, :2].var():.2e} (1/N = {1 / N:.2e}); subtracted")
    vv[:, :, :2] -= Vs

# ---- (1) CG system directory (not with --slab-frame) ----------------------------------------------------------
def write_system():
    old = os.path.join(ROOT, "Data", "Confined_H2O")
    z_old = np.load(os.path.join(old, "z.npy"))
    C = np.load(os.path.join(old, "pos0.npy"))[z_old == 5].copy()
    C[C[:, 2] > 40, 2] = H
    pos0 = np.concatenate([C, np.asarray(x[-1], np.float64)])
    vel0 = np.concatenate([np.zeros_like(C), np.asarray(v[-1], np.float64)])
    types = np.concatenate([np.full(len(C), 5), np.full(N, 7)]).astype(z_old.dtype)
    np.save(os.path.join(sysdir, "pos0.npy"), pos0)
    np.save(os.path.join(sysdir, "vel0.npy"), vel0)
    np.save(os.path.join(sysdir, "z.npy"), types)
    np.save(os.path.join(sysdir, "box.npy"), np.load(os.path.join(old, "box.npy")))
    for f in ("CG1_CG1.pot", "CG1_GR1.pot"):
        shutil.copy(os.path.join(old, "CG", f), os.path.join(sysdir, "CG", f))
    c = np.zeros(1000)
    for t0 in range(0, T - 1000, 10):
        c += (vv[t0:t0 + 1000] * vv[t0]).mean((1, 2))
    np.save(os.path.join(sysdir, "vacf_aa.npy"), (c / c[0]).astype(np.float64))
    xs = np.asarray(x[::10], np.float64)
    msd = np.array([((xs[k:] - xs[:-k or None]) ** 2).sum(-1).mean() if k else 0.0 for k in range(100)])
    np.save(os.path.join(sysdir, "msd_aa.npy"), np.repeat(msd, 10)[:1000] / 3)
    hist, e = np.histogram(zz[::10].ravel(), np.arange(0, H + 0.25, 0.5))
    np.save(os.path.join(sysdir, "z_aa.npy"), 0.5 * (e[1:] + e[:-1]))
    np.save(os.path.join(sysdir, "density_aa.npy"), hist / hist.sum())


if not SLAB:
    write_system()

# ---- (2) all-atom targets ------------------------------------------------------------------------------------
ca.Z_HI = H                                   # d = min(z, H - z) in the shared functions
z_, rho_z, dmid, rho_d = ca.density(zz)
np.savez(os.path.join(tgt, "aa_density.npz"), z=z_, rho_z=rho_z, d=dmid, rho_d=rho_d)
half = H / 2


def clip_edges(edges):
    e = edges[edges < half]
    return np.append(e, half)


for name, edges in (("aa_layer_vacf", clip_edges(ca.EDGES)), ("aa_layer_vacf_fine", clip_edges(ca.FINE_EDGES))):
    par, perp, nmean = ca.layer_vacf(zz, vv, edges)
    Th = T // 2
    halves = [ca.layer_vacf(zz[i * Th:(i + 1) * Th], vv[i * Th:(i + 1) * Th], edges) for i in range(2)]
    nrm = lambda q: q / q[:, :1]
    err = {k: np.sqrt(np.mean((nrm(halves[0][j]) - nrm(halves[1][j])) ** 2, 1)) / 2
           for j, k in enumerate(("par", "perp"))}
    if name == "aa_layer_vacf":
        np.savez(os.path.join(tgt, name + ".npz"), edges=edges, par=par, perp=perp, n_mean=nmean)
        np.savez(os.path.join(tgt, "aa_layer_vacf_noise.npz"), edges=edges, **err)
    else:
        np.savez(os.path.join(tgt, name + ".npz"), edges=edges, par=par, perp=perp, n_mean=nmean,
                 err_par=err["par"], err_perp=err["perp"])
    print(name, "edges", np.round(edges, 2).tolist(), "n_mean", np.round(nmean, 1).tolist(), flush=True)
print("rho centre (d > H/2 - 5):", rho_d[dmid > half - 5].mean(), "g/cm3")
print("done", sysdir, tgt)
