"""All-atom targets for water between two graphene walls: density and layer-conditioned VACFs.

The functions (density, layer_vacf, EDGES, FINE_EDGES) are used by aa/confined_graphene/make_dataset.py for
the re-run all-atom data (Data/Confined_H2O_v2_H<H>). Run as a script, this module analyses the earlier
80 A trajectory Data/Confined_H2O/AA (cross-check data set): 2123 water molecules (one bead each, type 7)
between two frozen graphene sheets (336 C each, type 5) at z = 0 and z = 80 A; box 29.47 x 29.78 A
(periodic in x, y; see LX); 100000 frames, 1 fs apart; velocities in A/fs.

Outputs (Data/Confined_H2O/targets/):
  aa_density.npz     mass density rho(z) and rho(d) (d = distance to the nearest wall), 0.1 A bins
  aa_layer_vacf.npz  VACF conditioned on the layer at the time origin, parallel (x, y) and
                     perpendicular (z) components, 1000 lags, for the d-bins in EDGES

    python experiments/confined/aa_targets.py            # density, layer VACF
    python experiments/confined/aa_targets.py --noise    # aa_layer_vacf_noise.npz (two halves)
    python experiments/confined/aa_targets.py --fine     # aa_layer_vacf_fine.npz (FINE_EDGES, with errors)
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from preprocess import load_atom_types  # noqa: E402

AA = os.path.join(ROOT, "Data", "Confined_H2O", "AA")
OUT = os.path.join(ROOT, "Data", "Confined_H2O", "targets")
Z_LO, Z_HI = 0.0, 80.0                   # graphene planes (A)
# periodic box of the graphene sheets (preprocess.CONFINED_BOX; before 2026-10-11 that constant had
# Lx = 25.17 A, the wall's x minimum, instead of the period 24 * 1.418 * sqrt(3) / 2 = 29.4726 A)
LX, LY = 29.472577, 29.778001
LXY = LX * LY                            # wall area (A^2)
TO_GCC = 18.01528 / 0.602214076          # molecules / A^3 -> g / cm^3
TMAX = 1000                              # VACF lags (fs)
# d-bins (A): the first three edges are the minima of the AA rho(d) (layers 1-3), then wider
# bins out to the channel centre (d = 30-40 A, bulk reference)
EDGES = np.array([0.0, 5.2, 8.3, 11.5, 14.5, 18.0, 22.0, 30.0, 40.0])
# fine bins for the continuous (knot) model: 0.5 A up to 8 A, 1 A to 12 A, 2 A to 20 A, then wide;
# the first bin also holds the few molecules closer than 3 A
FINE_EDGES = np.concatenate(([0.0], np.arange(3.0, 8.01, 0.5), [9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0]))


def load_water():
    z = load_atom_types(os.path.join(AA, "z.npy"))
    w = np.flatnonzero(z == 7)
    X = np.load(os.path.join(AA, "pos_COM.npy"), mmap_mode="r")
    V = np.load(os.path.join(AA, "vel_COM.npy"), mmap_mode="r")
    T = X.shape[0]
    zz = np.empty((T, len(w)), np.float32)
    vv = np.empty((T, len(w), 3), np.float32)
    for i in range(0, T, 5000):
        zz[i:i + 5000] = X[i:i + 5000, w, 2]
        vv[i:i + 5000] = V[i:i + 5000][:, w]
    return zz, vv


def density(zz, dz=0.1):
    edges = np.arange(Z_LO, Z_HI + dz / 2, dz)
    h, _ = np.histogram(zz[::10].ravel(), edges)
    rho_z = h / (len(zz[::10]) * LXY * dz) * TO_GCC
    d = np.minimum(zz[::10] - Z_LO, Z_HI - zz[::10]).ravel()
    dedges = np.arange(0, (Z_HI - Z_LO) / 2 + dz / 2, dz)
    hd, _ = np.histogram(d, dedges)
    rho_d = hd / (len(zz[::10]) * 2 * LXY * dz) * TO_GCC
    return 0.5 * (edges[1:] + edges[:-1]), rho_z, 0.5 * (dedges[1:] + dedges[:-1]), rho_d


def layer_vacf(zz, vv, edges, device="cuda:0", chunk=64):
    """C_b(t) = sum_{i,t0} 1[d_i(t0) in b] v_i(t0) v_i(t0 + t) / sum 1[...], t0 < T - TMAX."""
    T, N = zz.shape
    n0 = T - TMAX
    nfft = 1 << int(np.ceil(np.log2(T + TMAX)))
    d = np.minimum(zz - Z_LO, Z_HI - zz)
    b = np.digitize(d, edges) - 1                       # (T, N) bin index
    B = len(edges) - 1
    num = torch.zeros(B, 3, TMAX, dtype=torch.float64, device=device)
    cnt = np.array([(b[:n0] == k).sum() for k in range(B)], float)
    for i in range(0, N, chunk):
        v = torch.tensor(vv[:, i:i + chunk], device=device, dtype=torch.float64)   # (T, n, 3)
        Fv = torch.fft.rfft(v, n=nfft, dim=0)
        bi = torch.tensor(b[:n0, i:i + chunk], device=device)
        for k in range(B):
            a = torch.zeros_like(v)
            a[:n0] = v[:n0] * (bi == k)[..., None]
            c = torch.fft.irfft(torch.conj(torch.fft.rfft(a, n=nfft, dim=0)) * Fv, n=nfft, dim=0)[:TMAX]
            num[k] += c.sum(1).T
    C = (num.cpu().numpy() / cnt[:, None, None])
    par = 0.5 * (C[:, 0] + C[:, 1])
    perp = C[:, 2]
    return par, perp, cnt / n0


def main():
    os.makedirs(OUT, exist_ok=True)
    zz, vv = load_water()
    print("loaded", zz.shape, vv.shape, flush=True)
    z, rho_z, dmid, rho_d = density(zz)
    np.savez(os.path.join(OUT, "aa_density.npz"), z=z, rho_z=rho_z, d=dmid, rho_d=rho_d)
    print("density saved; rho(centre 15-40 A) =", rho_d[dmid > 15].mean(), flush=True)
    par, perp, nmean = layer_vacf(zz, vv, EDGES)
    np.savez(os.path.join(OUT, "aa_layer_vacf.npz"), edges=EDGES, par=par, perp=perp, n_mean=nmean)
    for k in range(len(EDGES) - 1):
        print(f"bin {EDGES[k]:5.1f}-{EDGES[k + 1]:5.1f} A  <n>={nmean[k]:6.1f}  "
              f"par C(0)={par[k, 0]:.3e}  perp C(0)={perp[k, 0]:.3e}")


def fine_targets():
    """Layer VACFs and their statistical error (two halves) on FINE_EDGES -> aa_layer_vacf_fine.npz."""
    zz, vv = load_water()
    par, perp, nmean = layer_vacf(zz, vv, FINE_EDGES)
    T = zz.shape[0] // 2
    halves = [layer_vacf(zz[i * T:(i + 1) * T], vv[i * T:(i + 1) * T], FINE_EDGES) for i in range(2)]
    n = lambda c: c / c[:, :1]
    err = {f"err_{key}": np.sqrt(np.mean((n(halves[0][j]) - n(halves[1][j])) ** 2, 1)) / 2
           for j, key in enumerate(("par", "perp"))}
    np.savez(os.path.join(OUT, "aa_layer_vacf_fine.npz"), edges=FINE_EDGES, par=par, perp=perp, n_mean=nmean, **err)
    for k in range(len(FINE_EDGES) - 1):
        print(f"bin {FINE_EDGES[k]:5.1f}-{FINE_EDGES[k + 1]:5.1f} A  <n>={nmean[k]:6.1f}  "
              f"error par {err['err_par'][k]:.4f} perp {err['err_perp'][k]:.4f}")


def noise_floor():
    """Statistical error of the AA layer VACFs: halves of the trajectory, sigma ~ |C_1 - C_2| / 2."""
    zz, vv = load_water()
    T = zz.shape[0] // 2
    halves = [layer_vacf(zz[i * T:(i + 1) * T], vv[i * T:(i + 1) * T], EDGES) for i in range(2)]
    n = lambda c: c / c[:, :1]
    out = {}
    for j, key in enumerate(("par", "perp")):
        d = n(halves[0][j]) - n(halves[1][j])
        out[key] = np.sqrt(np.mean(d ** 2, 1)) / 2   # per-bin RMS error of the full-length VACF
    np.savez(os.path.join(OUT, "aa_layer_vacf_noise.npz"), edges=EDGES, **out)
    for k in range(len(EDGES) - 1):
        print(f"bin {EDGES[k]:5.1f}-{EDGES[k + 1]:5.1f}  rms error par {out['par'][k]:.4f} perp {out['perp'][k]:.4f}")


if __name__ == "__main__":
    fine_targets() if "--fine" in sys.argv else noise_floor() if "--noise" in sys.argv else main()
