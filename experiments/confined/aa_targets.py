"""All-atom targets for water between two graphene walls: density and layer-conditioned VACFs.

Used by experiments/confined/allatom/make_dataset.py (systems graphene_water_H<H>), which sets Z_HI = H:
  density(zz)                     mass density along z and against the distance d to the nearest wall
  layer_vacf(zz, vv, edges)       VACF conditioned on the d-bin at the time origin, parallel (x, y) and
                                  perpendicular (z), 1000 lags
  EDGES (8 layer bins), FINE_EDGES (22 bins of the training target)
"""

import numpy as np
import torch

Z_LO, Z_HI = 0.0, 80.0                   # graphene planes (A)
# periodic box of the graphene sheets: x period 24 * 1.418 * sqrt(3) / 2, y period 7 * 3 * 1.418 A
LX, LY = 29.472577, 29.778001
LXY = LX * LY                            # wall area (A^2)
TO_GCC = 18.01528 / 0.602214076          # molecules / A^3 -> g / cm^3
TMAX = 1000                              # VACF lags (fs)
# d-bins (A): the first three edges are the minima of the AA rho(d) (layers 1-3), then wider
# bins out to the channel centre (d = 30-40 A, bulk reference)
EDGES = np.array([0.0, 5.2, 8.3, 11.5, 14.5, 18.0, 22.0, 30.0, 40.0])
# fine bins of the memory-field training target: 0.5 A up to 8 A, 1 A to 12 A, 2 A to 20 A, then wide;
# the first bin also holds the few molecules closer than 3 A
FINE_EDGES = np.concatenate(([0.0], np.arange(3.0, 8.01, 0.5), [9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0]))


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
