"""Contact-layer residence and layer exchange of confined water: an observable that the memory field is not
trained on (training uses layer VACFs over 1 ps; residence lasts tens of ps).

From z(t) of every molecule (wall distance d = min(z, H - z)):
  - survival probability S(t) that a molecule in the contact layer (d < d1, d1 = first minimum of the all-atom
    density) at t0 is still in it at every time up to t0 + t; excursions shorter than t_star are not counted
    as leaving (Impey, Madden and McDonald 1983); continuous (t_star = 0) and t_star = 2 ps;
  - residence time tau = int_0^tmax S dt (and the 1/e time);
  - exchange rate between the cores of the first two layers (transition-based: a molecule belongs to the core
    it visited last; rate = transitions 1 -> 2 per molecule and ps spent assigned to layer 1).

    python experiments/confined/residence.py aa H            # all-atom run data/raw/graphene_water_H<H>/aa/com.npy (every 5 fs)
    python experiments/confined/residence.py cg H RUN_DIR    # CG run of train.py with z_traj.npy (z every 5 steps = 5 fs)
    -> results/residence/<label>.npz and a printed summary
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "residence")
RAW = os.path.join(ROOT, "data", "raw")
DT_PS = 0.005                 # 5 fs between frames
T_MAX_PS = 20.0               # survival evaluated to 20 ps (CG productions are 20-50 ps)
ORIGIN_PS = 0.5               # spacing of time origins


def layer_geometry():
    """First density minimum d1 and the cores of layers 1 and 2 from the all-atom 80 A density (v2)."""
    D = np.load(os.path.join(ROOT, "data", "processed", "graphene_water_H80", "targets_slab", "aa_density.npz"))
    d, rho = D["d"], D["rho_d"]
    m = (d > 2.5) & (d < 9)
    dd, rr = d[m], rho[m]
    i1 = np.argmax(rr)                                     # first peak
    j = i1 + np.argmin(rr[i1:np.searchsorted(dd, 7.5)])    # first minimum after it
    i2 = j + np.argmax(rr[j:])                             # second peak
    return dict(d_peak1=dd[i1], d1=dd[j], d_peak2=dd[i2])


def fill_short_gaps(inside, n_star):
    """inside (T, N) bool; set False runs shorter than n_star frames that have True on both sides to True."""
    if n_star <= 0:
        return inside
    T = inside.shape[0]
    big = np.int32(1 << 30)
    pt = np.empty(inside.shape, np.int32)             # frames since the last True (0 on True)
    nt = np.empty(inside.shape, np.int32)             # frames until the next True
    pt[0] = np.where(inside[0], 0, big)
    for t in range(1, T):
        pt[t] = np.where(inside[t], 0, np.minimum(pt[t - 1] + 1, big))
    nt[-1] = np.where(inside[-1], 0, big)
    for t in range(T - 2, -1, -1):
        nt[t] = np.where(inside[t], 0, np.minimum(nt[t + 1] + 1, big))
    gap = pt.astype(np.int64) + nt - 1
    return inside | ((pt < big) & (nt < big) & (gap < n_star))


def survival(inside, n_max, stride):
    """Continuous survival in `inside` (T, N) from origins every `stride` frames, to lag n_max."""
    T = inside.shape[0]
    # frames until first exit, for every frame and molecule: run length of True from each frame
    run = np.zeros_like(inside, dtype=np.int32)
    run[-1] = inside[-1]
    for t in range(T - 2, -1, -1):
        run[t] = np.where(inside[t], run[t + 1] + 1, 0)
    S = np.zeros(n_max + 1)
    cnt = 0
    for t0 in range(0, T - n_max, stride):
        r = run[t0][inside[t0]]
        cnt += len(r)
        S += (r[:, None] > np.arange(n_max + 1)[None]).sum(0)
    return S / max(cnt, 1)


def exchange_rate(d, core1, core2):
    """Transition-based rate 1 -> 2 (per ps per molecule assigned to layer 1) and 2 -> 1."""
    T, N = d.shape
    state = np.full(N, -1)
    time_in = np.zeros(2)
    trans = np.zeros(2)
    for t in range(T):
        in1 = d[t] < core1
        in2 = (d[t] > core2[0]) & (d[t] < core2[1])
        trans[0] += np.sum((state == 0) & in2)
        trans[1] += np.sum((state == 1) & in1)
        state = np.where(in1, 0, np.where(in2, 1, state))
        time_in[0] += np.sum(state == 0)
        time_in[1] += np.sum(state == 1)
    return trans / (time_in * DT_PS)


def analyse(z, H, label):
    d = np.minimum(z, H - z)
    g = layer_geometry()
    inside = d < g["d1"]
    n_max, stride = int(T_MAX_PS / DT_PS), int(ORIGIN_PS / DT_PS)
    t = np.arange(n_max + 1) * DT_PS
    S0 = survival(inside, n_max, stride)
    S2 = survival(fill_short_gaps(inside, int(2.0 / DT_PS)), n_max, stride)
    tau = lambda S: (np.trapezoid(S, t), t[np.argmax(S < np.exp(-1))] if (S < np.exp(-1)).any() else np.nan)
    core1 = g["d_peak1"] + 0.5 * (g["d1"] - g["d_peak1"])
    core2 = (g["d_peak2"] - 0.5, g["d_peak2"] + 0.5)
    k = exchange_rate(d, core1, core2)
    m = (t >= 5) & (t <= 20)
    esc = -np.polyfit(t[m], np.log(S0[m]), 1)[0]          # long-time escape rate from the contact layer
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{label}.npz"), t_ps=t, S_cont=S0, S_2ps=S2, k12=k[0], k21=k[1], esc=esc,
             d1=g["d1"], core1=core1, core2=np.array(core2), frames=d.shape[0], n_mol=d.shape[1])
    print(f"{label}: frames {d.shape[0]} ({d.shape[0] * DT_PS:.0f} ps), molecules {d.shape[1]}, "
          f"contact layer d < {g['d1']:.2f} A (peaks {g['d_peak1']:.2f}, {g['d_peak2']:.2f})")
    for name, S in (("continuous", S0), ("t*=2ps", S2)):
        a, e = tau(S)
        print(f"  {name:10s}: int_0^{T_MAX_PS:g} S = {a:6.2f} ps, 1/e time {e:6.2f} ps, S(5 ps) {S[int(5 / DT_PS)]:.3f}, "
              f"S(20 ps) {S[-1]:.3f}")
    print(f"  escape rate (slope of ln S, 5-20 ps) {esc:.4f} /ps")
    print(f"  exchange cores d<{core1:.2f} | {core2[0]:.2f}-{core2[1]:.2f}: k12 {k[0]:.4f} /ps, k21 {k[1]:.4f} /ps")


def main():
    kind, H = sys.argv[1], float(sys.argv[2])
    if kind == "aa":
        com = np.load(os.path.join(RAW, f"graphene_water_H{H:g}", "aa", "com.npy"), mmap_mode="r")
        z = np.array(com[::5, :, 2], dtype=np.float32)       # every 5 fs
        analyse(z, H, f"aa_H{H:g}")
    else:
        run = sys.argv[3]
        z = np.load(os.path.join(run, "z_traj.npy"))
        z = z[:, z.std(0) > 1e-6]                              # drop the frozen wall beads
        analyse(z, H, os.path.basename(os.path.normpath(run)))


if __name__ == "__main__":
    main()
