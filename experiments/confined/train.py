"""Train a memory field (wall-distance-dependent GLE kernel) for water between graphene sheets (BPTT).

System: data/processed/<system> (CG water beads between frozen graphene walls at z = 0 and z = --wall-gap,
tabulated CG1_CG1 / CG1_GR1 potentials, 300 K, 1 fs). Target: the all-atom VACF conditioned on the
d-bin at the time origin, parallel and perpendicular to the walls (data/processed/<system>/targets_slab/,
written by experiments/confined/allatom/make_dataset.py --slab-frame, or --data-dir).

Models (--model, --bins, --groups):
  --model knot          memory field: every mode parameter its own curve in d (knots, straight lines
                        between), separate for (x, y) and z; use with --target fine and --smooth
  --model knot --superpose   the same with the effects of both walls added (with --load-state)
  --bins 1 --groups 1   one kernel for every bead and component (position-independent)
  --bins 8 --groups 2   one kernel per d-bin, separate for (x, y) and z
Initialisation: the bulk-water kernel results/H2O_osc_baoab (4 modes, experiments/train_osc.py) everywhere.
--iterations 0 only runs the production evaluation of the initial kernel (bulk, or a saved one with --load-state).
The production stores z of every bead every 5 steps (z_traj.npy) for experiments/confined/residence.py.

    python experiments/confined/train.py --system graphene_water_H80 --wall-gap 80 --model knot \
        --groups 2 --target fine --smooth 3 -n 300 --avg-last 50 --prod 50000 --out results/memory_field_H80
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from diffgle import memory_field as mf  # noqa: E402
from diffgle.constants import DT, PROCESSED_DIR, RESULT_DIR, TEMPERATURE, TIMEFACTOR  # noqa: E402
from diffgle.memory_field import TabulatedGPU, ZBAOAB, ZDampedOsc, ZKnotDampedOsc, ZKnotSuperposed, layer_vacf_cg  # noqa: E402
from diffgle.potentials import load_table  # noqa: E402
from diffgle.systems import ALIASES, WALL, WATER, load_system  # noqa: E402
from experiments.confined.aa_targets import LX, LY  # noqa: E402  (density normalisation)

DATA = None          # directory of the AA targets, set in main()
TD = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bins", "knot"], default="bins",
                   help="bins: one kernel per d-bin; knot: memory field, every mode parameter its own curve in d "
                        "(diffgle.memory_field.MemoryField)")
    p.add_argument("--system", default="graphene_water_H80", help="diffgle.systems name (data/processed/<system>)")
    p.add_argument("--wall-gap", type=float, default=80.0, help="distance between the graphene sheets (A)")
    p.add_argument("--data-dir", default=None, help="directory of the AA targets (default data/processed/<system>/targets_slab)")
    p.add_argument("--target", choices=["coarse", "fine"], default="coarse",
                   help="AA layer VACFs on the 8 layer bins (aa_layer_vacf.npz) or on 22 fine bins (aa_layer_vacf_fine.npz)")
    p.add_argument("--smooth", type=float, default=0.0, help="weight of the knot model's smoothness penalty")
    p.add_argument("--superpose", action="store_true",
                   help="knot model: add the effects of both walls (MemoryFieldSuperposed; use with --load-state)")
    p.add_argument("--init-bins", default=None,
                   help="knot model: start from a trained 8-bin state dict (every knot takes its bin's values)")
    p.add_argument("--init-bins-clean", action="store_true",
                   help="with --init-bins: keep the contact-layer bin, replace all other bins by their bead-weighted mean")
    p.add_argument("--load-state", default=None,
                   help="load a saved thermostat state dict after building the model (with -n 0: production only)")
    p.add_argument("--avg-last", type=int, default=0,
                   help="use the mean of the parameters over the last N iterations for saving and production")
    p.add_argument("--bins", type=int, choices=[1, 8], default=8)
    p.add_argument("--groups", type=int, choices=[1, 2], default=2)
    p.add_argument("--iterations", "-n", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--window", type=int, default=2000, help="BPTT window (steps); 1000 VACF lags + 1000 steps of time origins")
    p.add_argument("--stride", type=int, default=10, help="VACF time-origin stride in training")
    p.add_argument("--equil", type=int, default=3000)
    p.add_argument("--prod", type=int, default=20000, help="production steps for the evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=os.path.join(RESULT_DIR, "H2O_osc_baoab", "thermostat_state_dict.pth"))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def targets(device, fine=False, data_dir=None):
    """Edges, normalised AA layer VACFs (par, perp) and per-bin weights from data_dir (default: DATA, set in main).

    Weights: 1 for the coarse target (as in all earlier runs); for the fine target the mean number of
    molecules per bin divided by its mean, so that bins count in proportion to their statistics."""
    d = np.load(os.path.join(data_dir or DATA, "aa_layer_vacf_fine.npz" if fine else "aa_layer_vacf.npz"))
    norm = lambda c: torch.tensor(c / c[:, :1], dtype=torch.float, device=device)
    w = d["n_mean"] / d["n_mean"].mean() if fine else np.ones(len(d["edges"]) - 1)
    if fine:
        w[d["edges"][1:] <= 3.0] = 0.0   # bins closer than 3 A: ~3 molecules in the AA data, noise only
    return d["edges"], norm(d["par"]), norm(d["perp"]), torch.tensor(w, dtype=torch.float, device=device)


def bulk_modes(path):
    sd = torch.load(path, map_location="cpu")
    sp = torch.nn.functional.softplus
    return sp(sd["raw_a"]), sp(sd["raw_gamma1"]), sp(sd["raw_gamma2"]), sd["omega"]


def loss_fn(vs, zs, edges, aa_par, aa_perp, stride, weights=None):
    """Sum over bins (weighted) of the squared VACF error, parallel + perpendicular; also per-bin parts."""
    par, perp, cnt = layer_vacf_cg(vs, zs, edges, TD, stride)
    par, perp = par / par[:, :1], perp / perp[:, :1]
    l_par = (par - aa_par).pow(2).sum(1)
    l_perp = (perp - aa_perp).pow(2).sum(1)
    w = 1.0 if weights is None else weights
    terms = w * (l_par + l_perp)
    if weights is not None:          # a weight-0 bin may be empty in a window (NaN): drop it instead of 0 * NaN
        terms = torch.where(weights > 0, terms, torch.zeros_like(terms))
    return terms.sum(), par, perp, l_par, l_perp


def density(zs_np, dz=0.1):
    d = np.minimum(zs_np, mf.Z_HI - zs_np).ravel()
    edges = np.arange(0, mf.Z_HI / 2 + dz / 2, dz)
    h, _ = np.histogram(d, edges)
    lxy = LX * LY
    return 0.5 * (edges[1:] + edges[:-1]), h / (zs_np.shape[0] * 2 * lxy * dz) * 18.01528 / 0.602214076


def main():
    args = parse_args()
    assert args.window >= TD, "BPTT window must cover the 1000 VACF lags"
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    dev = args.device
    global DATA
    mf.set_wall_gap(args.wall_gap)     # wall_distance(z) = min(z, wall_gap - z) everywhere in memory_field
    args.system = ALIASES.get(args.system, args.system)
    DATA = os.path.abspath(args.data_dir) if args.data_dir else os.path.join(PROCESSED_DIR, args.system, "targets_slab")
    s = load_system(args.system, dev)
    # GPU version of load_system's CG1_CG1 / CG1_GR1 forces (same forces as the fixed
    # diffgle.potentials.Tabulated_specific, without the per-step CPU spline evaluation)
    d = os.path.join(PROCESSED_DIR, args.system, "CG")
    s.potential = TabulatedGPU(s.cell, s.types, [load_table(os.path.join(d, "CG1_CG1.pot")),
                                                 load_table(os.path.join(d, "CG1_GR1.pot"))],
                               [[WATER, WATER], [WATER, WALL]])
    edges_aa, aa_par, aa_perp, _ = targets(dev)                       # coarse: always evaluated
    edges_tr, tr_par, tr_perp, tr_w = targets(dev, fine=args.target == "fine")   # training target
    edges_model = edges_aa if args.bins == 8 else np.array([0.0, 40.0])
    mob = s.mobile
    q, v = s.q0[mob].clone(), s.v0[mob].clone() * TIMEFACTOR  # vel0.npy is in A/fs, the integrator uses A / (TIMEFACTOR fs)
    mass = s.mass[mob]
    a, g1, g2, w = bulk_modes(args.init)
    if args.model == "knot":
        if args.init_bins:
            sd = torch.load(args.init_bins, map_location="cpu")
            if args.init_bins_clean:
                n = torch.tensor(np.load(os.path.join(DATA, "aa_layer_vacf.npz"))["n_mean"], dtype=torch.float)
                for k in ("raw_a", "raw_gamma1", "raw_gamma2", "omega"):
                    vals = sd[k].clone()                               # (P, B, M)
                    mean = (vals[:, 1:] * n[1:, None]).sum(1) / n[1:].sum()
                    vals[:, 1:] = mean[:, None]
                    sd[k] = vals
            th = ZKnotDampedOsc(*ZKnotDampedOsc.init_from_bins(sd, edges_aa), groups=args.groups).to(dev)
        elif args.superpose:
            th = ZKnotSuperposed(a, g1, g2, w, groups=args.groups, wall_gap=args.wall_gap).to(dev)
        else:
            th = ZKnotDampedOsc(a, g1, g2, w, groups=args.groups).to(dev)
    else:
        th = ZDampedOsc(edges_model, a, g1, g2, w, groups=args.groups).to(dev)
    if args.load_state:
        sd = torch.load(args.load_state, map_location=dev)
        if args.superpose:
            sd.pop("d_grid", None)          # the superposed model tabulates in z over its own channel
        th.load_state_dict(sd)
    integ = ZBAOAB(s.potential, th, mass, DT, TEMPERATURE, s.q0, mob)

    with torch.no_grad():
        q, vs, zs, temps = integ.simulate(q, v, args.equil)
    v = vs[-1]
    print(f"equil <T> (last half) {np.mean(temps[len(temps) // 2:]):.1f} K", flush=True)

    opt = torch.optim.Adam(th.parameters(), lr=args.lr)
    # cosine decay to eta_min = 0.05 lr (absolute, as CosineAnnealingLR: the same eta_min for every group)
    cosf = lambda i: 0.5 * (1 + np.cos(np.pi * min(i, args.iterations) / max(args.iterations, 1)))
    eta_min = 0.05 * args.lr
    lams = [lambda i, b=g["lr"]: (eta_min + (b - eta_min) * cosf(i)) / b for g in opt.param_groups]   # b: initial lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lams)
    hist = []
    t_start = time.time()
    avg, n_avg = None, 0
    for it in range(args.iterations):
        th.detach_state()
        q, vs, zs, temps = integ.simulate(q, v.detach(), args.window)
        v = vs[-1]
        loss, _, _, l_par, l_perp = loss_fn(vs, zs, edges_tr, tr_par, tr_perp, args.stride, tr_w)
        pen = args.smooth * th.smoothness() if args.model == "knot" and args.smooth > 0 else torch.zeros((), device=dev)
        opt.zero_grad()
        (loss + pen).backward()
        torch.nn.utils.clip_grad_norm_(th.parameters(), 1.0)
        opt.step()
        sched.step()
        if args.avg_last and it >= args.iterations - args.avg_last:
            cur = [p.detach().clone() for p in th.parameters()]
            avg = cur if avg is None else [a_ + c_ for a_, c_ in zip(avg, cur)]
            n_avg += 1
        hist.append(dict(loss=loss.item(), penalty=pen.item(), l_par=l_par.tolist(), l_perp=l_perp.tolist(), T=float(np.mean(temps))))
        print(f"{it:4d} loss {loss.item():.4e} pen {pen.item():.2e}  par {l_par.sum().item():.3e} perp {l_perp.sum().item():.3e}  "
              f"<T> {np.mean(temps):6.1f}  {time.time() - t_start:6.0f} s", flush=True)
    if avg is not None:
        torch.save(th.state_dict(), os.path.join(args.out, "thermostat_state_dict_last.pth"))
        with torch.no_grad():
            for p_, a_ in zip(th.parameters(), avg):
                p_.copy_(a_ / n_avg)
        print(f"parameters averaged over the last {n_avg} iterations", flush=True)
    torch.save(th.state_dict(), os.path.join(args.out, "thermostat_state_dict.pth"))

    # production: separate equilibration, then args.prod steps
    th.detach_state()
    with torch.no_grad():
        q, vs, zs, temps = integ.simulate(q, v.detach(), 2000)
        q, vs, zs, temps = integ.simulate(q, vs[-1], args.prod)
        _, par, perp, l_par, l_perp = loss_fn(vs, zs, edges_aa, aa_par, aa_perp, 10)
        _, par_f, perp_f, l_par_f, l_perp_f = loss_fn(vs, zs, edges_tr, tr_par, tr_perp, 10)
        tgrid = torch.arange(TD, device=dev, dtype=torch.float) * DT
        if args.model == "knot":
            # kernels at the population-weighted mean d of the AA bins (so the bin analyses apply),
            # and on a 0.25 A grid
            d_eval = torch.tensor([3.83, 6.79, 9.91, 13.0, 16.25, 19.99, 25.99, 35.0], device=dev)
            kappa = th.kernel_at(d_eval, tgrid).cpu().numpy()
            d_fine = torch.arange(2.5, 40.01, 0.25, device=dev)
            kappa_d = th.kernel_at(d_fine, tgrid[:400]).cpu().numpy() * (1e3 / TIMEFACTOR) ** 2
        else:
            kappa = th.continuous_kernel(tgrid).cpu().numpy()         # (P, B, TD), sim units
    zs_np = zs[::10].cpu().numpy()
    # z of every bead every 5 steps, for analyses beyond the trained VACF (layer residence, exchange)
    np.save(os.path.join(args.out, "z_traj.npy"), zs[::5].cpu().numpy().astype(np.float32))
    dmid, rho = density(zs_np)
    n = TD
    res = dict(
        edges=edges_aa, edges_model=edges_model, par=par.cpu().numpy(), perp=perp.cpu().numpy(),
        rmse_par=np.sqrt(l_par.cpu().numpy() / n), rmse_perp=np.sqrt(l_perp.cpu().numpy() / n),
        T_mean=float(np.mean(temps)), T_std=float(np.std(temps)), d=dmid, rho_d=rho,
        kappa_ps2=kappa * (1e3 / TIMEFACTOR) ** 2,
        t_fs=np.arange(TD), history=hist,
        params=([p.detach().cpu().numpy() for p in th.params_at(d_eval)] if args.model == "knot"
                else [p.detach().cpu().numpy() for p in th.params()]),
        edges_target=edges_tr, par_target=par_f.cpu().numpy(), perp_target=perp_f.cpu().numpy(),
        rmse_par_target=np.sqrt(l_par_f.cpu().numpy() / n), rmse_perp_target=np.sqrt(l_perp_f.cpu().numpy() / n),
        **(dict(d_fine=d_fine.cpu().numpy(), kappa_d_ps2=kappa_d,
                params_d=[p.detach().cpu().numpy() for p in th.params_at(d_fine)]) if args.model == "knot" else {}),
        **(dict(knots=th.knots.cpu().numpy(), values=th.values.detach().cpu().numpy()) if args.model == "knot" else {}),
        config=vars(args),
    )
    with open(os.path.join(args.out, "result.pkl"), "wb") as f:
        pickle.dump(res, f)
    print("production <T> %.1f K" % res["T_mean"])
    for k in range(len(edges_aa) - 1):
        print(f"bin {edges_aa[k]:5.1f}-{edges_aa[k + 1]:5.1f}  rmse par {res['rmse_par'][k]:.4f} perp {res['rmse_perp'][k]:.4f}")


if __name__ == "__main__":
    main()
