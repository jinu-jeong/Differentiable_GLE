"""Mode-count scan for the damped-oscillator GLE (GLE-BAOAB), CG water and CG CO2.

Same training as experiments/train_osc.py except the number of
modes M: init by least-squares fit of M modes to results/<system>/plot_dict.pkl (DiffGLE kernel),
torch.manual_seed(0), 3000 equilibration steps, consecutive 1200-step windows, VACF over 1000
lags with origin stride 20, loss = sum of squared VACF differences, Adam lr 1e-2 with cosine
decay to 5 %, grad-norm clip 1.0, 150 iterations. Production as experiments/bulk/eval_production.py:
from data/processed/<system>, 5000 equilibration + 10000 production steps with BAOAB, noise seeds 0 1 2,
VACF over 1000 lags with origin stride 10, RMSE over the 1000 lags, mean kinetic T.

    cd Differentiable_GLE && python experiments/bulk/mode_scan.py train -s H2O -m 2   # -> results/modescan_H2O_M2/
    cd Differentiable_GLE && python experiments/bulk/mode_scan.py production -s H2O -m 2  # production only, after a crash
    cd Differentiable_GLE && python experiments/bulk/mode_scan.py collect             # -> results/bulk/modescan_*.pkl, modescan_summary.json
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE, TIMEFACTOR  # noqa: E402
from diffgle.integrators import BAOAB  # noqa: E402
from diffgle.observables import vacf  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from diffgle.thermostats import GLE_TS_DampedOsc  # noqa: E402
from experiments.train_osc import init_thermostat  # noqa: E402

OUT = os.path.join(ROOT, "results", "bulk")
os.makedirs(OUT, exist_ok=True)
SYSTEMS = ["H2O", "CO2"]
MODES = [1, 2, 3, 4, 6, 8]
PS2 = (1000 / TIMEFACTOR) ** 2  # internal time unit^-2 -> ps^-2
T_FS = np.arange(601)  # 0..600 fs


def out_dir(system, M, top_of=None):
    d = os.path.join(RESULT_DIR, f"modescan_{system}_M{M}")
    return d if top_of is None else os.path.join(d, f"init_top{M}_of_M{top_of}")


def init_top_modes(system, M, K):
    """Supplementary init: the M largest-amplitude modes of the M=K least-squares fit (from that run's record)."""
    rec = pickle.load(open(os.path.join(out_dir(system, K), "record.pkl"), "rb"))
    a, g1, g2, w = rec["init_params"]
    idx = np.sort(np.argsort(-a)[:M])
    th = GLE_TS_DampedOsc(*(torch.tensor(x[idx]) for x in (a, g1, g2, w)))
    kernel = torch.as_tensor(pickle.load(open(os.path.join(RESULT_DIR, system, "plot_dict.pkl"), "rb"))["memory_kernel"],
                             dtype=torch.float64)
    with torch.no_grad():  # same objective as diffgle.thermostats.damped_osc._fit_once
        W = th.double().discrete_kernel(kernel.numel(), DT)
        loss = ((W - kernel) / kernel.abs().max()).pow(2).mean() + 1e-2 * ((W.sum() - kernel.sum()) / kernel.sum()).pow(2)
    print(f"init from modes {idx.tolist()} of the M={K} fit; fit-objective loss {loss.item():.3e}")
    return th.float(), float(loss)


def mode_shares(a, g1, g2, w):
    amp = a / a.sum()
    integ = a * g2 / (g1 * g2 + w ** 2)  # int_0^inf a [exp(-A t)]_11 dt = a [A^-1]_11
    return amp, integ / integ.sum(), integ


def production(s, state_dict, seed, dev):
    """Protocol of experiments/bulk/eval_production.py (osc models): fresh thermostat state, BAOAB."""
    torch.manual_seed(seed)
    M = state_dict["raw_a"].numel()
    th = GLE_TS_DampedOsc(*(torch.ones(M) for _ in range(4))).to(dev)
    th.load_state_dict(state_dict)
    integ = BAOAB(s.potential, th, s.mass, DT, TEMPERATURE)
    with torch.no_grad():
        q, vs, _ = integ.simulate(s.q0, s.v0, 5000)
        v = vs[-1]
        # 10000 production steps in chunks (same RNG sequence), velocities kept on the CPU to bound GPU memory
        chunks, temps = [], []
        for _ in range(10):
            q, vs, t = integ.simulate(q, v, 1000)
            v = vs[-1]
            chunks.append(vs[1:].cpu())
            temps += t
        c = vacf(torch.cat(chunks).to(dev), 1000, stride=10)
    aa = s.targets.vacf.float()
    loss = (c - aa).pow(2).sum().item()
    return dict(seed=seed, vacf=c.cpu().numpy(), loss=loss, rmse=float(np.sqrt(loss / len(aa))),
                T_mean=float(np.mean(temps)), T_std=float(np.std(temps)))


def train(system, M, iterations, device, seeds, top_of=None):
    d = out_dir(system, M, top_of)
    os.makedirs(d, exist_ok=True)
    args = argparse.Namespace(system=system, init_kernel=None, n_modes=M, lr=1e-2, window=1200, equil=3000,
                              iterations=iterations, device=device, init_top_of=top_of)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    t_start = time.time()
    torch.manual_seed(0)
    s = load_system(system, device)
    vacf_aa = s.targets.vacf.float()
    td = len(vacf_aa)

    init_mode, fit_loss, fit_error = "fit", None, None
    if top_of is not None:
        th, fit_loss = init_top_modes(system, M, top_of)
        init_mode = f"top{M}_of_M{top_of}_fit"
    else:
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                th, ref_kernel = init_thermostat(args)
            print(buf.getvalue(), end="")
            for line in buf.getvalue().splitlines():
                if "best fit loss" in line:
                    fit_loss = float(line.split()[-1])
            ok = all(torch.isfinite(p).all() for p in th.params()) and fit_loss is not None and np.isfinite(fit_loss)
            if not ok:
                raise RuntimeError(f"non-finite fit (loss {fit_loss})")
        except Exception as e:  # fall back to generic init
            fit_error = repr(e)
            print(f"fit failed for M={M}: {fit_error}; generic init")
            init_mode = "generic"
            args.init_kernel = "none"
            torch.manual_seed(0)
            th, ref_kernel = init_thermostat(args)
    th = th.to(device)
    integ = BAOAB(s.potential, th, s.mass, DT, TEMPERATURE)

    def params():
        return [x.detach().cpu().numpy().astype(np.float64).copy() for x in th.params()]

    init_params = params()
    with torch.no_grad():
        kappa_init = th.continuous_kernel(torch.tensor(T_FS / TIMEFACTOR, device=device, dtype=torch.float)).cpu().numpy() * PS2

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        q, vs, _ = integ.simulate(s.q0, s.v0, args.equil)
    v = vs[-1]
    opt = torch.optim.Adam(th.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iterations, eta_min=args.lr * 0.05)
    hist, temps_hist, t_iter = [], [], []
    for it in range(args.iterations):
        torch.cuda.synchronize()
        t0 = time.time()
        th.detach_state()
        q, vs, temps = integ.simulate(q, v.detach(), args.window)
        v = vs[-1]
        c = vacf(vs, td)
        loss = (c - vacf_aa).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(th.parameters(), 1.0)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        t_iter.append(time.time() - t0)
        hist.append(loss.item())
        temps_hist.append(float(np.mean(temps)))
        if it % 10 == 0 or it == args.iterations - 1:
            print(f"[{system} M={M}] {it:4d} loss {loss.item():.4e} <T> {np.mean(temps):6.1f} {t_iter[-1]:.2f}s",
                  flush=True)
    peak_mem = torch.cuda.max_memory_allocated(device)
    sd = {k: v_.detach().cpu() for k, v_ in th.state_dict().items()}
    torch.save(sd, os.path.join(d, "thermostat_state_dict.pth"))

    a, g1, g2, w = params()
    with torch.no_grad():
        kappa = th.continuous_kernel(torch.tensor(T_FS / TIMEFACTOR, device=device, dtype=torch.float)).cpu().numpy() * PS2
    amp, integ_share, integ_abs = mode_shares(a, g1, g2, w)

    rec = dict(
        system=system, M=M, n_params=4 * M, init_mode=init_mode, fit_loss=fit_loss, fit_error=fit_error,
        init_params=init_params, kappa_init_ps2=kappa_init, final_params=dict(a=a, gamma1=g1, gamma2=g2, omega=w),
        loss_history=np.array(hist), T_train_history=np.array(temps_hist),
        t_fs=T_FS, kappa_ps2=kappa, kappa0_ps2=float(kappa[0]), kappa_integral_ps1=float(integ_abs.sum() * 1000 / TIMEFACTOR),
        amp_share=amp, integral_share=integ_share,
        n_eff_modes=int(np.sum((amp > 0.05) | (integ_share > 0.05))),
        peak_mem_MB=peak_mem / 2 ** 20, s_per_iter=float(np.mean(t_iter)), s_per_iter_median=float(np.median(t_iter)),
        vacf_aa=vacf_aa.cpu().numpy(), train_wall_s=time.time() - t_start,
    )
    with open(os.path.join(d, "train_record.pkl"), "wb") as f:
        pickle.dump(rec, f)
    finish(system, M, device, seeds, s, top_of)


def finish(system, M, device, seeds, s=None, top_of=None):
    """Production runs for a trained model (also usable on its own after a crash)."""
    d = out_dir(system, M, top_of)
    t_start = time.time()
    rec = pickle.load(open(os.path.join(d, "train_record.pkl"), "rb"))
    if s is None:
        s = load_system(system, device)
    sd = torch.load(os.path.join(d, "thermostat_state_dict.pth"), map_location=device)
    prod = [production(s, sd, seed, device) for seed in seeds]
    rmse = [p["rmse"] for p in prod]
    Tm = [p["T_mean"] for p in prod]
    rec.update(production=prod, rmse_mean=float(np.mean(rmse)), rmse_std=float(np.std(rmse)),
               T_mean=float(np.mean(Tm)), T_std=float(np.std(Tm)),
               wall_s=rec["train_wall_s"] + time.time() - t_start)
    with open(os.path.join(d, "record.pkl"), "wb") as f:
        pickle.dump(rec, f)
    print(f"[{system} M={M}] done: fit {rec['fit_loss']} ({rec['init_mode']}), RMSE {rec['rmse_mean']:.3e} +- {rec['rmse_std']:.1e}, "
          f"T {rec['T_mean']:.1f}, {rec['s_per_iter']:.2f} s/it, {rec['peak_mem_MB']:.0f} MB, "
          f"eff modes {rec['n_eff_modes']}, wall {rec['wall_s']:.0f}s", flush=True)


def collect():
    summary = {}
    for system in SYSTEMS:
        recs = {}
        for M in MODES:
            p = os.path.join(out_dir(system, M), "record.pkl")
            if os.path.exists(p):
                recs[M] = pickle.load(open(p, "rb"))
            for K in MODES:  # supplementary runs (--init-top-of K), keyed e.g. "2_top2_of_M4"
                p = os.path.join(out_dir(system, M, K), "record.pkl")
                if os.path.exists(p):
                    recs[f"{M}_top{M}_of_M{K}"] = pickle.load(open(p, "rb"))
        with open(os.path.join(OUT, f"modescan_{system}.pkl"), "wb") as f:
            pickle.dump(recs, f)
        summary[system] = {
            str(M): dict(n_params=r["n_params"], init_mode=r["init_mode"], fit_loss=r["fit_loss"],
                         rmse_mean=r["rmse_mean"], rmse_std=r["rmse_std"], rmse_seeds=[p["rmse"] for p in r["production"]],
                         T_mean=r["T_mean"], T_std=r["T_std"], s_per_iter=r["s_per_iter"], peak_mem_MB=r["peak_mem_MB"],
                         n_eff_modes=r["n_eff_modes"], amp_share=[float(x) for x in r["amp_share"]],
                         integral_share=[float(x) for x in r["integral_share"]], kappa0_ps2=r["kappa0_ps2"],
                         loss_first=float(r["loss_history"][0]), loss_last10_mean=float(np.mean(r["loss_history"][-10:])),
                         wall_s=r["wall_s"])
            for M, r in recs.items()}
    with open(os.path.join(OUT, "modescan_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for system, rows in summary.items():
        print(system)
        for M, r in rows.items():
            print(f"  M={M:>2} P={r['n_params']:2d} {r['init_mode']:14s} fit {r['fit_loss']} RMSE {r['rmse_mean']:.2e}+-{r['rmse_std']:.1e} "
                  f"T {r['T_mean']:.1f} {r['s_per_iter']:.2f}s/it {r['peak_mem_MB']:.0f}MB eff {r['n_eff_modes']} "
                  f"amp {np.round(r['amp_share'], 3)} int {np.round(r['integral_share'], 3)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "production", "collect"])
    p.add_argument("--system", "-s", choices=SYSTEMS, default="H2O")
    p.add_argument("--modes", "-m", type=int, default=4)
    p.add_argument("--iterations", "-n", type=int, default=150)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--init-top-of", type=int, default=None,
                   help="supplementary: init from the M largest modes of the M=K fit, output in a subfolder")
    a = p.parse_args()
    if a.cmd == "train":
        train(a.system, a.modes, a.iterations, a.device, a.seeds, a.init_top_of)
    elif a.cmd == "production":
        finish(a.system, a.modes, a.device, a.seeds, top_of=a.init_top_of)
    else:
        collect()


if __name__ == "__main__":
    main()
