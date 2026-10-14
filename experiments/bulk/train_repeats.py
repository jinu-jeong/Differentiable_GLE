"""Training repeats of the damped-oscillator GLE model (reproducibility across seeds and inits).

Same settings as experiments/train_osc.py: M = 4 modes, GLE-BAOAB,
3000 equilibration steps, consecutive 1200-step windows, training VACF over 1000 lags with origin
stride 20, loss = sum (VACF_CG - VACF_AA)^2, Adam lr 1e-2 with cosine decay to 5 %, grad-norm clip 1.0.

init "diffgle" : least-squares fit of the modes to Result/<system>/plot_dict.pkl (DiffGLE kernel)
init "generic" : generic modes spread over 20 fs - 1 ps (init_kernel = "none")

Each trained model is evaluated with the production protocol of experiments/bulk/eval_production.py
(start from Data/<system>, 5000 + 10000 BAOAB steps, noise seeds 0 1 2, VACF over 1000 lags with
origin stride 10, kinetic temperature over production).

    cd Differentiable_GLE && python experiments/bulk/train_repeats.py run -s H2O --init diffgle --seed 0
        -> Result/repeats_H2O_diffgle_s0/run_n150.pkl (+ thermostat_n150.pth, config_n150.json)
    cd Differentiable_GLE && python experiments/bulk/train_repeats.py collect
        -> Result/bulk/repeats_H2O.pkl, Result/bulk/repeats_CO2.pkl, Result/bulk/repeats_summary.json
"""

import argparse
import glob
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
from diffgle.observables import temperature, vacf  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from experiments.train_osc import init_thermostat  # noqa: E402

OUT = os.path.join(ROOT, "Result", "bulk")
os.makedirs(OUT, exist_ok=True)
PS2 = (1000 / TIMEFACTOR) ** 2  # internal time^-2 -> ps^-2
T_KERNEL = np.arange(601)  # fs


def params_np(th):
    return {k: x.detach().cpu().numpy().astype(float).copy()
            for k, x in zip(("a", "gamma1", "gamma2", "omega"), th.params())}


def kernel_ps2(th, dev):
    with torch.no_grad():
        t = torch.as_tensor(T_KERNEL, dtype=torch.float32, device=dev) * DT
        return th.continuous_kernel(t).cpu().numpy().astype(float) * PS2


def production(s, th, seed, dev):
    """eval_production.evaluate for an 'osc' model with BAOAB (fresh integrator and aux state)."""
    torch.manual_seed(seed)
    th.reset_state(None)
    th._oc = None
    integ = BAOAB(s.potential, th, s.mass, DT, TEMPERATURE)
    with torch.no_grad():
        q, v = s.q0, s.v0
        for _ in range(5000):
            q, v = integ.step(q, v)
        vs, temps = [], []
        for _ in range(10000):
            q, v = integ.step(q, v)
            vs.append(v.clone())
            temps.append(temperature(v, integ.mass))
        c = vacf(torch.stack(vs), 1000, stride=10)
    aa = s.targets.vacf.float()
    loss = (c - aa).pow(2).sum().item()
    return dict(vacf=c.cpu().numpy(), loss=loss, rmse=float(np.sqrt(loss / len(aa))),
                T_mean=float(np.mean(temps)), T_std=float(np.std(temps)))


def run(a):
    t_start = time.time()
    dev = a.device
    out_dir = os.path.join(RESULT_DIR, f"repeats_{a.system}_{a.init}_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"n{a.iterations}"
    args = argparse.Namespace(system=a.system, init_kernel=None if a.init == "diffgle" else "none", n_modes=4,
                              lr=1e-2, window=1200, equil=3000, iterations=a.iterations, device=dev)
    with open(os.path.join(out_dir, f"config_{tag}.json"), "w") as f:
        json.dump(dict(vars(args), init=a.init, seed=a.seed, noise_seeds=a.noise_seeds), f, indent=1)

    torch.manual_seed(a.seed)
    s = load_system(args.system, dev)
    vacf_aa = s.targets.vacf.float()
    td = len(vacf_aa)
    th, _ = init_thermostat(args)
    th = th.to(dev)
    rec = dict(system=a.system, init=a.init, seed=a.seed, iterations=a.iterations,
               init_params=params_np(th), kappa_init=kernel_ps2(th, dev), t_kernel_fs=T_KERNEL,
               vacf_aa=vacf_aa.cpu().numpy())
    integ = BAOAB(s.potential, th, s.mass, DT, TEMPERATURE)
    with torch.no_grad():
        q, vs, _ = integ.simulate(s.q0, s.v0, args.equil)
    v = vs[-1]
    opt = torch.optim.Adam(th.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iterations, eta_min=args.lr * 0.05)
    hist, temps_hist = [], []
    t_train = time.time()
    for it in range(args.iterations):
        th.detach_state()
        q, vs, temps = integ.simulate(q, v.detach(), args.window)
        v = vs[-1]
        loss = (vacf(vs, td) - vacf_aa).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(th.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(loss.item())
        temps_hist.append(float(np.mean(temps)))
        if it % 10 == 0 or it == args.iterations - 1:
            print(f"[{a.system} {a.init} s{a.seed} {tag}] {it:4d} loss {loss.item():.4e} <T> {np.mean(temps):6.1f}",
                  flush=True)
    rec["train_wall_s"] = time.time() - t_train
    rec["loss_history"] = np.array(hist)
    rec["train_T_history"] = np.array(temps_hist)
    rec["final_params"] = params_np(th)
    rec["kappa_final"] = kernel_ps2(th, dev)
    torch.save(th.state_dict(), os.path.join(out_dir, f"thermostat_{tag}.pth"))

    rec["production"] = {}
    for ns in a.noise_seeds:
        r = production(s, th, ns, dev)
        rec["production"][ns] = r
        print(f"[{a.system} {a.init} s{a.seed} {tag}] production seed {ns}: RMSE {r['rmse']:.3e} T {r['T_mean']:.1f}",
              flush=True)
    rec["rmse_mean"] = float(np.mean([r["rmse"] for r in rec["production"].values()]))
    rec["T_mean"] = float(np.mean([r["T_mean"] for r in rec["production"].values()]))
    rec["wall_s"] = time.time() - t_start
    with open(os.path.join(out_dir, f"run_{tag}.pkl"), "wb") as f:
        pickle.dump(rec, f)
    print(f"[{a.system} {a.init} s{a.seed} {tag}] done in {rec['wall_s']:.0f} s: RMSE {rec['rmse_mean']:.3e} "
          f"T {rec['T_mean']:.1f}", flush=True)


def lobe(k):
    """(time fs, depth / kappa(0)) of the most negative point of kappa, or None if kappa >= 0."""
    i = int(np.argmin(k))
    return (float(T_KERNEL[i]), float(k[i] / k[0])) if k[i] < 0 else (None, 0.0)


def max_dev(ks):
    """max over runs and t of |kappa_i - kappa_mean| / kappa_mean(0)."""
    ks = np.asarray(ks)
    m = ks.mean(0)
    return float(np.abs(ks - m).max() / m[0])


def collect(a):
    summary = {}
    for system in ("H2O", "CO2"):
        runs = {}
        for p in sorted(glob.glob(os.path.join(RESULT_DIR, f"repeats_{system}_*_s*", "run_n*.pkl"))):
            r = pickle.load(open(p, "rb"))
            runs[(r["init"], r["iterations"], r["seed"])] = r
        if not runs:
            continue
        with open(os.path.join(OUT, f"repeats_{system}.pkl"), "wb") as f:
            pickle.dump(runs, f)
        sys_sum = {"runs": {}, "groups": {}, "between_inits": {}}
        groups = {}
        for (init, n, seed), r in sorted(runs.items()):
            lt, ld = lobe(r["kappa_final"])
            sys_sum["runs"][f"{init}_n{n}_s{seed}"] = dict(
                rmse_mean=r["rmse_mean"], T_mean=r["T_mean"],
                rmse_per_noise_seed=[r["production"][k]["rmse"] for k in sorted(r["production"])],
                T_per_noise_seed=[r["production"][k]["T_mean"] for k in sorted(r["production"])],
                kappa0_ps2=float(r["kappa_final"][0]), lobe_min_t_fs=lt, lobe_min_rel=ld,
                integral_kappa_ps1=float(np.trapezoid(r["kappa_final"], dx=1e-3)),
                loss_first=float(r["loss_history"][0]), loss_last10_mean=float(np.mean(r["loss_history"][-10:])),
                train_wall_s=r["train_wall_s"], wall_s=r["wall_s"],
                init_params=_lst(r["init_params"]), final_params=_lst(r["final_params"]))
            groups.setdefault((init, n), []).append(r)
        for (init, n), rs in groups.items():
            rm = [r["rmse_mean"] for r in rs]
            tm = [r["T_mean"] for r in rs]
            sys_sum["groups"][f"{init}_n{n}"] = dict(
                n_seeds=len(rs), rmse_mean=float(np.mean(rm)), rmse_sd=float(np.std(rm, ddof=1)) if len(rs) > 1 else 0.0,
                T_mean=float(np.mean(tm)), T_sd=float(np.std(tm, ddof=1)) if len(rs) > 1 else 0.0,
                kappa_max_rel_dev=max_dev([r["kappa_final"] for r in rs]),
                kappa_mean_ps2=np.mean([r["kappa_final"] for r in rs], 0)[::10].tolist())
        for n in sorted({n for _, n in groups}):
            if ("diffgle", 150) in groups and ("generic", n) in groups:
                kd = np.mean([r["kappa_final"] for r in groups[("diffgle", 150)]], 0)
                kg = np.mean([r["kappa_final"] for r in groups[("generic", n)]], 0)
                sys_sum["between_inits"][f"diffgle_n150_vs_generic_n{n}"] = dict(
                    mean_kernel_max_rel_diff=float(np.abs(kd - kg).max() / kd[0]),
                    all_runs_max_rel_dev=max_dev([r["kappa_final"] for r in groups[("diffgle", 150)] + groups[("generic", n)]]))
        summary[system] = sys_sum
    summary["_note"] = ("kappa in ps^-2 on t = 0..600 fs; max_rel_dev = max_t |kappa_i - kappa_mean| / kappa_mean(0); "
                        "lobe = most negative point of kappa (time, kappa/kappa(0)); rmse/T = mean over noise seeds 0,1,2; "
                        "sd = sample sd over training seeds")
    with open(os.path.join(OUT, "repeats_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v["groups"] for k, v in summary.items() if k != "_note"}, indent=1))


def _lst(d):
    return {k: np.asarray(v).tolist() for k, v in d.items()}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", "-s", choices=["H2O", "CO2"], required=True)
    r.add_argument("--init", choices=["diffgle", "generic"], required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--iterations", "-n", type=int, default=150)
    r.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--device", default="cuda:0")
    sub.add_parser("collect")
    a = p.parse_args()
    run(a) if a.cmd == "run" else collect(a)


if __name__ == "__main__":
    main()
