"""Production evaluation of the H2O models quoted in the paper, under one protocol.

Protocol: start from Data/H2O, 5000 equilibration steps, 10000
production steps (1 fs). VACF over 1000 lags with origins every 10 steps; MSD over 1000 lags
with origins every 100 steps (unwrapped positions); RDF from 20 frames (every 500 steps);
kinetic temperature averaged over production. Writes Result/bulk/production_eval.json
(scalars) and Result/bulk/production_eval_curves.pkl (curves).

    cd Differentiable_GLE && python experiments/bulk/eval_production.py --seeds 0 1 2
    cd Differentiable_GLE && python experiments/bulk/eval_production.py --system CO2
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE  # noqa: E402
from diffgle.integrators import BAOAB, VV_GLE, VVEulerRHS  # noqa: E402
from diffgle.observables import MSD_computer, RDF_computer, temperature, vacf  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from diffgle.thermostats import GLE_TS, GLE_TS_DampedOsc  # noqa: E402
from main import integrate  # noqa: E402

OUT = os.path.join(ROOT, "Result", "bulk")
os.makedirs(OUT, exist_ok=True)
MODELS_CO2 = {
    "conv_diffgle": ("conv", os.path.join(RESULT_DIR, "CO2", "model_state_dict.pth"), "vveuler"),
    "osc4_baoab": ("osc", os.path.join(RESULT_DIR, "CO2_osc_baoab", "thermostat_state_dict.pth"), "baoab"),
    "conv_exact": ("conv", os.path.join(RESULT_DIR, "CO2_conv_exact", "model_state_dict.pth"), "vveuler"),
    "osc4_vv": ("osc", os.path.join(RESULT_DIR, "CO2_osc_vv", "thermostat_state_dict.pth"), "vv"),
}
MODELS = {
    # name: (kind, path, integrator)
    "conv_diffgle": ("conv", os.path.join(RESULT_DIR, "H2O", "model_state_dict.pth"), "vveuler"),
    "conv_exact": ("conv", os.path.join(RESULT_DIR, "H2O_ablation_loss_gradient", "full_exact", "model_state_dict.pth"), "vveuler"),
    "osc4_baoab": ("osc", os.path.join(RESULT_DIR, "H2O_osc_baoab", "thermostat_state_dict.pth"), "baoab"),
    "osc4_vv": ("osc", os.path.join(RESULT_DIR, "H2O_osc", "thermostat_state_dict.pth"), "vv"),
}


def run_conv(s, path, dev):
    """VVEulerRHS + explicit Euler through main.integrate (check_descent protocol)."""
    h = torch.load(path, map_location=dev)["Thermostat.h"].flatten()
    th = GLE_TS(gamma_uniform=0.001, filter_length=h.numel()).to(dev)
    with torch.no_grad():
        th.h.copy_(h.view_as(th.h))
    return VVEulerRHS(s.potential, th, TEMPERATURE, DT, s.mass)


def simulate_integ(integ, q, v, n, record):
    vs, qs, temps = [], [], []
    for i in range(n):
        q, v = integ.step(q, v) if isinstance(integ, BAOAB) else _vv_step(integ, q, v)
        if record:
            vs.append(v.clone())
            qs.append(q.clone())
            temps.append(temperature(v, integ.mass))
    return q, v, vs, qs, temps


def _vv_step(integ, q, v):
    q, vs, _ = integ.simulate(q, v, 1)
    return q, vs[-1]


def evaluate(name, seed, s, dev, models):
    kind, path, integ_name = models[name]
    torch.manual_seed(seed)
    N = s.N
    with torch.no_grad():
        if kind == "conv":
            func = run_conv(s, path, dev)
            y0 = integrate(func, torch.concatenate((s.v0, s.q0)), 5000)[-1]
            n0 = len(func.Temperature_log)
            y = integrate(func, y0, 10001)[1:]
            vs, qs = list(y[:, :N]), list(y[:, N:])
            temps = list(func.Temperature_log)[n0:]
        else:
            sd = torch.load(path, map_location=dev)
            M = sd["raw_a"].numel()
            th = GLE_TS_DampedOsc(*(torch.ones(M) for _ in range(4))).to(dev)
            th.load_state_dict(sd)
            integ = (BAOAB if integ_name == "baoab" else VV_GLE)(s.potential, th, s.mass, DT, TEMPERATURE)
            q, v, _, _, _ = simulate_integ(integ, s.q0, s.v0, 5000, False)
            _, _, vs, qs, temps = simulate_integ(integ, q, v, 10000, True)
        V = torch.stack(vs)
        Q = torch.stack(qs)
        c = vacf(V, 1000, stride=10)
        msd = MSD_computer(1000)(Q)
        r, g = RDF_computer(s.cell, dev)(Q[::500])
    aa = s.targets.vacf.float()
    loss = (c - aa).pow(2).sum().item()
    return dict(
        vacf=c.cpu().numpy(), msd=msd.cpu().numpy(), r=r.cpu().numpy(), rdf=g.cpu().numpy(),
        loss=loss, rmse=float(np.sqrt(loss / len(aa))), T_mean=float(np.mean(temps)),
        T_std=float(np.std(temps)),
    )


def diffusion(msd, t0=500, t1=999):
    """D (A^2/fs) from a linear fit of MSD between t0 and t1 fs: MSD = 6 D t + b."""
    t = np.arange(len(msd), dtype=float)
    return float(np.polyfit(t[t0:t1], msd[t0:t1], 1)[0] / 6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="H2O", choices=["H2O", "CO2"])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    models = MODELS if args.system == "H2O" else MODELS_CO2
    args.models = args.models or list(models)
    s = load_system(args.system, args.device)
    msd_aa = s.targets.msd.cpu().numpy()
    r_aa, g_aa = s.targets.rdf
    D_aa = diffusion(msd_aa)
    curves = {"AA": dict(vacf=s.targets.vacf.cpu().numpy(), msd=msd_aa, r=r_aa, rdf=g_aa)}
    table = {"AA": dict(D_A2_per_fs=D_aa)}
    for name in args.models:
        rows = []
        for seed in args.seeds:
            res = evaluate(name, seed, s, args.device, models)
            D = diffusion(res["msd"])
            g_interp = np.interp(r_aa, res["r"], res["rdf"])
            mask = r_aa < res["r"].max()
            rdf_rmse = float(np.sqrt(np.mean((g_interp[mask] - g_aa[mask]) ** 2)))
            curves[(name, seed)] = res
            row = dict(seed=seed, loss=res["loss"], rmse=res["rmse"], T_mean=res["T_mean"], T_std=res["T_std"],
                       D_A2_per_fs=D, D_rel_err=(D - D_aa) / D_aa, rdf_rmse=rdf_rmse)
            rows.append(row)
            print(name, row, flush=True)
        agg = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
               for k in ("loss", "rmse", "T_mean", "D_A2_per_fs", "D_rel_err", "rdf_rmse")}
        table[name] = dict(path=models[name][1], integrator=models[name][2], seeds=rows, mean_std=agg)
    tag = "" if args.system == "H2O" else "_CO2"
    with open(os.path.join(OUT, f"production_eval{tag}.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(OUT, f"production_eval{tag}_curves.pkl"), "wb") as f:
        pickle.dump(curves, f)
    print("saved")


if __name__ == "__main__":
    main()
