"""Descent test: does a step along -A (paper adjoint) or -E (exact gradient) lower the expected loss?

The window-level pathwise derivative is dominated by the force feedback / chaos that both gradients
ignore (experiments/check_gradient_fd.py), so it cannot decide which gradient is useful. Here the
criterion is the production VACF loss (5000 equilibration + 10000 production frames, same
protocol as experiments/eval_osc.py; seed-to-seed spread ~0.001) after a step
h -> h - eta |h| g/|g|, with g averaged over the windows saved by diagnose_paper_adjoint.

    python -m experiments.check_descent
"""

import argparse
import glob
import json
import os

import torch

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE
from diffgle.integrators import VVEulerRHS
from diffgle.observables import vacf
from diffgle.systems import load_system
from diffgle.thermostats import GLE_TS
from main import integrate


def production_loss(s, h, seed, device):
    torch.manual_seed(seed)
    th = GLE_TS(gamma_uniform=0.001, filter_length=h.numel()).to(device)
    with torch.no_grad():
        th.h.copy_(h.view_as(th.h))
    func = VVEulerRHS(s.potential, th, TEMPERATURE, DT, s.mass)
    with torch.no_grad():
        y0 = integrate(func, torch.concatenate((s.v0, s.q0)), 5000)[-1]
        vs = integrate(func, y0, 10001)[:, :s.N]
        c = vacf(vs, 1000, stride=10)
    return (c - s.targets.vacf).pow(2).sum().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--diagnosis", default=os.path.join(RESULT_DIR, "H2O_adjoint_diagnosis"))
    p.add_argument("--kernel", default=os.path.join(RESULT_DIR, "H2O", "model_state_dict.pth"))
    p.add_argument("--etas", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    s = load_system("H2O", args.device)
    h0 = torch.load(args.kernel, map_location=args.device)["Thermostat.h"].flatten()
    grads = [torch.load(f) for f in sorted(glob.glob(os.path.join(args.diagnosis, "grads_window*.pt")))]
    g = {k: torch.stack([gw[k] for gw in grads]).mean(0).to(args.device) for k in ["A", "E"]}
    print(f"{len(grads)} windows, cos(mean A, mean E) = {torch.nn.functional.cosine_similarity(g['A'], g['E'], 0).item():+.3f}")

    rows = {"base": production_loss(s, h0, args.seed, args.device)}
    print(f"base: {rows['base']:.4f}", flush=True)
    for name in ["A", "E"]:
        d = g[name] / g[name].norm()
        for eta in args.etas:
            key = f"-{name}, eta {eta:g}"
            rows[key] = production_loss(s, h0 - eta * h0.norm() * d, args.seed, args.device)
            print(f"{key}: {rows[key]:.4f}  (change {rows[key] - rows['base']:+.4f})", flush=True)
    with open(os.path.join(args.diagnosis, "descent_check.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
