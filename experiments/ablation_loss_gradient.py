"""2x2 training ablation on the paper's free filter: loss definition x gradient method.

    loss     paper : VACF_computer (lag velocities detached, gradient only through the origins; main.py)
             full  : vacf()        (gradient through origins and lags; train_osc.py, ablation_conv_bptt.py)
    gradient adjoint : odeint_adjoint with GLE_TS (main.py)
             exact   : exact discrete gradient through time (ConvGLE.backward_adjoint)

Everything else is shared: initial filter results/H2O, window of 1200 frames following each other,
3000 equilibration frames, Adam lr 3e-4 with cosine decay to 5%, grad-norm clip 1.0, 150 iterations.
Motivation: experiments/check_descent.py (descent test at a single kernel).

    python -m experiments.ablation_loss_gradient --loss full --grad exact
    python -m experiments.ablation_loss_gradient --evaluate      # production loss of all finished runs
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE
from diffgle.integrators import VVEulerRHS
from diffgle.observables import VACF_computer, vacf
from diffgle.systems import load_system
from diffgle.thermostats import GLE_TS, ConvGLE
from experiments.check_descent import production_loss
from main import integrate

OUT = os.path.join(RESULT_DIR, "H2O_ablation_loss_gradient")
WINDOW = 1200  # frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=["paper", "full"])
    p.add_argument("--grad", choices=["adjoint", "exact"])
    p.add_argument("--iterations", "-n", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--equil", type=int, default=3000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--evaluate", action="store_true", help="evaluate all finished runs (and the initial filter)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    return p.parse_args()


def train(args):
    torch.manual_seed(0)
    out = os.path.join(OUT, f"{args.loss}_{args.grad}")
    os.makedirs(out, exist_ok=True)
    s = load_system("H2O", args.device)
    N, aa = s.N, s.targets.vacf
    VACF = VACF_computer(1000)
    loss_fn = (lambda V: (VACF(V) - aa).pow(2).sum()) if args.loss == "paper" else (lambda V: (vacf(V, 1000) - aa).pow(2).sum())
    h0 = torch.load(os.path.join(RESULT_DIR, "H2O", "model_state_dict.pth"), map_location=args.device)["Thermostat.h"]

    if args.grad == "adjoint":
        th = GLE_TS(gamma_uniform=0.001, filter_length=1000).to(args.device)
        func = VVEulerRHS(s.potential, th, TEMPERATURE, DT, s.mass)
        func.load_state_dict({"Thermostat.h": h0}, strict=True)
        params = list(func.parameters())
        with torch.no_grad():
            y0 = integrate(func, torch.concatenate((s.v0, s.q0)), args.equil)[-1]
    else:
        model = ConvGLE(h0, s.mass, DT, TEMPERATURE).to(args.device)
        params = list(model.parameters())
        q, V, _, _ = model.simulate(s.potential, s.q0, s.v0, args.equil - 1)
        v = V[-1].view(N, 3)

    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iterations, eta_min=args.lr * 0.05)
    history = []
    for it in range(args.iterations):
        opt.zero_grad()
        if args.grad == "adjoint":
            y = integrate(func, y0, WINDOW)
            loss = loss_fn(y[:, :N])
            with th.adjoint_backward():
                loss.backward()
            y0 = y[-1].detach()
            loss = loss.item()
        else:
            q, V, Vall, Wall = model.simulate(s.potential, q, v, WINDOW - 1)
            v = V[-1].view(N, 3)
            Vg = V.clone().requires_grad_(True)
            loss_t = loss_fn(Vg.view(WINDOW, N, 3))
            gV = torch.autograd.grad(loss_t, Vg)[0]
            model.backward_adjoint(gV, Vall, Wall)
            loss = loss_t.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        history.append(loss)
        print(f"{it:4d} loss {loss:.4e}", flush=True)

    h = (th.h if args.grad == "adjoint" else model.h).detach().view(1, 1, -1).cpu()
    torch.save({"Thermostat.h": h}, os.path.join(out, "model_state_dict.pth"))
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "loss_history": history}, f)
    print(f"Saved to {out}")


def evaluate(args):
    s = load_system("H2O", args.device)
    runs = {"initial (Result/H2O)": os.path.join(RESULT_DIR, "H2O", "model_state_dict.pth")}   # label kept for older outputs
    runs.update({os.path.basename(os.path.dirname(p)): p for p in sorted(glob.glob(os.path.join(OUT, "*", "model_state_dict.pth")))})
    rows = {}
    for name, path in runs.items():
        h = torch.load(path, map_location=args.device)["Thermostat.h"].flatten()
        vals = [production_loss(s, h, seed, args.device) for seed in args.seeds]
        rows[name] = {"seeds": vals, "mean": float(np.mean(vals))}
        print(f"{name:24s} production loss {np.mean(vals):.4f}  (seeds {', '.join(f'{x:.4f}' for x in vals)})", flush=True)
    with open(os.path.join(OUT, "evaluation.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    a = parse_args()
    evaluate(a) if a.evaluate else train(a)
