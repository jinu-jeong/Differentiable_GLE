"""Production evaluation of trained damped-osc thermostats (H2O), any kernel x any integrator.

Same protocol as experiments/bulk/eval_production.py: 5000 equilibration steps
from Data/H2O, then 10000 production steps; VACF with origin stride 10.

    python -m experiments.eval_osc Result/H2O_osc Result/H2O_osc_baoab --integrators vv baoab
"""

import argparse
import os
import pickle

import numpy as np
import torch

from diffgle.constants import DT, TEMPERATURE
from diffgle.integrators import INTEGRATORS
from diffgle.observables import vacf
from diffgle.systems import load_system
from diffgle.thermostats import GLE_TS_DampedOsc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="result dirs containing thermostat_state_dict.pth")
    p.add_argument("--integrators", nargs="+", default=["vv", "baoab"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--equil", type=int, default=5000)
    p.add_argument("--prod", type=int, default=10000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None, help="pickle with VACFs (default: <first run>/eval.pkl)")
    args = p.parse_args()

    dev = args.device
    s = load_system("H2O", dev)
    pot, mass, q0, v0 = s.potential, s.mass, s.q0, s.v0
    aa = s.targets.vacf.float()

    out = {"AA": aa.cpu().numpy()}
    for run in args.runs:
        sd = torch.load(os.path.join(run, "thermostat_state_dict.pth"), map_location=dev)
        M = sd["raw_a"].numel()
        for name in args.integrators:
            for seed in args.seeds:
                torch.manual_seed(seed)
                th = GLE_TS_DampedOsc(*(torch.ones(M) for _ in range(4))).to(dev)
                th.load_state_dict(sd)
                integ = INTEGRATORS[name](pot, th, mass, DT, TEMPERATURE)
                with torch.no_grad():
                    q, vs, _ = integ.simulate(q0, v0, args.equil)
                    _, vs, temps = integ.simulate(q, vs[-1], args.prod)
                    c = vacf(vs, len(aa), stride=10)
                loss = (c - aa).pow(2).sum().item()
                out[(os.path.basename(run.rstrip("/")), name, seed)] = c.cpu().numpy()
                print(f"kernel {os.path.basename(run.rstrip('/')):14s} integrator {name:5s} seed {seed}: "
                      f"VACF loss {loss:.4f}  <T> {np.mean(temps):.1f} K", flush=True)
    path = args.out or os.path.join(args.runs[0], "eval.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
