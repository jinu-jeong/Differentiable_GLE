"""DiffGLE paper pipeline: learn a free-filter GLE memory kernel from the all-atom VACF.

    python main.py -s H2O
    python main.py -s CO2
    python main.py -s graphene_water_H80

Stages (all system-specific numbers come from diffgle/systems.py:SYSTEMS):
    1. Markovian Langevin baseline run (diagnostics only)
    2. GLE_TS thermostat: initialise (optionally from a pretrained kernel), check the
       noise autocorrelation, equilibrate 3000 steps
    3. Training: per iteration, short no-grad equilibration, then a window of
       max(filter_length, 1000) + 10 frames integrated with odeint_adjoint (Euler, VV baked in),
       loss = || VACF_CG - VACF_AA ||^2, gradient w.r.t. the filter h via the adjoint
    4. Save model_state_dict.pth / plot_dict.pkl / config.json, 10000-step production run, xyz dump

Outputs go to results/<system>/ (iteration snapshots in frames/).
Lengths are in frames: n frames = initial state + (n - 1) Euler steps of 1 fs.
The damped-osc kernel + BAOAB + BPTT variant lives in experiments/train_osc.py.
"""

import argparse
import dataclasses
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint_adjoint

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE
from diffgle.integrators import VVEulerRHS
from diffgle.io import write_xyz_dump
from diffgle.observables import MSD_computer, RDF_computer, VACF_computer, density_computer
from diffgle.systems import SYSTEMS, load_system
from diffgle.thermostats import GLE_TS, Langevin_TS

WEIGHT_DECAY = 1e-6
GLE_EQUIL_FRAMES = 3000
PRODUCTION_FRAMES = 10000
VACF_LEN = 1000  # training target length (lags of 1 fs)


def parse_args():
    p = argparse.ArgumentParser(description="DiffGLE paper pipeline (free-filter GLE, adjoint training)")
    p.add_argument("--system", "-s", choices=list(SYSTEMS), default="H2O")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", "-n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="default: results/<system>")
    p.add_argument("--pretrained", default=None, help="override the system's pretrained kernel (state dict)")
    p.add_argument("--no-pretrained", action="store_true", help="do not initialise from a pretrained kernel")
    p.add_argument("--plot-every", type=int, default=10, help="save a training snapshot every N iterations")
    p.add_argument("--show", action="store_true", help="also show plots interactively")
    return p.parse_args()


# ---------------------------------------------------------------- helpers

def integrate(func, y0, n_frames):
    """n_frames states (y0 + n_frames - 1 fixed-step Euler steps) with odeint_adjoint.

    Gradients w.r.t. func's parameters come from the adjoint. n_frames = 1 returns y0 only.
    """
    t = torch.tensor(np.arange(0, DT * n_frames, DT)).to(y0.device)
    return odeint_adjoint(func, y0, t, method="euler")


def optimizer_at(schedule, it):
    """(name, lr) of the last schedule entry with start_iter <= it."""
    return [(name, lr) for start, name, lr in schedule if it >= start][-1]


class Figures:
    def __init__(self, out_dir, show):
        self.out_dir, self.show = out_dir, show
        os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    def save(self, name, frame=False):
        plt.savefig(os.path.join(self.out_dir, "frames" if frame else "", name), bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close()

    def lines(self, name, *curves, xlim=None):
        """curves: (x or None, y, style)."""
        plt.figure()
        for x, y, style in curves:
            y = y.detach().cpu() if torch.is_tensor(y) else y
            x = x.detach().cpu() if torch.is_tensor(x) else x
            plt.plot(y, style) if x is None else plt.plot(x, y, style)
        if xlim:
            plt.xlim(xlim)
        self.save(name)


def structure(s, traj_q):
    """RDF (bulk) or water density profile along z (confined) of a position trajectory."""
    if s.cfg.confined:
        return density_computer(s.cell, z_mask=s.mobile, axis=2, device=s.cell.device, dr=0.5)(traj_q)
    return RDF_computer(s.cell, s.cell.device)(traj_q)


def mobile(s, traj):
    return traj if s.mobile is None else traj[:, s.mobile]


# ---------------------------------------------------------------- pipeline

def run(args):
    torch.manual_seed(args.seed)
    cfg = SYSTEMS[args.system]
    out_dir = args.output_dir or os.path.join(RESULT_DIR, args.system)
    fig = Figures(out_dir, args.show)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "system": dataclasses.asdict(cfg)}, f, indent=2)

    s = load_system(args.system, args.device)
    N, T = s.N, TEMPERATURE
    struct_name = "density" if cfg.confined else "rdf"
    struct_aa = s.targets.density if cfg.confined else s.targets.rdf
    MSD, VACF = MSD_computer(cfg.diagnostic_len), VACF_computer(cfg.diagnostic_len)

    # 1. Markovian Langevin baseline -------------------------------------------------
    func = VVEulerRHS(s.potential, Langevin_TS(gamma=cfg.langevin_gamma).to(args.device), T, DT, s.mass, s.frozen)
    v_init = s.v0 * 0 if cfg.baseline_zero_velocity else s.v0
    func(0., torch.concatenate((v_init, s.q0)))  # warm-up call of the paper code (draws one noise sample)
    with torch.no_grad():
        y_base = integrate(func, torch.concatenate((v_init, s.q0)), cfg.baseline_frames)
    fig.lines("temperature_langevin.png", (None, np.array(func.Temperature_log), 'k'))
    x_base, struct_base = structure(s, y_base[::10, N:])
    msd_base, vacf_base = MSD(mobile(s, y_base[:, N:])), VACF(mobile(s, y_base[:, :N]))
    fig.lines(f"{struct_name}_langevin.png", (x_base, struct_base, 'r'), (*struct_aa, 'k--'))
    fig.lines("msd_langevin.png", (None, msd_base, 'r'), (None, s.targets.msd, 'k--'))
    fig.lines("vacf_langevin.png", (None, vacf_base, 'r'), (None, s.targets.vacf, 'k--'))

    # 2. GLE thermostat: init, noise check, equilibration ------------------------------
    thermostat = GLE_TS(gamma_uniform=0.001, filter_length=cfg.filter_length, trapezoidal=cfg.trapezoidal_kernel).to(args.device)
    func = VVEulerRHS(s.potential, thermostat, T, DT, s.mass, s.frozen)
    pretrained = args.pretrained or cfg.pretrained
    if pretrained and not args.no_pretrained:
        if not os.path.exists(pretrained):
            raise FileNotFoundError(f"pretrained kernel {pretrained} not found "
                                    f"(run the source system first, pass --pretrained PATH or --no-pretrained)")
        # strict: a checkpoint without 'Thermostat.h' of the right length is an error, not a silent no-op
        func.load_state_dict(torch.load(pretrained, map_location=args.device), strict=True)
        print("Transfer learning from", pretrained)
    func(0., torch.concatenate((s.v0 * 0, s.q0)))  # fills the thermostat buffers
    racf_empirical, racf_theory = thermostat.noise_autocorrelation()
    fig.lines("filter.png", (None, thermostat.h.detach().cpu().flatten(), 'bo--'))
    fig.lines("memory_kernel.png", (None, thermostat.memory_kernel.detach().cpu().flatten(), 'bo--'))
    fig.lines("racf.png", (None, racf_theory, 'bo--'), (None, racf_empirical, 'ro--'))

    with torch.no_grad():
        y_gle = integrate(func, torch.concatenate((s.v0, s.q0)), GLE_EQUIL_FRAMES)
    y0 = y_gle[-1]
    x_gle, struct_gle = structure(s, y_gle[::10, N:] if cfg.confined else y_gle[-1000::10, N:])
    fig.lines(f"{struct_name}_compare.png", (*struct_aa, 'k--'), (x_base, struct_base, 'b'), (x_gle, struct_gle, 'r'))
    fig.lines("msd_compare.png", (None, msd_base, 'b'), (None, MSD(mobile(s, y_gle[:, N:])), 'r'))
    fig.lines("vacf_compare.png", (None, vacf_base, 'b'), (None, VACF(mobile(s, y_gle[:, :N])), 'r'))

    # 3. Training ------------------------------------------------------------------------
    VACF = VACF_computer(VACF_LEN)
    VACF_AA = s.targets.vacf
    window = max(cfg.filter_length, VACF_LEN) + 10
    opt_cfg = optimizer_at(cfg.optimizer_schedule, 0)
    optimizer = getattr(torch.optim, opt_cfg[0])(func.parameters(), lr=opt_cfg[1], weight_decay=WEIGHT_DECAY)

    for it in range(args.iterations):
        if optimizer_at(cfg.optimizer_schedule, it) != opt_cfg:
            opt_cfg = optimizer_at(cfg.optimizer_schedule, it)
            optimizer = getattr(torch.optim, opt_cfg[0])(func.parameters(), lr=opt_cfg[1], weight_decay=WEIGHT_DECAY)

        with torch.no_grad():
            y0 = integrate(func, y0, cfg.pre_equil_frames)[-1].detach()
        y_curr = integrate(func, y0, window)
        y0 = (y_curr[-1] if cfg.restart == "end" else y_curr[cfg.filter_length + 1]).detach()

        VACF_curr = VACF(mobile(s, y_curr[:, :N]))
        loss = (VACF_curr - VACF_AA).pow(2).sum()
        optimizer.zero_grad()
        with thermostat.adjoint_backward():
            loss.backward(retain_graph=True)
        optimizer.step()
        print(it, loss.item())

        if it % args.plot_every == 0 or it == args.iterations - 1:
            plt.figure(figsize=(10, 8))
            for i, (y, style, label) in enumerate([
                ((VACF_curr, VACF_AA), ('r', 'k--'), "VACF"),
                ((thermostat.h,), ('r',), "filter h"),
                ((thermostat.h.grad,), ('g--',), "dL/dh"),
                ((thermostat.memory_kernel,), ('r',), "memory kernel"),
            ]):
                plt.subplot(4, 1, i + 1)
                for yy, st in zip(y, style):
                    plt.plot(yy.detach().cpu().flatten(), st)
                plt.ylabel(label)
            fig.save(f"optimization_iter_{it:03d}.png", frame=True)

    # 4. Save, production run -----------------------------------------------------------
    model_path = os.path.join(out_dir, "model_state_dict.pth")
    torch.save(func.state_dict(), model_path)
    plot_dict = {
        "r": np.asarray(struct_aa[0].cpu() if torch.is_tensor(struct_aa[0]) else struct_aa[0]),
        "RDF_AA": np.asarray(struct_aa[1].cpu() if torch.is_tensor(struct_aa[1]) else struct_aa[1]),
        "RDF_CG": struct_gle.cpu().numpy(),
        "VACF_GLE": VACF_curr.detach().cpu().numpy(),
        "VACF_AA": VACF_AA.detach().cpu().numpy(),
        "filter": thermostat.h.clone().detach().cpu().flatten().numpy(),
        "filter_gradient": thermostat.h.grad.clone().detach().cpu().flatten().numpy(),
        "memory_kernel": thermostat.memory_kernel.clone().detach().cpu().flatten().numpy(),
    }
    with open(os.path.join(out_dir, "plot_dict.pkl"), "wb") as f:
        pickle.dump(plot_dict, f)

    with torch.no_grad():
        y_prod = integrate(func, y0, PRODUCTION_FRAMES)
    fig.lines("vacf_production.png", (None, VACF(mobile(s, y_prod[:, :N])), 'r'), (None, VACF_AA, 'k--'))
    x_prod, struct_prod = structure(s, y_prod[:, N:] if cfg.confined else y_prod[-1000::10, N:])
    fig.lines(f"{struct_name}_production.png", (*struct_aa, 'k--'), (x_prod, struct_prod, 'r'),
              xlim=[0, 80] if cfg.confined else None)

    xyz_path = os.path.join(out_dir, f"{args.system}.xyz")
    write_xyz_dump(xyz_path, s.types, y_prod[:, N:, :], s.cell)
    print(f"Saved model to {model_path}")
    print(f"Saved trajectory to {xyz_path}")


if __name__ == "__main__":
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")
    run(args)
