"""Damped-oscillator memory kernel (diffgle/thermostats/damped_osc.py), trained with BPTT.

kappa(t) = sum_k a_k [exp(-A_k t)]_11 with 4 parameters per mode; negative lobes allowed,
positive definite (exact FDT) by construction. Per iteration: a window of --window steps is
integrated with plain autograd and loss = || VACF_CG - VACF_AA ||^2 is backpropagated
through the whole window (the CG force is detached, so the graph only holds v and the
aux variables: ~0.6 MB per step for H2O, independent of the memory length, vs ~26 MB per step
for the paper's 1000-tap filter; see experiments/bulk/measure_memory.py).

    python -m experiments.train_osc -s H2O                    # -> Result/H2O_osc_baoab
    python -m experiments.train_osc -s H2O --integrator vv    # legacy VV+Euler -> Result/H2O_osc
    python -m experiments.train_osc -s CO2 --n-modes 4

Initialisation: least-squares fit of the modes to the kernel in Result/<system>/plot_dict.pkl
(output of main.py) if present, otherwise generic modes spread over 20 fs - 1 ps.
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE, TIMEFACTOR  # noqa: E402
from diffgle.integrators import INTEGRATORS  # noqa: E402
from diffgle.observables import vacf  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from diffgle.thermostats import GLE_TS_DampedOsc, fit_damped_osc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="DiffGLE with damped-oscillator memory kernel (BPTT)")
    p.add_argument("--system", "-s", choices=["H2O", "CO2"], default="H2O")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", "-n", type=int, default=200)
    p.add_argument("--n-modes", "-m", type=int, default=4)
    p.add_argument("--init-kernel", default=None,
                   help="plot_dict.pkl with a 'memory_kernel' to fit for initialisation "
                        "(default: Result/<system>/plot_dict.pkl if present, 'none' to skip)")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--window", type=int, default=1200, help="BPTT window length (steps), >= VACF length")
    p.add_argument("--equil", type=int, default=3000, help="equilibration steps before training")
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument("--integrator", choices=list(INTEGRATORS), default="baoab")
    p.add_argument("--output-dir", default=None,
                   help="default: Result/<system>_osc_baoab (baoab) or Result/<system>_osc (vv)")
    return p.parse_args()


def init_thermostat(args):
    path = args.init_kernel or os.path.join(RESULT_DIR, args.system, "plot_dict.pkl")
    if args.init_kernel != "none" and os.path.exists(path):
        kernel = pickle.load(open(path, "rb"))["memory_kernel"]
        print(f"Fitting {args.n_modes} modes to {path}")
        a, g1, g2, w, loss = fit_damped_osc(kernel, DT, args.n_modes)
        print(f"  best fit loss {loss:.3e}")
        return GLE_TS_DampedOsc(a, g1, g2, w), kernel
    print("No reference kernel; generic initialisation")
    tau = torch.logspace(np.log10(20 / TIMEFACTOR), np.log10(1000 / TIMEFACTOR), args.n_modes)
    gamma = 1 / tau
    omega = torch.linspace(0, 2 * np.pi / (200 / TIMEFACTOR), args.n_modes)
    a = torch.full((args.n_modes,), 1.0 / args.n_modes) * gamma
    return GLE_TS_DampedOsc(a, gamma, gamma, omega), None


def plot_iteration(path, vacf_aa, vacf_cg, W, ref_kernel, history):
    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    ax[0].plot(vacf_aa.cpu(), "k--", label="AA")
    ax[0].plot(vacf_cg.detach().cpu(), "r", label="GLE-osc")
    ax[0].set_ylabel("VACF")
    ax[0].legend()
    if ref_kernel is not None:
        ax[1].plot(ref_kernel, "k--", label="init reference")
    ax[1].plot(W, "r", label="kernel")
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].set_ylabel("memory kernel")
    ax[1].legend()
    ax[2].semilogy(history)
    ax[2].set_ylabel("loss")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args):
    torch.manual_seed(0)
    suffix = "_osc_baoab" if args.integrator == "baoab" else "_osc"
    out_dir = args.output_dir or os.path.join(RESULT_DIR, f"{args.system}{suffix}")
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    s = load_system(args.system, args.device)
    vacf_aa = s.targets.vacf.float()
    td_max = len(vacf_aa)
    assert args.window >= td_max

    thermostat, ref_kernel = init_thermostat(args)
    thermostat = thermostat.to(args.device)
    integrator = INTEGRATORS[args.integrator](s.potential, thermostat, s.mass, DT, TEMPERATURE)
    print(f"Integrator: {args.integrator}")

    with torch.no_grad():
        q, vs, temps = integrator.simulate(s.q0, s.v0, args.equil)
    v = vs[-1]
    print(f"Equilibration: <T> over last half = {np.mean(temps[len(temps) // 2:]):.1f} K")

    optimizer = torch.optim.Adam(thermostat.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations, eta_min=args.lr * 0.05)
    history = []

    for it in range(args.iterations):
        thermostat.detach_state()
        q, vs, temps = integrator.simulate(q, v.detach(), args.window)
        v = vs[-1]
        vacf_cg = vacf(vs, td_max)
        loss = (vacf_cg - vacf_aa).pow(2).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(thermostat.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history.append(loss.item())
        a, g1, g2, w = (p.detach().cpu().numpy() for p in thermostat.params())
        print(f"{it:4d} loss {loss.item():.4e}  <T> {np.mean(temps):6.1f}  "
              f"a {np.round(a, 3)} g1 {np.round(g1, 3)} g2 {np.round(g2, 3)} w {np.round(w, 3)}")
        if it % args.plot_every == 0 or it == args.iterations - 1:
            with torch.no_grad():
                W = thermostat.discrete_kernel(td_max, DT).cpu().numpy()
            plot_iteration(os.path.join(out_dir, "frames", f"optimization_iter_{it:03d}.png"),
                           vacf_aa, vacf_cg, W, ref_kernel, history)

    torch.save(thermostat.state_dict(), os.path.join(out_dir, "thermostat_state_dict.pth"))

    # production run
    thermostat.detach_state()
    with torch.no_grad():
        q, vs, temps = integrator.simulate(q, v.detach(), 10000)
        vacf_prod = vacf(vs, td_max)
        W = thermostat.discrete_kernel(td_max, DT).cpu().numpy()
        kappa = thermostat.continuous_kernel(torch.arange(td_max, device=args.device) * DT).cpu().numpy()
    print(f"Production: <T> = {np.mean(temps):.1f} K, VACF loss = {(vacf_prod - vacf_aa).pow(2).sum().item():.4e}")

    plt.figure()
    plt.plot(vacf_aa.cpu(), "k--", label="AA")
    plt.plot(vacf_prod.cpu(), "r", label="GLE-osc")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "vacf_production.png"), bbox_inches="tight")
    plt.close()

    a, g1, g2, w = (p.detach().cpu().numpy() for p in thermostat.params())
    plot_dict = {
        "VACF_GLE": vacf_prod.cpu().numpy(), "VACF_AA": vacf_aa.cpu().numpy(),
        "memory_kernel": W, "kappa": kappa,
        "a": a, "gamma1": g1, "gamma2": g2, "omega": w,
        "loss_history": np.array(history),
    }
    with open(os.path.join(out_dir, "plot_dict.pkl"), "wb") as f:
        pickle.dump(plot_dict, f)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    run(parse_args())
