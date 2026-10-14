"""Ablation: original conv-filter GLE (GLE_TS, K = c * h*h) trained with exact BPTT gradients.

Separates the effect of the kernel representation (damped-osc vs free filter)
from the effect of the gradient (adjoint vs BPTT). Everything else follows
experiments/train_osc.py: same VACF loss, window, equilibration, optimizer and grad clipping.

Plain autograd would store a (filter_length x D) history per step (~30 GB for H2O),
so the backward pass is the hand-written discrete adjoint of the recurrence

    v_{n+1} = v_n + dt * (a_n - sum_j K_j v_{n-j} + sum_j h_j w_{n-j})

with the forces a_n constant (Tabulated force is detached). This is exactly the
BPTT gradient; the O(L * W) correlations are done with FFTs.

    python -m experiments.ablation_conv_bptt --lr 3e-4                  # -> Result/H2O_conv_bptt_lr0.0003
    python -m experiments.ablation_conv_bptt --lr 3e-4 --paper-detach   # -> Result/H2O_conv_bptt_paperdetach_lr0.0003

--paper-detach uses the same detached paths as GLE_TS (one kernel factor, noise path), so that
paper-adjoint vs this run isolates the exact-through-time gradient, and this run vs the default
isolates the extra gradient paths (see diffgle/thermostats/conv.py).

The discrete adjoint is checked against autograd in tests/test_conv_adjoint.py.
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffgle.constants import BOLTZMAN, DT, RESULT_DIR, TEMPERATURE  # noqa: E402
from diffgle.observables import vacf  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from diffgle.thermostats import ConvGLE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", "-n", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--window", type=int, default=1200)
    p.add_argument("--equil", type=int, default=3000)
    p.add_argument("--paper-detach", action="store_true", help="detach h like GLE_TS (kernel factor, noise path)")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def run(args):
    torch.manual_seed(0)
    tag = "_paperdetach" if args.paper_detach else ""
    out = args.output_dir or os.path.join(RESULT_DIR, f"H2O_conv_bptt{tag}_lr{args.lr:g}")
    os.makedirs(out, exist_ok=True)
    s = load_system("H2O", args.device)
    N, mass = s.N, s.mass
    vacf_aa = s.targets.vacf.float()
    td = len(vacf_aa)

    # start from the adjoint-trained filter (same starting point the damped-osc fit used)
    h0 = torch.load(os.path.join(RESULT_DIR, "H2O", "model_state_dict.pth"), map_location=args.device)["Thermostat.h"]
    model = ConvGLE(h0, mass, DT, TEMPERATURE, paper_detach=args.paper_detach).to(args.device)
    q, V, _, _ = model.simulate(s.potential, s.q0, s.v0, args.equil)
    v = V[-1].view(N, 3)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iterations, eta_min=args.lr * 0.05)
    history = []
    for it in range(args.iterations):
        q, V, Vall, Wall = model.simulate(s.potential, q, v, args.window)
        v = V[-1].view(N, 3)
        Vg = V.clone().requires_grad_(True)
        loss = (vacf(Vg.view(-1, N, 3), td) - vacf_aa).pow(2).sum()
        gV = torch.autograd.grad(loss, Vg)[0]
        opt.zero_grad()
        model.backward_adjoint(gV, Vall, Wall)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        history.append(loss.item())
        T = ((mass.flatten().repeat_interleave(3) * V.pow(2)).sum(1) / (3 * N * BOLTZMAN)).mean().item()
        print(f"{it:4d} loss {loss.item():.4e}  <T> {T:6.1f}", flush=True)

    # state dict compatible with main.py (VVEulerRHS + GLE_TS)
    torch.save({"Thermostat.h": model.h.detach().view(1, 1, -1)}, os.path.join(out, "model_state_dict.pth"))
    with torch.no_grad():
        K = model.kernel().cpu().numpy()
    pickle.dump({"memory_kernel": K, "h": model.h.detach().cpu().numpy(), "loss_history": np.array(history)},
                open(os.path.join(out, "plot_dict.pkl"), "wb"))
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].semilogy(history); ax[0].set_ylabel("loss")
    ax[1].plot(K); ax[1].axhline(0, color="gray", lw=0.5); ax[1].set_ylabel("memory kernel")
    fig.savefig(os.path.join(out, "training.png"), bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    run(parse_args())
