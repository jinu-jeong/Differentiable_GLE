"""Free filter (GLE_TS convention) trained with the lag-differentiated loss and the exact discrete gradient.

Same procedure as the "full" + "exact" cell of experiments/ablation_loss_gradient.py (which is
hard-coded to H2O), generalised to any system: initial filter results/<system>/model_state_dict.pth,
loss = || vacf(V, 1000) - VACF_AA ||^2 (gradient through origins and lags), exact gradient via
ConvGLE.backward_adjoint, window of 1200 frames following each other, 3000 equilibration frames,
Adam lr 3e-4 with cosine decay to 5%, grad-norm clip 1.0, 150 iterations, torch.manual_seed(0).

    cd Differentiable_GLE && python experiments/bulk/train_filter_exact.py --system CO2
        -> results/CO2_conv_exact/{model_state_dict.pth, history.json}
           results/bulk/CO2_conv_exact_kernel.npy (GLE_TS.construct_memory kernel, flattened)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE  # noqa: E402
from diffgle.observables import vacf  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from diffgle.thermostats import GLE_TS, ConvGLE  # noqa: E402

WINDOW = 1200  # frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", "-s", default="CO2", choices=["H2O", "CO2"])
    p.add_argument("--iterations", "-n", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--equil", type=int, default=3000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None, help="default: results/<system>_conv_exact")
    p.add_argument("--kernel-out", default=None, help="default: results/bulk/<system>_conv_exact_kernel.npy")
    return p.parse_args()


def train(args):
    torch.manual_seed(0)
    out = args.output_dir or os.path.join(RESULT_DIR, f"{args.system}_conv_exact")
    os.makedirs(out, exist_ok=True)
    s = load_system(args.system, args.device)
    N, aa = s.N, s.targets.vacf
    loss_fn = lambda V: (vacf(V, 1000) - aa).pow(2).sum()  # noqa: E731
    h0 = torch.load(os.path.join(RESULT_DIR, args.system, "model_state_dict.pth"), map_location=args.device)["Thermostat.h"]
    print(f"{args.system}: N = {N}, filter length {h0.numel()}", flush=True)

    model = ConvGLE(h0, s.mass, DT, TEMPERATURE).to(args.device)
    params = list(model.parameters())
    t0 = time.time()
    q, V, _, _ = model.simulate(s.potential, s.q0, s.v0, args.equil - 1)
    v = V[-1].view(N, 3)

    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iterations, eta_min=args.lr * 0.05)
    history = []
    for it in range(args.iterations):
        opt.zero_grad()
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
        print(f"{it:4d} loss {loss:.4e}  ({time.time() - t0:.0f} s)", flush=True)

    h = model.h.detach().view(1, 1, -1).cpu()
    torch.save({"Thermostat.h": h}, os.path.join(out, "model_state_dict.pth"))
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "loss_history": history, "wall_time_s": time.time() - t0}, f)

    # per-step kernel in the GLE_TS.construct_memory convention
    th = GLE_TS(gamma_uniform=0.001, filter_length=h.numel())
    with torch.no_grad():
        th.h.copy_(h)
    th.construct_memory(TEMPERATURE, s.mass.cpu(), DT)
    kernel = th.memory_kernel.detach().flatten().numpy()
    kpath = args.kernel_out or os.path.join(ROOT, "results", "bulk", f"{args.system}_conv_exact_kernel.npy")
    os.makedirs(os.path.dirname(kpath), exist_ok=True)
    np.save(kpath, kernel)
    print(f"Saved to {out} and {kpath} (kernel length {kernel.size}); wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    train(parse_args())
