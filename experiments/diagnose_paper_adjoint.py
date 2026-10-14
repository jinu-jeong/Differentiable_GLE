"""Why is the paper's adjoint gradient inaccurate? Compare it with the exact gradient on the same window.

For one training window of the paper pipeline (H2O, GLE_TS + VVEulerRHS + odeint_adjoint, paper
loss with VACF_computer) the noise and thermostat output are recorded during the forward pass.
On exactly that trajectory we compute dLoss/dh in five ways:

    A     paper adjoint (odeint_adjoint with GLE_TS.adjoint_backward), as used in main.py
    E     exact discrete gradient through time, all paths differentiated        (ConvGLE)
    E_pd  exact through time, but GLE_TS's detach pattern (one kernel factor, noise path)
    M_pd  GLE_TS's detach pattern and no friction/memory coupling in the adjoint recursion
          (what the adjoint sees when the friction does not depend on v in the graph)
    M     no memory coupling, all h paths differentiated

Hypotheses:
    (i)   missing friction/memory coupling in the adjoint     -> compare E_pd vs M_pd
    (ii)  wrong history / time indexing in the reverse pass   -> compare A vs M_pd
    (iii) detached paths (kernel factor, noise)               -> compare E vs E_pd
Also: the adjoint's backward pass draws fresh noise -> A with two RNG seeds.

    python -m experiments.diagnose_paper_adjoint --windows 3                        # trained kernel
    python -m experiments.diagnose_paper_adjoint --windows 3 --kernel init --out results/H2O_adjoint_diagnosis_init
    python -m experiments.diagnose_paper_adjoint --windows 3 --loss full --out results/H2O_adjoint_diagnosis_fullloss
"""

import argparse
import json
import os

import numpy as np
import torch

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE
from diffgle.integrators import VVEulerRHS
from diffgle.observables import VACF_computer, vacf
from diffgle.systems import load_system
from diffgle.thermostats import ConvGLE, GLE_TS
from main import integrate

WINDOW = 1010   # frames, as main.py for H2O (max(L, 1000) + 10)
WARMUP = 1200   # frames with no grad before every window: fills the L = 1000 history buffers


class RecordingGLE_TS(GLE_TS):
    """GLE_TS that records, while self.recording, the white noise w_n it draws and its output."""

    recording = False

    def get_v_and_sample_w(self, v, dt):
        super().get_v_and_sample_w(v, dt)
        if self.recording:
            self.rec_w.append(self.w_list[:, 0, -1].detach().clone())

    def forward(self, v, T, dt, mass, t):
        out = super().forward(v, T, dt, mass, t)
        if self.recording:
            self.rec_out.append(out.detach().flatten().clone())
        return out


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kernel", default=os.path.join(RESULT_DIR, "H2O", "model_state_dict.pth"),
                   help="GLE_TS state dict to evaluate the gradient at ('init' = main.py's initial filter)")
    p.add_argument("--loss", choices=["paper", "full"], default="paper",
                   help="paper: VACF_computer (lags detached, main.py); full: vacf() (origins and lags, BPTT experiments)")
    p.add_argument("--out", default=os.path.join(RESULT_DIR, "H2O_adjoint_diagnosis"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    s = load_system("H2O", args.device)
    N, D = s.N, 3 * s.N
    th = RecordingGLE_TS(gamma_uniform=0.001, filter_length=1000).to(args.device)
    L = th.filter_length
    func = VVEulerRHS(s.potential, th, TEMPERATURE, DT, s.mass)
    if args.kernel != "init":
        func.load_state_dict(torch.load(args.kernel, map_location=args.device), strict=True)
    VACF = VACF_computer(1000)
    aa = s.targets.vacf
    if args.loss == "paper":
        loss_fn = lambda V: (VACF(V) - aa).pow(2).sum()  # main.py
    else:
        loss_fn = lambda V: (vacf(V, 1000) - aa).pow(2).sum()  # experiments/train_osc.py, ablation_conv_bptt.py

    y_last = torch.concatenate((s.v0, s.q0))
    rows = []
    for k in range(args.windows):
        # fresh buffers: the previous window's adjoint pass flipped / rolled them
        with torch.no_grad():
            y0 = integrate(func, y_last, WARMUP)[-1]
        H_v = th.v_list[:, 0, -(L - 1):].T.clone()  # (L-1, D) velocities preceding the window
        H_w = th.w_list[:, 0, -(L - 1):].T.clone()

        th.recording, th.rec_w, th.rec_out = True, [], []
        y = integrate(func, y0, WINDOW)
        th.recording = False
        W = len(th.rec_w)
        assert W == WINDOW - 1, W
        V = y[:, :N]
        loss = loss_fn(V)

        # --- exact gradients on the recorded trajectory
        Vd = V.detach().reshape(WINDOW, D).requires_grad_(True)
        gV = torch.autograd.grad(loss_fn(Vd.view(WINDOW, N, 3)), Vd)[0]
        Vall = torch.cat((H_v, Vd.detach()[:-1]))
        Wall = torch.cat((H_w, torch.stack(th.rec_w)))

        # consistency: ConvGLE's recursion reproduces GLE_TS's thermostat output on this trajectory
        conv = ConvGLE(th.h.detach(), s.mass, DT, TEMPERATURE)
        with torch.no_grad():
            K, hf = conv.kernel().flip(0), conv.h.flip(0)
            recon = torch.stack([-(K @ Vall[n:n + L]) + hf @ Wall[n:n + L] for n in range(0, W, 97)])
            ref = torch.stack(th.rec_out[0:W:97])
            consistency = ((recon - ref).abs().max() / ref.abs().max()).item()

        g = {}
        for name, pd, mem in [("E", False, True), ("E_pd", True, True), ("M_pd", True, False), ("M", False, False)]:
            conv = ConvGLE(th.h.detach(), s.mass, DT, TEMPERATURE, paper_detach=pd)
            conv.backward_adjoint(gV, Vall, Wall, memory_in_adjoint=mem)
            g[name] = conv.h.grad.detach().clone()

        # --- paper adjoint, twice with different noise in the backward pass
        saved = (th.v_list.clone(), th.w_list.clone(), list(th.w_history))
        for tag, seed in [("A", 1), ("A_seed2", 2)]:
            th.v_list, th.w_list, th.w_history = saved[0].clone(), saved[1].clone(), list(saved[2])
            torch.manual_seed(seed)
            th.h.grad = None
            with th.adjoint_backward():
                loss.backward(retain_graph=True)
            g[tag] = th.h.grad.detach().flatten().clone()

        row = {
            "window": k, "loss": loss.item(), "recursion_consistency_max_rel_err": consistency,
            "norms": {n: v.norm().item() for n, v in g.items()},
            "cos": {
                "A vs E (paper vs exact)": cos(g["A"], g["E"]),
                "A vs A_seed2 (backward noise)": cos(g["A"], g["A_seed2"]),
                "E_pd vs E  (iii detach)": cos(g["E_pd"], g["E"]),
                "M_pd vs E_pd (i memory in adjoint)": cos(g["M_pd"], g["E_pd"]),
                "A vs M_pd (ii history/indexing)": cos(g["A"], g["M_pd"]),
                "M vs E (i without detach)": cos(g["M"], g["E"]),
            },
            "rel_err": {
                "A vs E": rel(g["A"], g["E"]),
                "E_pd vs E": rel(g["E_pd"], g["E"]),
                "M_pd vs E_pd": rel(g["M_pd"], g["E_pd"]),
                "A vs M_pd": rel(g["A"], g["M_pd"]),
            },
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        torch.save({k2: v.cpu() for k2, v in g.items()}, os.path.join(args.out, f"grads_window{k}.pt"))
        y_last = y[-1].detach()
        del y, V, loss

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("mean cos:", {c: float(np.mean([r["cos"][c] for r in rows])) for c in rows[0]["cos"]})


if __name__ == "__main__":
    main()
