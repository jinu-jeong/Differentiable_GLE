"""Finite-difference check: which gradient (paper adjoint A or exact E) predicts the real loss change?

On one window of the paper pipeline (H2O) the noise is recorded, then the window is replayed from the
same initial state and buffers with the same noise for h + eps * d, d = +-A/|A| and +-E/|E|.
The central difference [L(h + eps d) - L(h - eps d)] / (2 eps) is the true directional derivative of
the window loss (including the force feedback that both gradients ignore); a correct gradient g
predicts g . d.

    python -m experiments.check_gradient_fd
"""

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from diffgle.constants import DT, RESULT_DIR, TEMPERATURE  # noqa: E402
from diffgle.integrators import VVEulerRHS  # noqa: E402
from diffgle.observables import VACF_computer  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from diffgle.thermostats import ConvGLE  # noqa: E402
from experiments.diagnose_paper_adjoint import WARMUP, WINDOW, RecordingGLE_TS  # noqa: E402
from main import integrate  # noqa: E402


class ReplayGLE_TS(RecordingGLE_TS):
    """Draws its white noise from self.replay (a list) instead of torch.randn when set."""

    replay = None

    def get_v_and_sample_w(self, v, dt):
        if self.replay is None:
            return super().get_v_and_sample_w(v, dt)
        self.v_list = torch.roll(self.v_list, -1, dims=2)
        self.v_list[:, :, -1] = v.view(-1, 1).clone().detach()
        self.w_list = torch.roll(self.w_list, -1, dims=2)
        self.w_list[:, :, -1] = self.replay[self._i].view(-1, 1)
        self._i += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kernel", default=os.path.join(RESULT_DIR, "H2O", "model_state_dict.pth"))
    p.add_argument("--eps", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--out", default=os.path.join(RESULT_DIR, "H2O_adjoint_diagnosis", "fd_check.json"))
    args = p.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    s = load_system("H2O", args.device)
    N, D = s.N, 3 * s.N
    th = ReplayGLE_TS(gamma_uniform=0.001, filter_length=1000).to(args.device)
    L = th.filter_length
    func = VVEulerRHS(s.potential, th, TEMPERATURE, DT, s.mass)
    if args.kernel != "init":
        func.load_state_dict(torch.load(args.kernel, map_location=args.device), strict=True)
    VACF, aa = VACF_computer(1000), s.targets.vacf
    loss_fn = lambda V: (VACF(V) - aa).pow(2).sum().item() if not V.requires_grad else (VACF(V) - aa).pow(2).sum()

    with torch.no_grad():
        y0 = integrate(func, torch.concatenate((s.v0, s.q0)), WARMUP)[-1]
    start = (th.v_list.clone(), th.w_list.clone())
    H_v, H_w = th.v_list[:, 0, -(L - 1):].T.clone(), th.w_list[:, 0, -(L - 1):].T.clone()

    # recorded window + both gradients
    th.recording, th.rec_w, th.rec_out = True, [], []
    y = integrate(func, y0, WINDOW)
    th.recording = False
    noise = [w.clone() for w in th.rec_w]
    V = y[:, :N]
    loss = loss_fn(V)
    Vd = V.detach().reshape(WINDOW, D).requires_grad_(True)
    gV = torch.autograd.grad(loss_fn(Vd.view(WINDOW, N, 3)), Vd)[0]
    conv = ConvGLE(th.h.detach(), s.mass, DT, TEMPERATURE)
    conv.backward_adjoint(gV, torch.cat((H_v, Vd.detach()[:-1])), torch.cat((H_w, torch.stack(noise))))
    gE = conv.h.grad.detach().clone()
    th.h.grad = None
    with th.adjoint_backward():
        loss.backward()
    gA = th.h.grad.detach().flatten().clone()
    h0 = th.h.detach().clone()

    def window_loss(h):
        th.v_list, th.w_list = start[0].clone(), start[1].clone()
        th.replay, th._i = noise, 0
        with torch.no_grad():
            th.h.copy_(h)
            val = loss_fn(integrate(func, y0, WINDOW)[:, :N])
        th.replay = None
        return val

    base = window_loss(h0)
    print(f"recorded loss {loss.item():.6f}, replayed loss {base:.6f} (must match)")
    rows = {"recorded_loss": loss.item(), "replayed_loss": base, "cos_A_E": torch.nn.functional.cosine_similarity(gA, gE, 0).item()}
    for name, g in [("E", gE), ("A", gA)]:
        d = g / g.norm()
        for eps in args.eps:
            step = eps * h0.norm() * d.view_as(h0)
            fd = (window_loss(h0 + step) - window_loss(h0 - step)) / (2 * eps * h0.norm().item())
            pred_E, pred_A = (gE @ d).item(), (gA @ d).item()
            rows[f"dir {name}, eps {eps:g}"] = {"finite_difference": fd, "predicted_by_E": pred_E, "predicted_by_A": pred_A}
            print(f"direction {name}/|{name}|  eps {eps:g}: finite diff {fd:+.4e} | E predicts {pred_E:+.4e} | A predicts {pred_A:+.4e}", flush=True)
    th.h.data.copy_(h0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
