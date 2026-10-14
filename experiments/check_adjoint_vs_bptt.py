"""Damped-osc GLE as a torchsde SDE: stochastic adjoint vs BPTT gradients.

State y = (v, q, s) with s the damped-oscillator aux variables (diffgle/thermostats/damped_osc.py), so
the system is Markovian and the stochastic adjoint (Li et al. 2020) is well
defined. For each window length the same Brownian path is used for

    BPTT    : torchsde.sdeint          + autograd through the solver (reference)
    adjoint : torchsde.sdeint_adjoint  (state reconstructed backwards in time)

on the identical forward discretisation, so any difference is adjoint error.
Solvers: 'euler' (Ito, as in the existing code) and 'reversible_heun'
(Stratonovich; algebraically reversible, Kidger et al. 2021). The noise is
additive, so Ito and Stratonovich describe the same SDE.

    python -m experiments.check_adjoint_vs_bptt --windows 100 300 1200
    python -m experiments.check_adjoint_vs_bptt --methods euler --windows 1200 --strides 1 5 20 100
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torchsde  # pip install --no-deps torchsde trampoline

from diffgle.constants import BOLTZMAN, DT, RESULT_DIR, TEMPERATURE
from diffgle.integrators import VV_GLE
from diffgle.observables import vacf
from diffgle.systems import load_system
from diffgle.thermostats import GLE_TS_DampedOsc


class OscSDE(torch.nn.Module):
    noise_type = "diagonal"

    def __init__(self, thermostat, potential, mass, T, dt, sde_type):
        super().__init__()
        self.sde_type = sde_type
        self.th = thermostat
        self.potential = potential
        self.mass = mass  # (N, 1)
        self.T = T
        self.dt = dt
        self.M = thermostat.n_modes

    def split(self, y):
        z = y.view(-1, 3, 2 + 2 * self.M)
        return z[..., 0], z[..., 1], z[..., 2:].reshape(-1, 3, self.M, 2)

    def join(self, dv, dq, ds):
        return torch.cat((dv[..., None], dq[..., None], ds.flatten(-2)), -1).view(1, -1)

    def f(self, t, y):
        v, q, s = self.split(y)
        c, A = self.coupling(), self.th.drift_matrix()
        # velocity Verlet baked into the drift, as in utility.SDE (force is detached)
        a0 = self.potential(q.detach()) / self.mass
        dq = v + 0.5 * a0 * self.dt
        a1 = self.potential((q + dq * self.dt).detach()) / self.mass
        dv = 0.5 * (a0 + a1) + (c * s).sum((-1, -2))
        ds = -torch.einsum("mij,...mj->...mi", A, s) - c * v[..., None, None]
        return self.join(dv, dq, ds)

    def g(self, t, y):
        _, g1, g2, _ = self.th.params()
        N = self.mass.shape[0]
        amp = torch.sqrt(2 * BOLTZMAN * self.T / self.mass)[..., None, None]  # (N, 1, 1, 1)
        ds = amp * torch.stack((g1, g2), -1).sqrt()  # (N, 1, M, 2)
        ds = ds.expand(N, 3, self.M, 2)
        zero = torch.zeros(N, 3, device=y.device, dtype=y.dtype)
        return self.join(zero, zero, ds)

    def coupling(self):
        a, *_ = self.th.params()
        c = torch.zeros(self.M, 2, device=a.device, dtype=a.dtype)
        c[:, 0] = torch.sqrt(a)
        return c


def setup(device):
    s = load_system("H2O", device)
    th = GLE_TS_DampedOsc(*(torch.ones(4) for _ in range(4))).to(device)
    th.load_state_dict(torch.load(os.path.join(RESULT_DIR, "H2O_osc", "thermostat_state_dict.pth"), map_location=device))
    aa = s.targets.vacf.float()
    # equilibrate with the trained thermostat, keep the aux state
    torch.manual_seed(0)
    integ = VV_GLE(s.potential, th, s.mass, DT, TEMPERATURE)
    with torch.no_grad():
        q, vs, _ = integ.simulate(s.q0, s.v0, 3000)
    v, aux = vs[-1], th.s.clone()
    th.reset_state()
    y0 = torch.cat((v[..., None], q[..., None], aux.flatten(-2)), -1).view(1, -1)
    return th, s.potential, s.mass, aa, y0


def grads(sde, y0, W, method, adjoint, aa, seed, k=1):
    """k: output stride. The adjoint resets its state to the stored forward state at every
    output time, so k is the length (in steps) of each backward reconstruction."""
    dt = sde.dt
    ts = torch.arange(0, W + 1, k, device=y0.device, dtype=y0.dtype) * dt
    bm = torchsde.BrownianInterval(t0=0.0, t1=float(ts[-1]), size=y0.shape, device=y0.device,
                                   dtype=y0.dtype, entropy=seed, levy_area_approximation="none")
    params = list(sde.th.parameters())
    kw = dict(dt=dt, method=method, bm=bm)
    cuda = y0.is_cuda
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() if cuda else 0
    t0 = time.time()
    if adjoint:
        adj = "adjoint_reversible_heun" if method == "reversible_heun" else "euler"
        ys = torchsde.sdeint_adjoint(sde, y0, ts, adjoint_method=adj, adjoint_params=params, **kw)
    else:
        ys = torchsde.sdeint(sde, y0, ts, **kw)
    v = sde.split(ys.view(len(ts), -1))[0].view(len(ts), -1, 3)
    td = min(1000, W // 2) // k
    loss = (vacf(v, td, stride=max(1, 10 // k)) - aa[::k][:td]).pow(2).sum()
    g = torch.autograd.grad(loss, params)
    if cuda:
        torch.cuda.synchronize()
    mem = (torch.cuda.max_memory_allocated() - base) / 2 ** 20 if cuda else float("nan")
    return torch.cat([x.flatten() for x in g]).detach(), loss.item(), time.time() - t0, mem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, nargs="+", default=[100, 300, 1200])
    p.add_argument("--methods", nargs="+", default=["euler", "reversible_heun"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--strides", type=int, nargs="+", default=[1], help="output strides k (backward reconstruction length)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=os.path.join(RESULT_DIR, "H2O_osc", "sde_adjoint_compare.pkl"))
    args = p.parse_args()

    th, pot, mass, aa, y0 = setup(args.device)
    rows = []
    for method in args.methods:
        sde = OscSDE(th, pot, mass, TEMPERATURE, DT, "stratonovich" if method == "reversible_heun" else "ito")
        for W, k, seed in [(W, k, s) for W in args.windows for k in args.strides for s in args.seeds]:
                g_ref, loss_ref, t_ref, m_ref = grads(sde, y0, W, method, False, aa, seed, k)
                g_adj, loss_adj, t_adj, m_adj = grads(sde, y0, W, method, True, aa, seed, k)
                cos = torch.nn.functional.cosine_similarity(g_ref, g_adj, dim=0).item()
                rel = ((g_adj - g_ref).norm() / g_ref.norm()).item()
                row = dict(method=method, W=W, k=k, seed=seed, cos=cos, rel_err=rel, loss_ref=loss_ref, loss_adj=loss_adj,
                           t_bptt=t_ref, t_adj=t_adj, mem_bptt=m_ref, mem_adj=m_adj,
                           g_bptt=g_ref.cpu().numpy(), g_adj=g_adj.cpu().numpy())
                rows.append(row)
                print(f"{method:16s} W={W:5d} k={k:3d} seed={seed}  cos {cos:+.4f}  rel.err {rel:.3e}  "
                      f"|g| bptt {g_ref.norm():.3e} adj {g_adj.norm():.3e}  "
                      f"time {t_ref:5.1f}/{t_adj:5.1f} s  mem {m_ref:6.0f}/{m_adj:6.0f} MiB", flush=True)
    with open(args.out, "wb") as f:
        pickle.dump(rows, f)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
