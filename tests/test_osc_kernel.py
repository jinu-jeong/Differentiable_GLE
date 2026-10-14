"""GLE_TS_DampedOsc: what the thermostat does step by step equals the kernel it reports."""

import numpy as np
import torch

from diffgle.constants import DT
from diffgle.thermostats import GLE_TS_DampedOsc


def make():
    # one damped cosine, one underdamped-oscillator-like, one slow exponential mode
    return GLE_TS_DampedOsc(a=[1.0, 0.1, 0.1], gamma1=[0.7, 0.05, 0.2], gamma2=[1.0, 0.1, 0.4], omega=[1.8, 5.0, 0.0])


def test_impulse_response_equals_discrete_kernel():
    """Noise off, velocity kick at step 0: the friction returned by forward() is -W_j."""
    th = make()
    th.BOLTZMAN = 0
    mass = torch.ones(1, 1)
    out = [-th(torch.zeros(1, 3) + (1.0 if n == 0 else 0.0), 300, DT, mass)[0, 0].item() for n in range(50)]
    W = th.discrete_kernel(50, DT).detach().numpy()
    assert np.allclose(out, W, rtol=1e-4, atol=1e-7)


def test_discrete_kernel_approximates_continuous():
    th = make()
    W = th.discrete_kernel(50, DT).detach().numpy()
    kappa = th.continuous_kernel(torch.arange(50) * DT).detach().numpy()
    assert np.abs(W / DT - kappa).max() / kappa[0] < 0.05


def test_o_step_is_stationary_at_target_temperature():
    """Free particles under the exact O step keep <v^2> = kT/m (discrete FDT)."""
    torch.manual_seed(0)
    th = make()
    N, m, T = 20000, 18.0, 300.0
    kTm = th.BOLTZMAN * T / m
    mass = torch.full((N, 1), m)
    v = torch.randn(N, 3) * np.sqrt(kTm)
    th.reset_state(v)
    th.s = torch.randn_like(th.s) * np.sqrt(kTm)
    th.prepare(DT)
    with torch.no_grad():
        v2 = []
        for i in range(400):
            v = th.O_step(v, T, DT, mass)
            v2.append(v.pow(2).mean().item())
    assert abs(np.mean(v2) / kTm - 1) < 0.01


if __name__ == "__main__":
    test_impulse_response_equals_discrete_kernel()
    test_discrete_kernel_approximates_continuous()
    test_o_step_is_stationary_at_target_temperature()
    print("ok")
