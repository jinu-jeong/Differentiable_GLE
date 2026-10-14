"""GLE-BAOAB samples a harmonic well correctly with a strongly oscillating kernel.

Exact values (discrete Lyapunov, experiments/check_integrator_bias.py): <q^2> w^2 m/kT = 1 and
<v^2> m/kT = 1 - (w dt / 2)^2 for BAOAB. The legacy VV+Euler scheme overheats this 60 fs
mode by ~40% with the trained H2O kernel.
"""

import numpy as np
import torch

from diffgle.constants import DT, TIMEFACTOR
from diffgle.integrators import BAOAB
from diffgle.thermostats import GLE_TS_DampedOsc


def test_baoab_harmonic_well():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, m, T = 20000, 18.0, 300.0
    w = 2 * np.pi / (60 / TIMEFACTOR)  # 60 fs period
    # H2O kernel shape: slow exponential, two oscillating modes, one fast damped mode
    th = GLE_TS_DampedOsc(a=[0.128, 0.029, 1.05, 0.112], gamma1=[0.166, 0.031, 0.637, 10.5],
                          gamma2=[0.354, 0.117, 1.05, 23.2], omega=[0.0, 1.86, 1.78, 14.3]).to(device)
    kTm = th.BOLTZMAN * T / m
    mass = torch.full((N, 1), m, device=device)
    integ = BAOAB(lambda q: -m * w ** 2 * q, th, mass, DT, T)
    q = torch.randn(N, 3, device=device) * np.sqrt(kTm) / w
    v = torch.randn(N, 3, device=device) * np.sqrt(kTm)
    acc_q, acc_v, n = 0.0, 0.0, 0
    with torch.no_grad():
        th.prepare(DT)
        for i in range(4000):
            q, v = integ.step(q, v)
            if i >= 1000 and i % 5 == 0:  # first 1000 steps: equilibrate the aux variables
                acc_q += q.pow(2).mean().item()
                acc_v += v.pow(2).mean().item()
                n += 1
    q_ratio = acc_q / n * w ** 2 / kTm
    v_ratio = acc_v / n / kTm
    assert abs(q_ratio - 1.0) < 0.015, q_ratio
    assert abs(v_ratio - (1 - (w * DT / 2) ** 2)) < 0.015, v_ratio


if __name__ == "__main__":
    test_baoab_harmonic_well()
    print("ok")
