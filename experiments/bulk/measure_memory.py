"""Measured peak GPU memory of plain-autograd BPTT through a CG GLE window (H2O, N = 2141).

conv  : the DiffGLE free filter (GLE_TS, L taps) with VV + Euler, used as is under autograd.
osc   : damped-oscillator modes (GLE_TS_DampedOsc, M modes) with GLE-BAOAB.
Forces are detached in both (as in training). For each setting the forward window of W steps
is run with autograd on, a VACF-type loss is backpropagated, and the peak allocated memory
above the pre-window baseline is recorded. Output: Result/bulk/memory_scaling.json

    cd Differentiable_GLE && python experiments/bulk/measure_memory.py
"""

import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from diffgle.constants import DT, TEMPERATURE  # noqa: E402
from diffgle.integrators import BAOAB, VV_GLE  # noqa: E402
from diffgle.observables import vacf  # noqa: E402
from diffgle.systems import load_system  # noqa: E402
from diffgle.thermostats import GLE_TS, GLE_TS_DampedOsc  # noqa: E402

OUT = os.path.join(ROOT, "Result", "bulk", "memory_scaling.json")
DEV = "cuda:0"


class ConvWrap(torch.nn.Module):
    """GLE_TS.forward takes an extra t argument that VV_GLE does not pass."""

    def __init__(self, ts):
        super().__init__()
        self.ts = ts

    def forward(self, v, T, dt, mass):
        return self.ts(v, T, dt, mass, None)


def osc_thermostat(M):
    tau = torch.logspace(0, 1.3, M)  # internal units, values irrelevant for memory
    return GLE_TS_DampedOsc(torch.full((M,), 0.3), 1 / tau, 2 / tau, torch.linspace(0, 2, M))


def measure(kind, size, W, s):
    torch.cuda.empty_cache()
    if kind == "conv":
        ts = GLE_TS(gamma_uniform=1e-3, filter_length=size).to(DEV)
        integ = VV_GLE(s.potential, ConvWrap(ts), s.mass, DT, TEMPERATURE)
        with torch.no_grad():  # fill the history buffer
            q, vs, _ = integ.simulate(s.q0, s.v0, size + 5)
    else:
        th = osc_thermostat(size).to(DEV)
        integ = BAOAB(s.potential, th, s.mass, DT, TEMPERATURE)
        with torch.no_grad():
            q, vs, _ = integ.simulate(s.q0, s.v0, 50)
        th.detach_state()
    v = vs[-1].detach()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        q, vs, _ = integ.simulate(q, v, W)
        loss = vacf(vs, W // 2).pow(2).sum()
        loss.backward()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        peak, status = None, "OOM"
    del integ
    torch.cuda.empty_cache()
    return dict(kind=kind, size=size, W=W, peak_MiB=peak, status=status)


def main():
    s = load_system("H2O", DEV)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    rows = []
    for L in (100, 250, 500, 1000):
        for W in (50, 100, 200):
            r = measure("conv", L, W, s)
            rows.append(r)
            print(r, flush=True)
            if r["status"] == "OOM":
                break
    for M in (2, 4, 8, 16):
        for W in (300, 600, 1200, 2400):
            r = measure("osc", M, W, s)
            rows.append(r)
            print(r, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(0), total_MiB=total, N=s.N, rows=rows), f, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
