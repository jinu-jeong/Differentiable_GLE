"""Free-filter (non-Markovian) GLE thermostats.

GLE_TS  : the paper's thermostat, integrated with utility-style VV+Euler (diffgle.integrators.VVEulerRHS)
          and trained with odeint_adjoint (main.py).
ConvGLE : same kernel / noise convention, flattened DOF and explicit buffers, with a hand-written
          discrete adjoint that gives the exact BPTT gradient without storing the sliding
          history per step (experiments/ablation_conv_bptt.py). Plain autograd would store an
          L-step velocity window per step (~26 MB/step for H2O, L = 1000). The hand derivation
          relies on the friction being linear in v and the force being detached; the Markovian
          damped-osc kernel needs neither (thermostats/damped_osc.py).

Gradient content differs between the two unless ConvGLE(paper_detach=True):
    GLE_TS  detaches one factor of h*h in the kernel, the noise path (h in R_n) and the velocity
            history (v_list), and its adjoint integrates backwards with flipped buffers.
    ConvGLE differentiates both kernel factors, the noise path and the full memory recursion.
    paper_detach=True keeps ConvGLE's exact recursion but uses GLE_TS's detach pattern for h,
    so an ablation can separate "exact gradient through time" from "which paths are detached".
"""

from contextlib import contextmanager

import numpy as np
import torch

from diffgle.constants import BOLTZMAN


class GLE_TS(torch.nn.Module):
    """Paper thermostat: free FIR filter h (filter_length taps), kernel K = c * (h * h).

    Colored noise R_n = sum_j h_j w_{n-j} and memory kernel K_j = RACF_j m dt / (kB T)
    (RACF = autocorrelation of h) satisfy the FDT by construction. forward() returns
    dv/dt = -sum_j K_j v_{n-j} + R_n from rolling velocity / white-noise buffers.
    trapezoidal=True halves the end taps of K (used for the confined systems).
    adjoint_backward() flips the buffers for odeint_adjoint's reverse pass (approximate).

    Note (paper behaviour, kept for reproducibility): the buffers are not rewound between
    training windows. When main.py restarts the next window from frame filter_length + 1
    (restart="after_filter"), v_list / w_list still hold the end of the previous window,
    reversed once more by adjoint_backward.
    """

    def __init__(self, gamma_uniform=0.001, filter_length=None, BOLTZMAN=BOLTZMAN, trapezoidal=False):
        super(GLE_TS, self).__init__()
        self.BOLTZMAN = BOLTZMAN
        self.trapezoidal = trapezoidal
        self.filter_length = filter_length
        dummy = torch.full((self.filter_length,), gamma_uniform)
        self.h = torch.nn.Parameter(dummy[None, None, :])
        self.v_list = None
        self.w_list = None
        self.w_history = []

    def construct_memory(self, T, mass, dt):
        h_padded = torch.nn.functional.pad(self.h, (0, self.h.size(2) - 1)).detach()
        self.theoretical_RACF = torch.nn.functional.conv1d(h_padded, self.h)
        self.memory_kernel = self.theoretical_RACF * mass[0, 0].item() * dt / (self.BOLTZMAN * T)
        self.memory_kernel_trapezoidal = self.memory_kernel.clone()
        if self.trapezoidal:
            self.memory_kernel_trapezoidal[..., 0] *= 0.5
            self.memory_kernel_trapezoidal[..., -1] *= 0.5

    def get_v_and_sample_w(self, v, dt):
        if self.v_list is None:
            n = len(v.flatten()) // 3
            self.v_list = torch.zeros(n * 3, 1, self.filter_length * 2 + 1, device=v.device)
            self.w_list = torch.zeros(n * 3, 1, self.filter_length * 2 + 1, device=v.device)

        self.v_list = torch.roll(self.v_list, -1, dims=2)
        self.v_list[:, :, -1] = v.view(-1, 1).clone().detach()
        self.w_list = torch.roll(self.w_list, -1, dims=2)
        self.w_list[:, :, -1] = torch.randn_like(v, device=v.device).view(-1, 1)

    def forward(self, v, T, dt, mass, t):
        self.construct_memory(T, mass, dt)
        self.get_v_and_sample_w(v, dt)
        kernel = self.memory_kernel_trapezoidal if self.trapezoidal else self.memory_kernel
        friction = -torch.nn.functional.conv1d(self.v_list[:, :, -self.filter_length:], kernel.flip(2))
        random = torch.nn.functional.conv1d(self.w_list[:, :, -self.filter_length:], self.h.flip(2).detach())
        delta_v_langevin = friction.view(v.shape) + random.view(v.shape)
        if len(self.w_history) == self.filter_length * 2:
            self.w_history.pop(0)
        self.w_history.append(random.clone().detach().cpu().numpy().flatten())
        return delta_v_langevin

    @contextmanager
    def adjoint_backward(self):
        try:
            yield
        finally:
            N = self.filter_length
            self.v_list[:, :, -N:] = self.v_list[:, :, -N:].flip(2)
            self.w_list[:, :, -N:] = self.w_list[:, :, -N:].flip(2)

    def noise_autocorrelation(self):
        """Empirical autocorrelation of the generated noise (from w_history) vs the theoretical h*h.

        Returns (empirical, theoretical) as numpy arrays; empirical is NaN until more than
        filter_length + 1 steps have been recorded.
        """
        data = np.array(self.w_history)
        td = self.filter_length + 1
        RACF = [(data[t0:t0 + td] * data[t0]).mean(1) for t0 in range(0, len(data) - td, 10)]
        empirical = np.mean(RACF, axis=0) if RACF else np.full(td, np.nan)
        return empirical, self.theoretical_RACF.detach().cpu().flatten().numpy()


def xcorr(lam, Y, L):
    """G[j] = sum_m sum_d lam[m, d] * Y[m + L - 1 - j, d] for j = 0..L-1 (via FFT over time)."""
    n = lam.shape[0] + Y.shape[0]
    nfft = 1 << (n - 1).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(Y, nfft, dim=0) * torch.fft.rfft(lam, nfft, dim=0).conj(), nfft, dim=0)
    c = c[:L].sum(1)  # c[k] = sum_m lam[m] . Y[m + k]
    return c.flip(0)


class ConvGLE(torch.nn.Module):
    """GLE_TS with the same kernel / noise convention, flattened DOF (D = 3N), explicit history buffers.

    paper_detach=True: detach the first factor of h*h and the noise path like GLE_TS (see module docstring).
    """

    def __init__(self, h, mass, dt, T, paper_detach=False):
        super().__init__()
        self.paper_detach = paper_detach
        self.h = torch.nn.Parameter(h.flatten().clone())
        self.L = self.h.numel()
        self.dt, self.T = dt, T
        self.m = float(mass[0, 0])
        self.v_hist = None  # (L-1, D) velocities before the current window
        self.w_hist = None

    def kernel(self):
        h = self.h
        h_padded = torch.nn.functional.pad(h, (0, self.L - 1))
        if self.paper_detach:
            h_padded = h_padded.detach()  # as GLE_TS.construct_memory
        racf = torch.nn.functional.conv1d(h_padded[None, None], h[None, None]).flatten()
        return racf * self.m * self.dt / (BOLTZMAN * self.T)

    @torch.no_grad()
    def simulate(self, potential, q, v, n_steps):
        """Integrate n_steps with no graph. Returns (q, V, Vall, Wall):
        V (n+1, D) velocities of this window incl. the initial one; Vall (L-1+n, D) the velocities
        each step's friction sees (history + window, last one excluded); Wall (L-1+n, D) the white noise.
        """
        D = v.numel()
        if self.v_hist is None:
            self.v_hist = torch.zeros(self.L - 1, D, device=v.device)
            self.w_hist = torch.zeros(self.L - 1, D, device=v.device)
        K, hf = self.kernel().flip(0), self.h.flip(0)
        Vall = torch.cat((self.v_hist, torch.empty(n_steps + 1, D, device=v.device)))
        Wall = torch.cat((self.w_hist, torch.randn(n_steps, D, device=v.device)))
        Vall[self.L - 1] = v.flatten()
        mass = self.m
        f0 = potential(q)
        for n in range(n_steps):
            vn = Vall[self.L - 1 + n]
            fric = -(K @ Vall[n:n + self.L])
            rand = hf @ Wall[n:n + self.L]
            a0 = f0.flatten() / mass
            q = q + (vn.view_as(q) + 0.5 * a0.view_as(q) * self.dt) * self.dt
            f1 = potential(q)
            accel = 0.5 * (a0 + f1.flatten() / mass)
            Vall[self.L + n] = vn + (accel + fric + rand) * self.dt
            f0 = f1
        self.v_hist = Vall[-(self.L - 1):].clone()
        self.w_hist = Wall[-(self.L - 1):].clone()
        return q, Vall[self.L - 1:], Vall[:-1], Wall

    def backward_adjoint(self, gV, Vall, Wall, memory_in_adjoint=True):
        """gV: dLoss/dv_n (explicit) for n = 0..W. Accumulates dLoss/dh into self.h.grad.

        memory_in_adjoint=False (diagnostic only) drops the friction/memory coupling from the
        adjoint recursion, i.e. lambda_n = gV_n + lambda_{n+1}, as happens when the friction does
        not depend on v in the differentiated graph (GLE_TS stores v detached).
        """
        Wn = gV.shape[0] - 1
        L, dt = self.L, self.dt
        with torch.no_grad():
            K = self.kernel()
            lam = torch.zeros(Wn + 1 + L, gV.shape[1], device=gV.device)
            lam[Wn] = gV[Wn]
            for n in range(Wn - 1, 0, -1):
                # v_{m+1} depends on v_n through identity (m = n) and -dt K_{m-n} (m = n..n+L-1)
                lam[n] = gV[n] + lam[n + 1]
                if memory_in_adjoint:
                    lam[n] -= dt * (K @ lam[n + 1:n + 1 + L])
            lam_next = lam[1:Wn + 1]  # lambda_{m+1}, m = 0..W-1
            gK = -dt * xcorr(lam_next, Vall, L)  # dLoss/dK_j = -dt sum_m lambda_{m+1} . v_{m-j}
            gh_noise = dt * xcorr(lam_next, Wall, L)  # dLoss/dh_j via the noise sum_j h_j w_{m-j}
            if self.paper_detach:
                gh_noise = torch.zeros_like(gh_noise)  # GLE_TS uses h.detach() in the noise
        torch.autograd.backward([self.kernel(), self.h], [gK, gh_noise])
