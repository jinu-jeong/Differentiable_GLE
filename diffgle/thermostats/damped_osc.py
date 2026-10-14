"""GLE thermostat with a damped-oscillator (complex-exponential) memory kernel.

Each mode k is a 2D auxiliary OU process (Markovian embedding, Ceriotti-style):

    A_k = [[g1_k,  w_k],      c_k = (sqrt(a_k), 0)
           [-w_k, g2_k]]      g1_k, g2_k > 0,  a_k > 0

    ds_k = (-A_k s_k - c_k v) dt + sqrt(2 kT/m) diag(sqrt(g1_k), sqrt(g2_k)) dW_k
    dv/dt|_GLE = sum_k c_k . s_k

which gives the kernel (per unit mass, simulation time units)

    kappa(t) = sum_k a_k [exp(-A_k t)]_{11}

Special cases of one mode:
    g1 = g2 = g : a e^{-g t} cos(w t)                          (damped cosine)
    g1 -> 0     : a e^{-g t} (cos(W t) + g/W sin(W t)), g=g2/2 (underdamped oscillator, zero slope at t=0)
    w -> 0      : sum of two positive exponentials

The symmetric part of A_k is diag(g1, g2) >= 0, so the stationary aux covariance
is kT/m * I, the FDT holds exactly and kappa is positive definite, while its
time-domain values can go negative. The cost is O(n_modes) per step and
gradients can be taken with plain autograd.

Why a Markovian embedding (beyond the kernel shape): backprop-through-time has to store, per step,
whatever the friction depends on. A free-filter kernel (thermostats/conv.py) depends on the last L
velocities, so autograd stores an L-step window per step (O(N L) memory per step, ~26 MB for H2O,
L = 1000). Here the history is summarised in 2M aux variables, so a step stores O(N (1 + 2M)) numbers
(~0.6 MB for M = 4), independent of how long the memory lasts. This is what makes exact gradients
through time possible with plain autograd (experiments/train_osc.py).

Two ways to integrate it:
    O_step  : exact propagation of the joint (v, s) block, for diffgle.integrators.BAOAB
              (GLE-BAOAB, Ceriotti et al. 2010). Discrete FDT exact, configurational
              sampling exact for harmonic wells. Preferred.
    forward : returns dv/dt for the legacy velocity-Verlet + Euler scheme
              (diffgle.integrators.VVEulerRHS / VV_GLE). The v <-> s coupling is explicit, which
              overheats fast motions (~40% for a 60 fs harmonic mode with the trained
              H2O kernel).
"""

import math

import torch


def _inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.where(x > 20, x, torch.log(torch.expm1(x)))


class GLE_TS_DampedOsc(torch.nn.Module):
    def __init__(self, a, gamma1, gamma2, omega, BOLTZMAN=0.001987191):
        """a, gamma1, gamma2, omega : 1D sequences of length n_modes (simulation units)."""
        super().__init__()
        self.BOLTZMAN = BOLTZMAN
        self.raw_a = torch.nn.Parameter(_inv_softplus(a).float())
        self.raw_gamma1 = torch.nn.Parameter(_inv_softplus(gamma1).float())
        self.raw_gamma2 = torch.nn.Parameter(_inv_softplus(gamma2).float())
        self.omega = torch.nn.Parameter(torch.as_tensor(omega, dtype=torch.float))
        self.s = None
        self._oc = None

    @property
    def n_modes(self):
        return self.raw_a.numel()

    def params(self):
        sp = torch.nn.functional.softplus
        return sp(self.raw_a), sp(self.raw_gamma1), sp(self.raw_gamma2), self.omega

    def drift_matrix(self):
        _, g1, g2, w = self.params()
        return torch.stack((torch.stack((g1, w), -1), torch.stack((-w, g2), -1)), -2)  # (M, 2, 2)

    def step_coefficients(self, dt):
        """Exact one-step propagator P = e^{-A dt}, input gain g = int_0^dt e^{-A u} c du, noise Cholesky."""
        a, g1, g2, _ = self.params()
        A = self.drift_matrix()
        M = A.shape[0]
        c = torch.zeros(M, 2, device=A.device, dtype=A.dtype)
        c[:, 0] = torch.sqrt(a)
        # Van Loan: expm([[-A, c], [0, 0]] dt) = [[P, g], [0, 1]]
        aug = torch.zeros(M, 3, 3, device=A.device, dtype=A.dtype)
        aug[:, :2, :2] = -A * dt
        aug[:, :2, 2] = c * dt
        E = torch.linalg.matrix_exp(aug)
        P, g = E[:, :2, :2], E[:, :2, 2]
        # stationary cov is (kT/m) I  ->  one-step noise cov (kT/m)(I - P P^T)
        eye = torch.eye(2, device=A.device, dtype=A.dtype)
        cov = eye - P @ P.transpose(-1, -2)
        L = torch.linalg.cholesky(cov + 1e-12 * eye)
        return c, P, g, L

    def discrete_kernel(self, n_lags, dt):
        """Effective per-step kernel W_j = c^T P^j g (same convention as GLE_TS.memory_kernel)."""
        c, _, g, _ = self.step_coefficients(dt)
        j = torch.arange(n_lags, device=g.device, dtype=g.dtype)
        Pj = torch.linalg.matrix_exp(-self.drift_matrix()[None] * (j * dt)[:, None, None, None])  # (L, M, 2, 2)
        return torch.einsum('mi,lmij,mj->l', c, Pj, g)

    def continuous_kernel(self, t):
        a, *_ = self.params()
        A = self.drift_matrix()
        E = torch.linalg.matrix_exp(-A[None] * t[:, None, None, None])
        return (a * E[..., 0, 0]).sum(-1)

    def reset_state(self, like=None):
        """Zero the aux variables (or drop them so they are lazily re-created)."""
        if like is None:
            self.s = None
            return
        self.s = torch.zeros(like.shape + (self.n_modes, 2), device=like.device, dtype=like.dtype)

    def detach_state(self):
        if self.s is not None:
            self.s = self.s.detach()

    def o_coefficients(self, dt):
        """Exact propagator of the joint linear (v, s) block over dt (Ceriotti GLE-BAOAB O step).

        d(v, s) = -A_p (v, s) dt + noise,  A_p = [[0, -c^T], [c, blockdiag(A_k)]].
        The symmetric part of A_p is diag(0, g1, g2, ...), so the stationary covariance
        is (kT/m) I and the one-step noise covariance is (kT/m)(I - T T^T), exactly.
        Returns T, S (float32) with x' = T x + sqrt(kT/m) S xi, x = (v, s_1, ..., s_M).
        """
        a, *_ = self.params()
        A = self.drift_matrix().double()
        c = torch.stack((torch.sqrt(a), torch.zeros_like(a)), -1).double().flatten()  # (2M,)
        Ap = torch.cat((
            torch.cat((torch.zeros(1, device=c.device, dtype=c.dtype), -c))[None],
            torch.cat((c[:, None], torch.block_diag(*A)), 1),
        ))
        T = torch.linalg.matrix_exp(-Ap * dt)
        eye = torch.eye(T.shape[0], device=T.device, dtype=T.dtype)
        S = torch.linalg.cholesky(eye - T @ T.T + 1e-12 * eye)
        return T.float(), S.float()

    def _oc_key(self, dt):
        # parameter versions change on every in-place update (optimizer.step, load_state_dict)
        return dt, torch.is_grad_enabled(), tuple(p._version for p in self.parameters())

    def prepare(self, dt):
        """Compute the O-step coefficients once (e.g. per training window) and cache them.

        The cache is keyed on dt, grad mode and the parameter versions, so it is recomputed
        automatically after an optimizer step or load_state_dict; calling prepare is optional.
        """
        self._oc = (self._oc_key(dt), self.o_coefficients(dt))

    def O_step(self, v, T, dt, mass, t=None):
        """Exact GLE O step: propagate (v, s) jointly over dt, return the new v (used by diffgle.integrators.BAOAB)."""
        if self.s is None:
            self.reset_state(v)
        if self._oc is None or self._oc[0] != self._oc_key(dt):
            self.prepare(dt)
        Tm, S = self._oc[1]
        x = torch.cat((v[..., None], self.s.flatten(-2)), -1)  # (N, 3, 1 + 2M)
        sigma = torch.sqrt(self.BOLTZMAN * T / mass)[..., None]  # (N, 1, 1)
        x = x @ Tm.T + sigma * (torch.randn_like(x) @ S.T)
        self.s = x[..., 1:].reshape(self.s.shape)
        return x[..., 0]

    def forward(self, v, T, dt, mass, t=None):
        """Advance the aux variables by dt with the current v, return dv/dt from the GLE.

        Explicit coupling for the legacy velocity-Verlet + Euler scheme (diffgle.integrators.VVEulerRHS / VV_GLE);
        biased for fast motions. Prefer O_step with diffgle.integrators.BAOAB.
        """
        if self.s is None:
            self.reset_state(v)
        c, P, g, L = self.step_coefficients(dt)
        sigma = torch.sqrt(self.BOLTZMAN * T / mass)[..., None, None]  # (N, 1, 1, 1)
        noise = torch.einsum('mij,...mj->...mi', L, torch.randn_like(self.s))
        self.s = torch.einsum('mij,...mj->...mi', P, self.s) - g * v[..., None, None] + sigma * noise
        # friction part equals -sum_j W_j v_{n-j}, same convention as GLE_TS
        return (c * self.s).sum((-1, -2))


def fit_damped_osc(kernel, dt, n_modes, n_restarts=4, verbose=True, **kw):
    """Least-squares fit to a discrete kernel W_j (GLE_TS.memory_kernel convention).

    Runs several initialisations and keeps the best.
    Returns (a, gamma1, gamma2, omega, loss); use them to initialise GLE_TS_DampedOsc.
    """
    best = None
    for seed in range(n_restarts):
        res = _fit_once(kernel, dt, n_modes, seed=seed, **kw)
        if verbose:
            print(f"  fit[{n_modes} modes, seed {seed}] loss {res[-1]:.3e}")
        if best is None or res[-1] < best[-1]:
            best = res
    return best


def _fit_once(kernel, dt, n_modes, n_steps=3000, lr=0.03, seed=0):
    gen = torch.Generator().manual_seed(seed)
    kernel = torch.as_tensor(kernel, dtype=torch.float64)
    L = kernel.numel()
    t_max = L * dt
    # Spread decay rates over [3/t_max, 1/(5 dt)] and frequencies over [0, pi/(10 dt)];
    # seed 0 is deterministic, the others are jittered.
    jitter = 0 if seed == 0 else 1
    gamma0 = torch.logspace(math.log10(3 / t_max), math.log10(1 / (5 * dt)), n_modes, dtype=torch.float64)
    gamma0 = gamma0 * torch.exp(jitter * torch.randn(n_modes, generator=gen, dtype=torch.float64))
    omega0 = torch.linspace(0, math.pi / (10 * dt), n_modes, dtype=torch.float64)
    omega0 = omega0 * torch.exp(jitter * torch.randn(n_modes, generator=gen, dtype=torch.float64))
    a0 = torch.full((n_modes,), float(kernel[0]) / dt / n_modes, dtype=torch.float64)

    model = GLE_TS_DampedOsc(a0, gamma0, gamma0, omega0).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    scale = kernel.abs().max()
    for i in range(n_steps):
        W = model.discrete_kernel(L, dt)
        # also pin the integral (total friction), which controls diffusion
        loss = ((W - kernel) / scale).pow(2).mean() + 1e-2 * ((W.sum() - kernel.sum()) / kernel.sum()).pow(2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    a, g1, g2, w = (p.detach().float() for p in model.params())
    return a, g1, g2, w, loss.item()
