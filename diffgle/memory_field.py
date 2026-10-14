"""Memory field: a GLE memory kernel that depends on the distance of a bead to a wall.

The kernel of a bead is a sum of M damped-oscillator modes (as in diffgle/thermostats/damped_osc.py),
each with four parameters (a, gamma1, gamma2, omega). In a memory field these parameters are functions
of the distance d to the nearest wall, kappa(t, d); directions parallel (x, y) and perpendicular (z) to
the walls have their own fields ("groups"). A position-independent kernel is the uniform special case.

ZKnotDampedOsc (alias MemoryField): every mode parameter is its own curve in d, given by its values at
    fixed knots (KNOTS) and straight lines between them, learned in unconstrained form
    (softplus^-1 of a, gamma1, gamma2; omega itself), so every mode is positive-real at every d.
ZKnotSuperposed (alias MemoryFieldSuperposed): the same field with the effects of both walls added.
ZDampedOsc (alias BinnedDampedOsc): one set of modes per d-bin (comparison models: 8-bin, single kernel).

O step: (v, s) of a bead is propagated with the exact one-step map of its own d,
x' = T(d) x + sqrt(kT/m) S(d) xi, S S^T = I - T T^T. Every map preserves N(0, kT/m I), so the O step
keeps the (v, s) distribution exact for any d-dependence, also when beads move between distances;
the integrator has only the usual BAOAB splitting error. For the memory field the maps are tabulated
on a grid of d (every h = 0.1 A) once per training window (prepare) and each bead uses the nearest entry.

ZBAOAB (alias FrozenWallBAOAB): B-A-O-A-B for the mobile beads; the walls stay where they are.
The force is detached (as in diffgle.potentials), so plain autograd through v gives BPTT gradients.
TabulatedGPU: tabulated pair forces evaluated on the GPU. layer_vacf_cg: VACF conditioned on the d-bin
at the time origin, parallel and perpendicular, differentiable through origins and lags.

Set the channel width with set_wall_gap(H) (walls at z = 0 and z = H) before building a model.
"""

from collections import deque

import numpy as np
import torch

from diffgle.observables import kinetic_to_temp
from diffgle.potentials import generate_nbr_list
from diffgle.thermostats.damped_osc import _inv_softplus

Z_LO, Z_HI = 0.0, 80.0  # graphene planes (A); change Z_HI with set_wall_gap


def set_wall_gap(H):
    """Walls at z = 0 and z = H (A): wall_distance(z) = min(z, H - z) everywhere in this module."""
    global Z_HI
    Z_HI = float(H)


def wall_distance(z):
    return torch.minimum(z - Z_LO, Z_HI - z)


class ZDampedOsc(torch.nn.Module):
    """Damped-oscillator modes per (component group, d-bin): parameters of shape (P, B, M)."""

    def __init__(self, edges, a, gamma1, gamma2, omega, groups=1, BOLTZMAN=0.001987191):
        """edges: d-bin edges (A), B + 1 values. a, gamma1, gamma2, omega: (M,) initial modes,
        copied to every group and bin. groups: 1 (isotropic) or 2 (x, y | z)."""
        super().__init__()
        self.BOLTZMAN = BOLTZMAN
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float))
        B, P = len(edges) - 1, groups
        rep = lambda x: torch.as_tensor(x, dtype=torch.float64).expand(P, B, -1).clone()
        self.raw_a = torch.nn.Parameter(_inv_softplus(rep(a)).float())
        self.raw_gamma1 = torch.nn.Parameter(_inv_softplus(rep(gamma1)).float())
        self.raw_gamma2 = torch.nn.Parameter(_inv_softplus(rep(gamma2)).float())
        self.omega = torch.nn.Parameter(rep(omega).float())
        self.group_comps = [[0, 1], [2]] if groups == 2 else [[0, 1, 2]]  # velocity components of each group
        self.s = None
        self._oc = None

    @property
    def shape(self):
        return tuple(self.raw_a.shape)  # (P, B, M)

    def params(self):
        sp = torch.nn.functional.softplus
        return sp(self.raw_a), sp(self.raw_gamma1), sp(self.raw_gamma2), self.omega

    def bins(self, z):
        b = torch.bucketize(wall_distance(z).contiguous(), self.edges[1:-1].contiguous(), right=True)
        return b

    def o_coefficients(self, dt):
        """T, S of shape (P, B, K, K), K = 1 + 2M, for x = (v, s_1, ..., s_M) of one component."""
        a, g1, g2, w = (p.double() for p in self.params())
        P, B, M = a.shape
        K = 1 + 2 * M
        Ap = torch.zeros(P, B, K, K, dtype=torch.float64, device=a.device)
        i = 1 + 2 * torch.arange(M, device=a.device)
        Ap[..., i, 0] = torch.sqrt(a)            # ds_k1 = -sqrt(a) v ...
        Ap[..., 0, i] = -torch.sqrt(a)           # dv = + sqrt(a) s_k1
        Ap[..., i, i] = g1
        Ap[..., i + 1, i + 1] = g2
        Ap[..., i, i + 1] = w
        Ap[..., i + 1, i] = -w
        T = torch.linalg.matrix_exp(-Ap * dt)
        eye = torch.eye(K, dtype=T.dtype, device=T.device)
        S = torch.linalg.cholesky(eye - T @ T.transpose(-1, -2) + 1e-12 * eye)
        return T.float(), S.float()

    def continuous_kernel(self, t):
        """kappa(t) for every (group, bin): (P, B, len(t))."""
        a, g1, g2, w = self.params()
        A = torch.stack((torch.stack((g1, w), -1), torch.stack((-w, g2), -1)), -2)  # (P, B, M, 2, 2)
        E = torch.linalg.matrix_exp(-A[None] * t[:, None, None, None, None, None])  # (L, P, B, M, 2, 2)
        return (a * E[..., 0, 0]).sum(-1).permute(1, 2, 0)

    def detach_state(self):
        if self.s is not None:
            self.s = self.s.detach()

    def _oc_key(self, dt):
        return dt, torch.is_grad_enabled(), tuple(p._version for p in self.parameters())

    def prepare(self, dt):
        self._oc = (self._oc_key(dt), self.o_coefficients(dt))

    def O_step(self, v, T, dt, mass, z):
        """v (N, 3), z (N,) positions along the wall normal. Returns the new v.

        Every bead is propagated with the map of every bin and the one of its own bin is picked
        with a one-hot mask: no Python loop over bins, and autograd stores only x, xi and the mask.
        """
        P, B, M = self.shape
        K = 1 + 2 * M
        if self.s is None:
            self.s = torch.zeros(v.shape + (2 * M,), device=v.device, dtype=v.dtype)
        if self._oc is None or self._oc[0] != self._oc_key(dt):
            self.prepare(dt)
        Tm, S = self._oc[1]
        x = torch.cat((v[..., None], self.s), -1)                       # (N, 3, K)
        xi = torch.randn_like(x)
        sigma = torch.sqrt(self.BOLTZMAN * T / mass)[..., None, None]  # (N, 1, 1, 1)
        onehot = torch.nn.functional.one_hot(self.bins(z), B).to(x.dtype)[:, None, :, None]   # (N, 1, B, 1)
        parts = []
        for p, comps in enumerate(self.group_comps):                    # comps are contiguous, in order
            xp, xip = x[:, comps[0]:comps[-1] + 1], xi[:, comps[0]:comps[-1] + 1]
            y = torch.einsum("nck,bjk->ncbj", xp, Tm[p]) + sigma * torch.einsum("nck,bjk->ncbj", xip, S[p])
            parts.append((y * onehot).sum(2))   # pick the own bin; saves only the mask (gather would save y)
        out = torch.cat(parts, 1)
        self.s = out[..., 1:]
        return out[..., 0]


def ostep_maps(a, g1, g2, w, dt):
    """Exact one-step maps T = expm(-A_p dt) and S (S S^T = I - T T^T) for parameters of shape (..., M)."""
    a, g1, g2, w = (p.double() for p in (a, g1, g2, w))
    M = a.shape[-1]
    K = 1 + 2 * M
    Ap = torch.zeros(a.shape[:-1] + (K, K), dtype=torch.float64, device=a.device)
    i = 1 + 2 * torch.arange(M, device=a.device)
    Ap[..., i, 0] = torch.sqrt(a)
    Ap[..., 0, i] = -torch.sqrt(a)
    Ap[..., i, i] = g1
    Ap[..., i + 1, i + 1] = g2
    Ap[..., i, i + 1] = w
    Ap[..., i + 1, i] = -w
    T = torch.linalg.matrix_exp(-Ap * dt)
    eye = torch.eye(K, dtype=T.dtype, device=T.device)
    S = torch.linalg.cholesky(eye - T @ T.transpose(-1, -2) + 1e-12 * eye)
    return T.float(), S.float()


class _TableOStep(torch.autograd.Function):
    """y_n = T[idx_n] x_n + sigma_n S[idx_n] xi_n for every bead n (x: (n, c, K)).

    Saves only x, xi, idx and the (shared) tables; the per-bead matrices are gathered again in
    backward, so the per-step memory does not grow with the size of the table.
    """

    @staticmethod
    def forward(ctx, x, xi, idx, T, S, sigma):
        y = torch.einsum("nkj,ncj->nck", T[idx], x) + sigma * torch.einsum("nkj,ncj->nck", S[idx], xi)
        ctx.save_for_backward(x, xi, idx, T, S, sigma)
        return y

    @staticmethod
    def backward(ctx, g):
        x, xi, idx, T, S, sigma = ctx.saved_tensors
        gx = torch.einsum("nkj,nck->ncj", T[idx], g) if ctx.needs_input_grad[0] else None
        gT = torch.zeros_like(T).index_add_(0, idx, torch.einsum("nck,ncj->nkj", g, x))
        gS = torch.zeros_like(S).index_add_(0, idx, torch.einsum("nck,ncj->nkj", sigma * g, xi))
        return gx, None, None, gT, gS, None


class _TabulatedOsc(torch.nn.Module):
    """Common part of the continuous-in-d thermostats: a subclass defines params_at(d) and n_modes.

    Before each window (prepare) the O-step maps are computed for d = 0, h, 2h, ..., d_max and kept in
    a list; at every step each bead uses the entry nearest to its own d. Every map preserves
    N(0, kT/m I), so the stationary (v, s) distribution is exact for any d-dependence.
    """

    def __init__(self, groups, d_max=40.0, h=0.1, BOLTZMAN=0.001987191):
        super().__init__()
        self.BOLTZMAN = BOLTZMAN
        self.register_buffer("d_grid", torch.arange(0, d_max + h / 2, h, dtype=torch.float))
        self.h = h
        self.group_comps = [[0, 1], [2]] if groups == 2 else [[0, 1, 2]]
        self.s = None
        self._oc = None

    def kernel_at(self, d, t):
        """kappa(t; d): (P, len(d), len(t)) in internal units."""
        a, g1, g2, w = self.params_at(d)
        A = torch.stack((torch.stack((g1, w), -1), torch.stack((-w, g2), -1)), -2)      # (P, D, M, 2, 2)
        E = torch.linalg.matrix_exp(-A[None] * t[:, None, None, None, None, None])
        return (a * E[..., 0, 0]).sum(-1).permute(1, 2, 0)

    def grid_coordinate(self, z):
        """Coordinate of the tabulated maps: the distance to the nearest wall."""
        return wall_distance(z)

    def detach_state(self):
        if self.s is not None:
            self.s = self.s.detach()

    def _oc_key(self, dt):
        return dt, torch.is_grad_enabled(), tuple(p._version for p in self.parameters())

    def prepare(self, dt):
        self._oc = (self._oc_key(dt), ostep_maps(*self.params_at(self.d_grid), dt))   # T, S: (P, G, K, K)

    def O_step(self, v, T, dt, mass, z):
        M = self.n_modes
        if self.s is None:
            self.s = torch.zeros(v.shape + (2 * M,), device=v.device, dtype=v.dtype)
        if self._oc is None or self._oc[0] != self._oc_key(dt):
            self.prepare(dt)
        Tm, S = self._oc[1]
        x = torch.cat((v[..., None], self.s), -1)                       # (N, 3, K)
        xi = torch.randn_like(x)
        sigma = torch.sqrt(self.BOLTZMAN * T / mass)[..., None]        # (N, 1, 1)
        idx = torch.clamp(torch.round(self.grid_coordinate(z) / self.h).long(), 0, len(self.d_grid) - 1)
        parts = [_TableOStep.apply(x[:, c[0]:c[-1] + 1], xi[:, c[0]:c[-1] + 1], idx, Tm[p], S[p], sigma)
                 for p, c in enumerate(self.group_comps)]
        out = torch.cat(parts, 1)
        self.s = out[..., 1:]
        return out[..., 0]


KNOTS = np.concatenate((np.arange(2.5, 8.01, 0.5), [9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0]))


class ZKnotDampedOsc(_TabulatedOsc):
    """Each of the 4 x M mode parameters (per component group) is its own 1D curve in d: values at
    knots (KNOTS: 0.5 A apart up to 8 A, then 1 A to 12 A, 2 A to 20 A) joined by straight lines,
    constant below the first knot and beyond the last one (the last knot is the channel-centre value).
    The curves are learned in the unconstrained form (softplus^-1 of a, gamma1, gamma2; omega itself).
    smoothness() is the mean squared second derivative of all curves (A^-2 units), added to the loss
    with a weight to keep neighbouring knots from fitting noise.
    """

    def __init__(self, a, gamma1, gamma2, omega, groups=2, knots=KNOTS, d_max=40.0, h=0.1, BOLTZMAN=0.001987191):
        """a, gamma1, gamma2, omega: (M,) bulk modes copied to every knot, or (P, n_knots, M) per knot."""
        super().__init__(groups, d_max, h, BOLTZMAN)
        P, Kn = groups, len(knots)
        self.register_buffer("knots", torch.as_tensor(knots, dtype=torch.float))

        def rep(x):
            x = torch.as_tensor(x, dtype=torch.float64)
            return x.expand(P, Kn, -1).clone() if x.dim() == 1 else x.clone()
        vals = torch.stack([_inv_softplus(rep(a)), _inv_softplus(rep(gamma1)), _inv_softplus(rep(gamma2)), rep(omega)])
        self.values = torch.nn.Parameter(vals.float())                  # (4, P, n_knots, M)

    @property
    def n_modes(self):
        return self.values.shape[-1]

    def raw_at(self, d):
        """Unconstrained parameters at d: (4, P, len(d), M), straight lines between knots."""
        k = self.knots
        dc = torch.clamp(d, k[0], k[-1])
        j = torch.clamp(torch.searchsorted(k, dc, right=True) - 1, 0, len(k) - 2)
        w = ((dc - k[j]) / (k[j + 1] - k[j]))[None, None, :, None]
        return self.values[:, :, j] * (1 - w) + self.values[:, :, j + 1] * w

    def params_at(self, d):
        raw = self.raw_at(d)
        sp = torch.nn.functional.softplus
        return sp(raw[0]), sp(raw[1]), sp(raw[2]), raw[3]

    def smoothness(self):
        k = self.knots
        h = k[1:] - k[:-1]
        slope = (self.values[:, :, 1:] - self.values[:, :, :-1]) / h[None, None, :, None]
        curv = (slope[:, :, 1:] - slope[:, :, :-1]) / (0.5 * (h[1:] + h[:-1]))[None, None, :, None]
        return curv.pow(2).mean()

    @staticmethod
    def init_from_bins(state_dict, edges, knots=KNOTS):
        """Per-knot (a, gamma1, gamma2, omega), each (P, n_knots, M), from a trained ZDampedOsc state dict:
        every knot takes the values of the bin that contains it."""
        sp = torch.nn.functional.softplus
        b = np.clip(np.searchsorted(edges, knots, side="right") - 1, 0, len(edges) - 2)
        b[-1] = len(edges) - 2          # the last knot is the channel-centre value: take the centre bin
        return (sp(state_dict["raw_a"])[:, b], sp(state_dict["raw_gamma1"])[:, b], sp(state_dict["raw_gamma2"])[:, b],
                state_dict["omega"][:, b])


class ZKnotSuperposed(ZKnotDampedOsc):
    """Knot model with the effects of both walls added, for a channel of width `wall_gap`:

        theta(z) = theta(d1) + theta(d2) - theta_c,   d1 = z, d2 = wall_gap - z,

    in the unconstrained parameters, with theta(d) the single-wall curves of a trained ZKnotDampedOsc and theta_c
    their channel-centre value (last knot). The maps are tabulated in z (0 ... wall_gap, every h). A bead far from
    one wall (d >= 20 A) gets exactly the single-wall model. Load a ZKnotDampedOsc state dict without 'd_grid'.
    """

    def __init__(self, a, gamma1, gamma2, omega, groups=2, wall_gap=80.0, knots=KNOTS, h=0.1, BOLTZMAN=0.001987191):
        super().__init__(a, gamma1, gamma2, omega, groups=groups, knots=knots, h=h, BOLTZMAN=BOLTZMAN)
        self.wall_gap = float(wall_gap)
        self.register_buffer("d_grid", torch.arange(0, self.wall_gap + h / 2, h, dtype=torch.float), persistent=False)

    def grid_coordinate(self, z):
        return z

    def params_at(self, z):
        z = torch.as_tensor(z, dtype=self.values.dtype, device=self.values.device)
        far = torch.full_like(z[:1], float(self.knots[-1]))
        raw = self.raw_at(z) + self.raw_at(self.wall_gap - z) - self.raw_at(far)
        sp = torch.nn.functional.softplus
        return sp(raw[0]), sp(raw[1]), sp(raw[2]), raw[3]


class TabulatedGPU(torch.nn.Module):
    """Same forces as diffgle.potentials.Tabulated_specific, evaluated on the GPU.

    Each table's cubic spline (as in Tabulated_specific) is sampled once on a 1e-4 A grid
    and linearly interpolated, so no CPU round trip per step. Pairs: all within 6 A (as
    generate_nbr_list); pairs whose types match no table (wall-wall) get no force.
    """

    def __init__(self, cell, z, tables, interactions, h=1e-4, rmax=6.0):
        super().__init__()
        from scipy.interpolate import interp1d
        self.cell = cell
        self.z = torch.as_tensor(z, device=cell.device)
        self.h = h
        r = np.arange(0, rmax + 2 * h, h)
        F = []
        for t in tables:
            f = interp1d(t[:, 1], t[:, 3], kind="cubic", fill_value="extrapolate", bounds_error=False)
            F.append(f(r))
        self.F = torch.tensor(np.stack(F), dtype=torch.float, device=cell.device)       # (n_tables, n_grid)
        nt = int(self.z.max()) + 1
        lut = torch.full((nt, nt), -1, dtype=torch.long)
        for i, (t1, t2) in enumerate(interactions):
            lut[t1, t2] = lut[t2, t1] = i
        self.lut = lut.to(cell.device)

    def forward(self, q):
        nbr, _, dist, unit = generate_nbr_list(q, self.cell)
        tab = self.lut[self.z[nbr[:, 0]], self.z[nbr[:, 1]]]
        keep = tab >= 0
        nbr, dist, unit, tab = nbr[keep], dist[keep, 0], unit[keep], tab[keep]
        u = dist / self.h
        i0 = u.long()
        w = (u - i0)
        fm = self.F[tab, i0] * (1 - w) + self.F[tab, i0 + 1] * w
        fv = fm[:, None] * unit
        force = torch.zeros_like(q)
        force.index_add_(0, nbr[:, 0], fv)
        force.index_add_(0, nbr[:, 1], -fv)
        return force.detach()


class ZBAOAB(torch.nn.Module):
    """BAOAB for the mobile beads; frozen beads (walls) are fixed in q_template."""

    def __init__(self, potential, thermostat, mass, dt, T_target, q_template, mobile):
        super().__init__()
        self.potential = potential
        self.thermostat = thermostat
        self.mass = mass                    # (n_mobile, 1)
        self.dt = dt
        self.T_target = T_target
        self.q_template = q_template.detach().clone()
        self.mobile = torch.nonzero(mobile).squeeze(1)
        self._cached = None
        self.Temperature_log = deque(maxlen=1_000_000)

    def force(self, q):
        if self._cached is not None and torch.equal(self._cached[0], q):
            return self._cached[1]
        qf = self.q_template.clone()
        qf[self.mobile] = q
        return self.potential(qf)[self.mobile]

    def step(self, q, v):
        f0 = self.force(q)
        v = v + 0.5 * (f0 / self.mass) * self.dt
        q = q + 0.5 * v.detach() * self.dt
        z_o = q[:, 2]
        v = self.thermostat.O_step(v, self.T_target, self.dt, self.mass, z_o)
        q = (q + 0.5 * v.detach() * self.dt).detach()
        f1 = self.force(q)
        v = v + 0.5 * (f1 / self.mass) * self.dt
        self._cached = (q, f1)
        KE = 0.5 * (self.mass * v.detach().pow(2)).sum()
        self.Temperature_log.append(kinetic_to_temp(KE, v.shape[0]).item())
        return q, v, z_o.detach()

    def simulate(self, q, v, n_steps):
        """Returns q, vs (n+1, N, 3), zs (n+1, N) positions along z at each recorded v, temperatures."""
        self.thermostat.prepare(self.dt)
        vs, zs = [v], [q[:, 2].detach()]
        for _ in range(n_steps):
            q, v, _ = self.step(q, v)
            vs.append(v)
            zs.append(q[:, 2])
        return q, torch.stack(vs), torch.stack(zs), list(self.Temperature_log)[-n_steps:]


def layer_vacf_cg(vs, zs, edges, td_max, stride=20):
    """Velocity autocorrelation conditioned on the d-bin at the time origin.

    vs (T, N, 3), zs (T, N). Returns par (B, td_max), perp (B, td_max), counts (B,), all
    unnormalised (divide by C(0) of the same bin to compare with the AA target). The gradient
    flows through the origin and the lagged velocities. vs is laid out once as (3, N, T) so
    that each origin is one batched matmul with a view of it (autograd keeps views, not copies).
    """
    edges_t = torch.as_tensor(edges, dtype=zs.dtype, device=zs.device)
    B = len(edges) - 1
    vt = vs.permute(2, 1, 0).contiguous()                                           # (3, N, T)
    C = torch.zeros(3, B, td_max, device=vs.device, dtype=vs.dtype)
    cnt = torch.zeros(B, device=vs.device, dtype=vs.dtype)
    for t0 in range(0, vs.shape[0] - td_max + 1, stride):
        b = torch.bucketize(wall_distance(zs[t0]).contiguous(), edges_t[1:-1].contiguous(), right=True)
        onehot = torch.nn.functional.one_hot(b, B).to(vs.dtype)                     # (N, B)
        w = onehot.T[None] * vt[:, None, :, t0]                                      # (3, B, N)
        C = C + torch.bmm(w, vt[:, :, t0:t0 + td_max])                               # (3, B, td)
        cnt = cnt + onehot.sum(0)
    par = 0.5 * (C[0] + C[1])
    return par / cnt[:, None], C[2] / cnt[:, None], cnt


# names used in the paper and the README
MemoryField = ZKnotDampedOsc
MemoryFieldSuperposed = ZKnotSuperposed
BinnedDampedOsc = ZDampedOsc
FrozenWallBAOAB = ZBAOAB
