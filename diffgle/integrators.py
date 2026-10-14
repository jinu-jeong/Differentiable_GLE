"""Integrators.

VVEulerRHS : paper scheme. A right-hand side for torchdiffeq's fixed-step Euler that bakes
             velocity Verlet into the drift; the thermostat's dv/dt is added explicitly.
             Only valid with method="euler". Used by main.py (odeint_adjoint training).
BAOAB      : B-A-O-A-B splitting (Leimkuhler & Matthews 2013). Uses the thermostat's exact
             O_step if it has one (GLE-BAOAB for GLE_TS_DampedOsc), else an Euler O step.
             Default for the damped-osc experiments.
VV_GLE     : explicit-loop version of VVEulerRHS (legacy comparison for damped-osc).

BAOAB and VV_GLE share simulate(q, v, n_steps) -> (q, vs (n+1, N, 3), temperatures) and
assume a detached force (diffgle.potentials), so q never carries a graph and plain
autograd through vs gives the BPTT gradient w.r.t. the thermostat parameters.

Why BAOAB: with the explicit thermostat of VV+Euler, a harmonic mode with a 60 fs period
equilibrates ~40% too hot under the trained H2O damped-osc kernel; GLE-BAOAB is exact for
positions and within (w dt / 2)^2 for velocities (experiments/check_integrator_bias.py).
"""

from collections import deque

import torch

from diffgle.observables import kinetic_to_temp, temperature


class VVEulerRHS(torch.nn.Module):
    """y = (v, q) stacked along dim 0. Euler step y += dt * rhs reproduces

        q' = q + dt v + dt^2/2 a(q)
        v' = v + dt/2 (a(q) + a(q')) + dt * thermostat(v)

    non_integrand_mask: particles kept frozen (e.g. confining walls).
    The attribute name `Thermostat` is kept so saved state dicts ('Thermostat.h') load.
    """

    def __init__(self, potential, Thermostat, Temp_target, dt, mass, non_integrand_mask=None):
        super().__init__()
        self.potential = potential
        self.Thermostat = Thermostat
        self.Temp_target = Temp_target
        self.dt = dt
        self.mass = mass
        self.non_integrand_mask = non_integrand_mask
        self.Temperature_log = deque(maxlen=1_000_000)

    def forward(self, t, state):
        original_state_shape = state.shape
        state = state.view(-1, 3)
        Natom = len(state) // 2

        with torch.set_grad_enabled(True):
            v0 = state[:Natom].requires_grad_(True)
            q0 = state[Natom:].requires_grad_(True)

            dv_dt_thermostat = self.Thermostat(v0, self.Temp_target, self.dt, self.mass, t=t) if self.Thermostat is not None else 0

            a0 = self.potential(q0) / self.mass
            dqdt = v0 + 0.5 * a0 * self.dt
            q1 = q0 + dqdt * self.dt
            a1 = self.potential(q1) / self.mass
            dvdt = 0.5 * (a0 + a1) + dv_dt_thermostat

        if self.non_integrand_mask is not None:
            dvdt = dvdt.clone()
            dqdt = dqdt.clone()
            dvdt[self.non_integrand_mask] = 0
            dqdt[self.non_integrand_mask] = 0

        # temperature of the moving particles only (frozen walls have no kinetic DOF)
        moving = slice(None) if self.non_integrand_mask is None else ~self.non_integrand_mask
        KE = (self.mass[moving] * (v0[moving].pow(2))).sum() / 2
        self.Temperature_log.append(kinetic_to_temp(KE, v0[moving].shape[0]).item())

        return torch.concatenate((dvdt, dqdt)).view(original_state_shape)


class BAOAB(torch.nn.Module):
    """B (half kick) - A (half drift) - O (thermostat, full dt) - A - B, with force caching."""

    def __init__(self, potential, thermostat, mass, dt, T_target):
        super().__init__()
        self.potential = potential
        self.thermostat = thermostat
        self.mass = mass
        self.dt = dt
        self.T_target = T_target
        self._cached = None  # (q, force(q)) from the last step
        self.Temperature_log = deque(maxlen=1_000_000)

    def _force(self, q):
        """Force at q, reusing the last step's force only if q is that same configuration."""
        if self._cached is not None and torch.equal(self._cached[0], q):
            return self._cached[1]
        return self.potential(q)

    def step(self, q, v):
        f0 = self._force(q)
        v = v + 0.5 * (f0 / self.mass) * self.dt                         # B
        q = q + 0.5 * v * self.dt                                         # A
        if hasattr(self.thermostat, "O_step"):                            # O
            v = self.thermostat.O_step(v, self.T_target, self.dt, self.mass)
        else:
            v = v + self.thermostat(v, self.T_target, self.dt, self.mass, None) * self.dt
        q = q + 0.5 * v * self.dt                                         # A
        f1 = self.potential(q)
        v = v + 0.5 * (f1 / self.mass) * self.dt                         # B
        self._cached = (q.detach(), f1)
        KE = 0.5 * (self.mass * v.detach().pow(2)).sum()
        self.Temperature_log.append(kinetic_to_temp(KE, v.shape[0]).item())
        return q, v

    def simulate(self, q, v, n_steps):
        # O-step coefficients depend on the parameters: recompute once per call (i.e. per window)
        if hasattr(self.thermostat, "prepare"):
            self.thermostat.prepare(self.dt)
        vs = [v]
        for _ in range(n_steps):
            q, v = self.step(q, v)
            q = q.detach()  # force is detached, so q never needs a graph
            vs.append(v)
        return q, torch.stack(vs), list(self.Temperature_log)[-n_steps:]


class VV_GLE(torch.nn.Module):
    """Same update as VVEulerRHS + Euler, as an explicit loop with one force evaluation per step."""

    def __init__(self, potential, thermostat, mass, dt, T_target):
        super().__init__()
        self.potential = potential
        self.thermostat = thermostat
        self.mass = mass
        self.dt = dt
        self.T_target = T_target
        self._cached = None  # (q, force(q)) at the end of the last call

    def simulate(self, q, v, n_steps):
        vs, temps = [v], []
        # reuse the last force only when continuing from the same configuration
        f0 = self._cached[1] if self._cached is not None and torch.equal(self._cached[0], q) else self.potential(q)
        for _ in range(n_steps):
            a0 = f0 / self.mass
            dv_thermo = self.thermostat(v, self.T_target, self.dt, self.mass)
            q = (q + (v + 0.5 * a0 * self.dt) * self.dt).detach()
            f1 = self.potential(q)
            v = v + (0.5 * (a0 + f1 / self.mass) + dv_thermo) * self.dt
            f0 = f1
            vs.append(v)
            temps.append(temperature(v.detach(), self.mass))
        self._cached = (q, f0)
        return q, torch.stack(vs), temps


INTEGRATORS = {"baoab": BAOAB, "vv": VV_GLE}
