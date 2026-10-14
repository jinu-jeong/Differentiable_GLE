"""Markovian Langevin thermostat (white noise, constant friction)."""

import torch

from diffgle.constants import BOLTZMAN


class Langevin_TS(torch.nn.Module):
    """dv/dt = -gamma v + sqrt(2 gamma kB T / m) xi, returned for an Euler step (v += dt * dv/dt)."""

    def __init__(self, gamma, BOLTZMAN=BOLTZMAN, Trainable=False):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.Tensor([gamma])) if Trainable else gamma
        self.BOLTZMAN = BOLTZMAN

    def forward(self, v, T, dt, mass, t):
        friction = -v * self.gamma
        Langevin_coeff = torch.sqrt(2.0 * self.gamma / mass * self.BOLTZMAN * T * dt)
        random = torch.randn_like(v, device=v.device) * Langevin_coeff / dt
        return friction + random
