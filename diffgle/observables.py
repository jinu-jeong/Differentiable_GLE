"""Observables: VACF, MSD, RDF, 1D density, temperature."""

import numpy as np
import torch

from diffgle.constants import BOLTZMAN
from diffgle.potentials import generate_nbr_list


def gaussian_smearing(centered, sigma):
    return 1 / (sigma * (2 * np.pi) ** 0.5) * torch.exp(-0.5 * (centered / sigma) ** 2)


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def temperature(v, mass):
    """Instantaneous kinetic temperature (K) of velocities v (N, 3) with masses (N, 1)."""
    return ((mass * v.pow(2)).sum() / (3 * v.shape[0] * BOLTZMAN)).item()


class VACF_computer(torch.nn.Module):
    """VACF used by the paper pipeline (main.py).

    Time origins every 10 frames; the lagged velocities are detached, so the gradient
    flows only through v(t0). Normalised by the (non-differentiated) value at lag 0.
    The sum is divided by the number of candidate origins, not the ~10x fewer used, so the
    un-normalised VACF (normalize=False) is ~10x too small; kept as in the paper code.
    Loss values from this VACF and from vacf() below are not comparable.
    """

    def __init__(self, td_max, ensemble_average=True, normalize=True):
        super().__init__()
        self.td_max = td_max
        self.ensemble_average = ensemble_average
        self.normalize = normalize

    def forward(self, v):
        if len(v.shape) == 2:
            v = v.view(len(v), -1, 3)

        t0_list = list(range(0, len(v) - self.td_max + 1))
        if not self.ensemble_average:
            t0_list = [0]

        VACF = torch.zeros(self.td_max, device=v.device)
        for t0 in t0_list[::10]:
            dummy = v[t0:t0 + self.td_max].detach()
            VACF += (dummy * v[t0]).mean((1, 2))
        VACF /= len(t0_list)

        if self.normalize:
            VACF /= VACF[0].item()
        return VACF


def vacf(v, td_max, stride=20):
    """Normalised VACF with the gradient through both time origins and lags (used for BPTT training)."""
    origins = range(0, v.shape[0] - td_max + 1, stride)
    c = torch.stack([(v[t0:t0 + td_max] * v[t0]).mean((1, 2)) for t0 in origins]).mean(0)
    return c / c[0]


class MSD_computer(torch.nn.Module):
    def __init__(self, td_max):
        super().__init__()
        self.td_max = td_max

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(len(x), -1, 3)
        t0_list = list(range(0, len(x) - self.td_max, 100))
        MSD = torch.zeros(self.td_max, device=x.device)
        for t0 in t0_list:
            MSD += (x[t0:t0 + self.td_max] - x[t0]).pow(2).mean((1, 2))
        MSD /= len(t0_list)
        return MSD


class RDF_computer(torch.nn.Module):
    """Radial distribution function g(r) with Gaussian-smeared histogram (bin 0.1 A)."""

    def __init__(self, cell, device, L_max=None):
        super().__init__()
        self.cell = cell
        self.L_max = self.cell[0, 0] / 2 if L_max is None else L_max
        self.dr = 0.1
        self.device = device
        self.r_list = torch.arange(0.5 * self.dr, self.L_max - self.dr * 2, self.dr, device=self.device)

    def forward(self, Traj):
        for t, q in enumerate(Traj):
            nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell, cutoff=self.L_max)
            pdist_gaussian = gaussian_smearing(pdist - self.r_list, self.dr).sum(0) * self.dr
            if t == 0:
                Pdist_gaussian = pdist_gaussian
            else:
                Pdist_gaussian += pdist_gaussian
        Pdist_gaussian /= (t + 1)

        v = 4 * np.pi / 3 * ((self.r_list + 0.5 * self.dr) ** 3 - (self.r_list - 0.5 * self.dr) ** 3)
        natom = len(Traj[0])
        gr = Pdist_gaussian / v * (torch.det(self.cell)) / (natom - 1) / natom * 2
        return self.r_list, gr


class density_computer(torch.nn.Module):
    """Normalised 1D number density profile of the particles in z_mask along `axis`."""

    def __init__(self, cell, z_mask, axis, device, dr=0.1):
        super().__init__()
        self.cell = cell
        self.dr = dr
        self.device = device
        self.z_mask = z_mask
        self.axis = axis
        self.r_list = [torch.arange(0.5 * self.dr, cell[i, i] - self.dr * 2, self.dr, device=self.device) for i in range(3)]

    def forward(self, Traj):
        target = Traj[:, self.z_mask, self.axis]
        r_list = self.r_list[self.axis]
        for t, q in enumerate(target):
            pdist_gaussian = gaussian_smearing(q.unsqueeze(1) - r_list, self.dr).sum(0) * self.dr
            if t == 0:
                Pdist_gaussian = pdist_gaussian
            else:
                Pdist_gaussian += pdist_gaussian
        Pdist_gaussian /= (t + 1)
        return r_list, Pdist_gaussian / Pdist_gaussian.sum()
