"""Tabulated CG pair potentials (forces only, detached: gradients never flow through the force)."""

import numpy as np
import torch
from scipy.interpolate import interp1d


def generate_nbr_list(coordinates, lattice_matrix, cutoff=6.0):
    """All pairs within cutoff in an orthorhombic periodic box.

    Returns (pairs (P, 2), image offsets, distances (P, 1), unit vectors (P, 3) pointing from j to i).
    """
    lattice_matrix_diag = torch.diag(lattice_matrix).view(1, 1, -1)

    displacement = coordinates[..., None, :, :] - coordinates[..., :, None, :]
    # minimum image convention
    offsets = ((displacement + lattice_matrix_diag / 2) // lattice_matrix_diag).detach()
    displacement = displacement - offsets * lattice_matrix_diag

    squared_displacement = torch.triu(displacement.pow(2).sum(-1))
    within_cutoff = (squared_displacement < cutoff ** 2) & (squared_displacement != 0)
    neighbor_indices = torch.nonzero(within_cutoff.to(torch.long), as_tuple=False)

    offsets = offsets[neighbor_indices[:, 0], neighbor_indices[:, 1], :]
    unit_vectors = displacement[neighbor_indices[:, 0], neighbor_indices[:, 1]]
    magnitudes = squared_displacement[neighbor_indices[:, 0], neighbor_indices[:, 1]].sqrt()
    unit_vectors = unit_vectors / magnitudes.view(-1, 1)
    actual_distances = magnitudes[:, None]

    return neighbor_indices.detach(), offsets, actual_distances, -unit_vectors


def load_table(path):
    """LAMMPS-style table file -> array with columns (index, r, energy, force), NaN rows dropped."""
    table = np.loadtxt(path, skiprows=3)
    return table[~np.isnan(table[:, -1])]


class Tabulated(torch.nn.Module):
    """Single pair potential for all particles; force magnitude from a cubic spline of the table."""

    def __init__(self, cell, table):
        super().__init__()
        self.cell = cell
        self.force_magnitude = interp1d(table[:, 1], table[:, 3], kind='cubic', fill_value="extrapolate", bounds_error=False)

    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        force = torch.zeros_like(q, device=q.device)
        force_magnitude = torch.tensor(self.force_magnitude(pdist.detach().cpu().numpy())).to(q.device).to(q.dtype)
        force_vector = force_magnitude * unit_vector
        force.index_add_(0, nbr_list[:, 0], force_vector)
        force.index_add_(0, nbr_list[:, 1], -force_vector)
        return force.detach()


class Tabulated_specific(torch.nn.Module):
    """Type-specific pair potentials: tables[i] applies to particle-type pair interactions[i] (e.g. [7, 5])."""

    def __init__(self, cell, z, tables, interactions):
        super().__init__()
        self.cell = cell
        self.z = z.to(cell.device)
        self.force_magnitude = []
        for table, interaction in zip(tables, interactions):
            f = interp1d(table[:, 1], table[:, 3], kind='cubic', fill_value="extrapolate", bounds_error=False)
            self.force_magnitude.append([f, torch.tensor(interaction).to(cell.device)])

    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        force = torch.zeros_like(q, device=q.device)
        for f, interaction in self.force_magnitude:
            force_magnitude = torch.tensor(f(pdist.detach().cpu().numpy())).to(q.device).to(q.dtype)
            force_vector = force_magnitude * unit_vector
            atom_types_pair = self.z[nbr_list]
            # pair (i, j) uses this table if its types are (a, b) or (b, a). (Was
            # ((pair == ab) | (pair == ba)).all(1), an element-wise OR that also matched (a, a) and (b, b),
            # so every water-water pair got the water-wall table on top of its own.)
            mask = (atom_types_pair == interaction).all(dim=1) | (atom_types_pair == interaction.flip(0)).all(dim=1)
            force.index_add_(0, nbr_list[mask, 0], force_vector[mask])
            force.index_add_(0, nbr_list[mask, 1], -force_vector[mask])
        return force.detach()
