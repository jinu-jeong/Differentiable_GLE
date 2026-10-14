"""Per-system settings and data loading. Every system-specific number lives in SYSTEMS.

data/processed/<name>/ must contain pos0.npy, vel0.npy, box.npy, vacf_aa.npy, msd_aa.npy, CG/*.pot,
plus r_aa.npy / rdf_aa.npy (bulk) or z.npy, z_aa.npy, density_aa.npy (confined). They are made from
data/raw/<name>/ by preprocess.py (H2O, CO2) or by experiments/confined/allatom/make_dataset.py
(graphene_water_H<H>).

Systems: H2O, CO2 (bulk); graphene_water_H<H>, H = 20, 30, 40, 80 (water between graphene sheets
H A apart, all-atom references run with experiments/confined/allatom/). Names used before
2026-10-13 are accepted through ALIASES.

Trajectory lengths are counted in frames: n frames = the initial state + (n - 1) steps of 1 fs
(this is how main.py's odeint time grid works; kept for bit-compatibility with the paper runs).
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torchmd.systems import System

from diffgle.constants import PROCESSED_DIR
from diffgle.potentials import Tabulated, Tabulated_specific, load_table

WATER, WALL = 7, 5  # particle type ids in z.npy (also used as element index in xyz output)


@dataclass(frozen=True)
class SystemConfig:
    mass: float                     # CG bead mass (amu)
    # --- paper pipeline (main.py) ---
    filter_length: int              # GLE_TS filter taps (= memory length in steps of 1 fs)
    langevin_gamma: float           # friction of the Markovian baseline run
    baseline_frames: int            # length of the Markovian baseline run
    baseline_zero_velocity: bool    # start the baseline from v = 0
    pre_equil_frames: int           # no-grad frames before every training window (1 = none)
    restart: str                    # next window starts from "end" of the window or "after_filter" (frame filter_length+1)
    optimizer_schedule: tuple       # ((start_iter, optimizer name, lr), ...)
    diagnostic_len: int = 1000      # MSD / VACF length for the diagnostic plots
    trapezoidal_kernel: bool = False
    pretrained: Optional[str] = None  # transfer-learning init (state dict of another run)
    # --- system description ---
    confined: bool = False          # walls (type 5) are frozen; observables use water (type 7) only
    notes: str = field(default="", compare=False)

    def __post_init__(self):
        assert self.optimizer_schedule[0][0] == 0, "optimizer_schedule must start at iteration 0"
        assert self.restart in ("end", "after_filter")


SYSTEMS = {
    "H2O": SystemConfig(
        mass=18.0, filter_length=1000, langevin_gamma=0.05, baseline_frames=2000, baseline_zero_velocity=False,
        pre_equil_frames=1, restart="end",  # 1 frame: windows follow each other directly
        optimizer_schedule=((0, "Adam", 3e-4), (100, "Adam", 1e-4), (200, "Adam", 1e-5)),
        notes="bulk water, 2141 CG beads",
    ),
    "CO2": SystemConfig(
        mass=44.1, filter_length=200, langevin_gamma=0.05, baseline_frames=2000, baseline_zero_velocity=False,
        pre_equil_frames=200, restart="after_filter",
        optimizer_schedule=((0, "SGD", 1e-3),),
        notes="bulk CO2, 1700 CG beads",
    ),
}
# water between graphene sheets H A apart; all-atom references run with experiments/confined/allatom/ (exact
# molecular centres of mass, 300 K); main.py settings as for the earlier 80 A confined system of DiffGLE
for _H in (20, 30, 40, 80):
    SYSTEMS[f"graphene_water_H{_H}"] = SystemConfig(
        mass=18.01528, filter_length=1000, langevin_gamma=0.5, baseline_frames=1000, baseline_zero_velocity=True,
        pre_equil_frames=2000, restart="after_filter",
        optimizer_schedule=((0, "Adam", 1e-5),),
        diagnostic_len=500, trapezoidal_kernel=True, confined=True,
        notes=f"water between graphene sheets {_H} A apart (experiments/confined/allatom/run_confined.py)",
    )

# names used before 2026-10-13 (folders Data/<old name>), kept so that old commands and run configs still load
ALIASES = {f"Confined_H2O_v2_H{_H}": f"graphene_water_H{_H}" for _H in (20, 30, 40, 80)}


def load_system(name, device):
    """Initial configuration, CG potential and all-atom targets of a system.

    Returns a namespace with: cfg, N, mass (N, 1), q0, v0 (N, 3), cell (3, 3), potential,
    types (N,) particle type ids, frozen (N,) bool or None, mobile (N,) bool or None,
    and targets: vacf, msd, plus rdf=(r, g) (bulk) or density=(z, rho) (confined).
    """
    name = ALIASES.get(name, name)
    cfg = SYSTEMS[name]
    d = os.path.join(PROCESSED_DIR, name)
    load = lambda f: np.load(os.path.join(d, f))

    pos0, vel0, box = load("pos0.npy"), load("vel0.npy"), load("box.npy")
    N = pos0.shape[0]
    md_system = System(N, nreplicas=1, precision=torch.float, device=device)
    md_system.set_positions(torch.tensor(pos0, device=device, dtype=torch.float)[:, :, None])
    md_system.set_box(box)
    md_system.set_velocities(torch.tensor(vel0, device=device, dtype=torch.float)[None])
    cell = md_system.box[0]

    s = SimpleNamespace(cfg=cfg, name=name, N=N, cell=cell,
                        mass=torch.full((N, 1), cfg.mass).to(device),
                        q0=md_system.pos[0], v0=md_system.vel[0])
    t = lambda a: torch.tensor(a, device=device)
    s.targets = SimpleNamespace(vacf=t(load("vacf_aa.npy")), msd=t(load("msd_aa.npy")))

    if cfg.confined:
        z = t(load("z.npy"))
        s.types, s.frozen, s.mobile = z, z == WALL, z == WATER
        # Known quirk, kept for bit-reproducibility of the paper runs: vel0.npy carries thermal
        # velocities on the frozen wall beads. Walls never move (dq = dv = 0), but VVEulerRHS
        # evaluates the second VV force at q0 + (v0 + a0 dt/2) dt for all beads, so during that
        # force call the walls are displaced by ~v dt (~3e-4 A). Zeroing s.v0[s.frozen] removes
        # this but changes the (chaotic) trajectories.
        s.potential = Tabulated_specific(
            cell, z,
            [load_table(os.path.join(d, "CG", "CG1_CG1.pot")), load_table(os.path.join(d, "CG", "CG1_GR1.pot"))],
            [[WATER, WATER], [WATER, WALL]],
        ).to(device)
        s.targets.density = (t(load("z_aa.npy")), t(load("density_aa.npy")))
    else:
        s.types, s.frozen, s.mobile = [WATER] * N, None, None
        s.potential = Tabulated(cell, load_table(os.path.join(d, "CG", "CG_CG.pot"))).to(device)
        s.targets.rdf = (load("r_aa.npy"), load("rdf_aa.npy"))
    return s
