# %% Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from force import *
from utility import *

from torchdiffeq import odeint_adjoint

# Set global font size for matplotlib
plt.rcParams.update({"font.size": 16})

# Reproducibility (IMPORTANT for deterministic adjoint with pre-sampled noise)
torch.manual_seed(0)
np.random.seed(0)

# %% Configuration
CONFIG = {
    "CO2": {"mass": 44.0, "box": [40, 40, 40], "filter_length": 1000},
    "H2O": {"mass": 18.0, "box": [40, 40, 40], "filter_length": 1000},
    "Confined_H2O": {"mass": 18.0, "box": [25.174493, 29.778001, 112], "filter_length": 1000},
}

MOLECULE = "H2O"

# %% Global Variables
DEVICE = "cuda:0"
PRECISION = torch.float
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
DT = 1  # femtoseconds (fs)

# Physical time step used by the integrator time-grid (same as your t construction)
DT_PHYS = DT / TIMEFACTOR

# %% Load Data
X = torch.tensor(np.load(f"./{MOLECULE}/AA/pos_COM.npy"), dtype=PRECISION, device=DEVICE)
V = torch.tensor(np.load(f"./{MOLECULE}/AA/vel_COM.npy"), dtype=PRECISION, device=DEVICE)

N_ATOMS = X.shape[1]
MASS = torch.full((N_ATOMS, 1), CONFIG[MOLECULE]["mass"], dtype=PRECISION, device=DEVICE)

# %% Functions
def initialize_system(positions, velocities, box, device=DEVICE, precision=PRECISION):
    """Initializes the simulation system with positions, velocities, and a diagonalized box."""
    positions = torch.tensor(positions, dtype=precision, device=device)
    velocities = torch.tensor(velocities, dtype=precision, device=device)
    box = torch.diag(torch.tensor(box[:3], dtype=precision, device=device))
    return positions, velocities, box

def maxwell_boltzmann_velocity(masses, temperature, device=DEVICE, boltzmann=BOLTZMAN):
    """Generate Maxwell-Boltzmann distributed velocities in 3D."""
    stddev = torch.sqrt(boltzmann * temperature / masses)  # (N, 1)
    velocities = torch.normal(mean=0.0, std=stddev.repeat(1, 3)).to(device)  # (N, 3)
    return velocities

def wrap_positions(q, box_diag):
    """Wrap positions into [0, L) using a differentiable-ish remainder (piecewise)."""
    return q - torch.floor(q / box_diag) * box_diag

# %% Initialize System
positions, velocities, box = initialize_system(
    positions=(X[-1] % CONFIG[MOLECULE]["box"][0]),
    velocities=maxwell_boltzmann_velocity(MASS, temperature=300),
    box=CONFIG[MOLECULE]["box"],
)
q, cell, v = positions, box, velocities
box_diag = torch.diag(cell).view(1, 3)

# %% Load Potential
tabulated_data = np.loadtxt(f"./{MOLECULE}/CG/CG_CG.pot", skiprows=3)
tabulated_data = tabulated_data[~np.isnan(tabulated_data[:, -1])]
potential_gt = Tabulated(cell, tabulated_data).to(DEVICE)

# =============================================================================
# 1) Demonstration with Langevin Thermostat (UNCHANGED)
# =============================================================================
thermostat_Langevin = Langevin_TS(gamma=0.05).to(DEVICE)
simulation_Langevin = SDE(
    potential_gt,
    Thermostat=thermostat_Langevin,
    Temp_target=300,
    timestep=DT,
    TIMEFACTOR=TIMEFACTOR,
    mass=MASS,
    non_integrand_mask=None,
    saver=False,
)
simulation_Langevin.force_mode = True

y0 = torch.cat((v, q))
_ = simulation_Langevin(0.0, y0)

T = 2000
t = torch.tensor(np.arange(0, DT_PHYS * T, DT_PHYS), dtype=PRECISION, device=DEVICE)

with torch.no_grad():
    y_Langevin = odeint_adjoint(simulation_Langevin, y0, t, method="euler")

plt.plot(simulation_Langevin.Temperature_log, "k")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (K)")
plt.show()

RDF_computer = RDF_computer(cell, DEVICE)
MSD_computer = MSD_computer(1000)
VACF_computer = VACF_computer(1000)
VACF_computer.normalize = True
VACF_computer.ensemble_average = True

r_Langevin, RDF_Langevin = RDF_computer(y_Langevin[::10, N_ATOMS:])
MSD_Langevin = MSD_computer(y_Langevin[:, N_ATOMS:])
VACF_Langevin = VACF_computer(y_Langevin[:, :N_ATOMS])

plt.plot(r_Langevin.cpu(), RDF_Langevin.cpu(), "k")
plt.xlabel("Distance (Å)")
plt.ylabel("RDF")
plt.show()

plt.plot(MSD_Langevin.cpu(), "k")
plt.xlabel("Time Steps")
plt.ylabel("MSD (Å²)")
plt.show()

plt.plot(VACF_Langevin.cpu(), "k")
plt.xlabel("Time Steps")
plt.ylabel("VACF")
plt.show()

# =============================================================================
# 2) OPTION A (RECOMMENDED): Markovian embedding with sum-of-exponentials kernel
#    - No internal mutable history buffers
#    - No torch.randn() inside forward
#    - Noise is pre-sampled and indexed by time-grid => deterministic for adjoint
# =============================================================================

class ExpKernelMarkovianGLE(nn.Module):
    """
    Markovian embedding for a GLE whose memory kernel is parameterized as:
        K(t) = sum_{i=1..M} kappa_i * exp(-lambda_i t),   with kappa_i >= 0, lambda_i > 0

    Auxiliary variables s_i follow:
        ds_i = (-lambda_i s_i + kappa_i v) dt + sqrt(2 k_B T kappa_i lambda_i) dW_i
    and velocity follows:
        dv = (F(q) - sum_i s_i) / m * dt

    We implement it as a deterministic ODEFunc for torchdiffeq-euler by
    pre-sampling eps ~ N(0,1) and using:
        dW_i ≈ sqrt(dt) * eps[n]
    so inside the derivative we inject:
        (sqrt(2 k_B T kappa_i lambda_i) * eps[n]) / sqrt(dt)
    because Euler step multiplies by dt.
    """

    def __init__(
        self,
        potential,
        cell,
        mass,
        temp_target=300.0,
        kB=BOLTZMAN,
        dt=DT_PHYS,
        n_modes=8,
        seed=0,
        max_steps_noise=20000,
        log_temperature=False,
    ):
        super().__init__()
        self.potential = potential
        self.cell = cell
        self.mass = mass
        self.temp_target = float(temp_target)
        self.kB = float(kB)
        self.dt = float(dt)
        self.n_modes = int(n_modes)

        # positivity constraints via softplus
        # initialize to something mild (you can tune)
        self._raw_lambdas = nn.Parameter(torch.full((self.n_modes,), 1.0, dtype=PRECISION, device=DEVICE))
        self._raw_kappas  = nn.Parameter(torch.full((self.n_modes,), 1e-3, dtype=PRECISION, device=DEVICE))

        self.softplus = nn.Softplus()

        # pre-sampled noise schedule: shape (max_steps, N, 3, M)
        self.max_steps_noise = int(max_steps_noise)
        self.seed = int(seed)
        self.register_buffer("_noise_eps", torch.zeros(1, dtype=PRECISION, device=DEVICE), persistent=False)
        self._build_noise_schedule(self.seed, self.max_steps_noise)

        # optional logging (OFF during training with adjoint; ON for evaluation)
        self.log_temperature = bool(log_temperature)
        self.Temperature_log = []

        # cache sizes
        self.N = int(self.mass.shape[0])
        self.box_diag = torch.diag(self.cell).view(1, 3)

    def _build_noise_schedule(self, seed, max_steps):
        g = torch.Generator(device=DEVICE)
        g.manual_seed(seed)
        # eps ~ N(0,1)
        eps = torch.randn(
            (max_steps, self.N, 3, self.n_modes),
            dtype=PRECISION,
            device=DEVICE,
            generator=g,
        )
        self._noise_eps = eps

    def resample_noise(self, seed=None):
        if seed is None:
            seed = self.seed + 1
        self.seed = int(seed)
        self._build_noise_schedule(self.seed, self.max_steps_noise)

    def get_params_positive(self):
        lambdas = self.softplus(self._raw_lambdas) + 1e-8   # (M,)
        kappas  = self.softplus(self._raw_kappas)  + 1e-12  # (M,)
        return lambdas, kappas

    def memory_kernel_discrete(self, num_steps):
        """
        Returns K[n] = sum_i kappa_i exp(-lambda_i * n*dt)
        shape: (num_steps,)
        """
        lambdas, kappas = self.get_params_positive()
        n = torch.arange(num_steps, device=DEVICE, dtype=PRECISION)
        t = n * self.dt
        K = torch.zeros_like(t)
        for i in range(self.n_modes):
            K = K + kappas[i] * torch.exp(-lambdas[i] * t)
        return K

    def _force_from_potential(self, q_wrapped):
        """
        Robustly try common interfaces:
          - potential(q) -> force
          - potential(q) -> (E, force)
        Adjust here if your Tabulated API differs.
        """
        out = self.potential(q_wrapped)
        if isinstance(out, tuple) or isinstance(out, list):
            # guess: (energy, force)
            F = out[-1]
        else:
            F = out
        return F

    def _temperature(self, v):
        # T = 2K/(dof*kB), K = 0.5 sum m v^2
        K = 0.5 * (self.mass * (v ** 2)).sum()
        dof = 3.0 * float(self.N)
        T = 2.0 * K / (dof * self.kB)
        return T

    def forward(self, t, y):
        """
        y shape: ((2 + M) * N, 3)
          y[:N]         = v
          y[N:2N]       = q
          y[2N:]        = s flattened as (M*N, 3)
        returns dy/dt with same shape
        """
        N = self.N
        M = self.n_modes

        v = y[:N, :]
        q = y[N:2 * N, :]
        s_flat = y[2 * N:, :]              # (M*N, 3)
        s = s_flat.view(M, N, 3)           # (M, N, 3)
        s_sum = s.sum(dim=0)               # (N, 3)

        # wrap for PBC before force
        q_wrapped = wrap_positions(q, self.box_diag)

        # forces
        F = self._force_from_potential(q_wrapped)

        # dv/dt and dq/dt
        dv = (F - s_sum) / self.mass
        dq = v

        # map time to step index (deterministic under your euler time-grid)
        # idx = round(t/dt)
        idx = torch.round(t / self.dt).to(torch.long)
        idx = torch.clamp(idx, 0, self.max_steps_noise - 1)

        eps = self._noise_eps[idx, :, :, :]   # (N, 3, M)

        lambdas, kappas = self.get_params_positive()  # (M,), (M,)

        # reshape params for broadcasting
        lambdas_b = lambdas.view(M, 1, 1)
        kappas_b  = kappas.view(M, 1, 1)

        # noise amplitude from FDT
        # sigma_i = sqrt(2 kB T kappa_i lambda_i)
        sigma = torch.sqrt(2.0 * self.kB * self.temp_target * kappas * lambdas).view(M, 1, 1)

        # ds/dt with Euler-consistent noise injection:
        # s_{n+1} = s_n + dt*(-lambda s + kappa v) + sigma*sqrt(dt)*eps
        # => ds/dt = (-lambda s + kappa v) + sigma*eps/sqrt(dt)
        eps_perm = eps.permute(2, 0, 1).contiguous()  # (M, N, 3)
        ds = (-lambdas_b * s + kappas_b * v.view(1, N, 3)) + sigma * eps_perm / np.sqrt(self.dt)

        # optional logging (DO NOT use during training; recompute in adjoint would spam logs)
        if self.log_temperature:
            with torch.no_grad():
                self.Temperature_log.append(self._temperature(v).detach().cpu().item())

        dy = torch.cat(
            (
                dv,
                dq,
                ds.view(M * N, 3),
            ),
            dim=0,
        )
        return dy

# Instantiate Markovian GLE dynamics
N_MODES = 8  # you can tune (e.g., 4, 8, 16)
func_gle = ExpKernelMarkovianGLE(
    potential=potential_gt,
    cell=cell,
    mass=MASS,
    temp_target=300.0,
    kB=BOLTZMAN,
    dt=DT_PHYS,
    n_modes=N_MODES,
    seed=0,
    max_steps_noise=30000,
    log_temperature=False,   # IMPORTANT: keep False during training
).to(DEVICE)

# Initial state for Markovian GLE: add auxiliary s_i = 0
s0 = torch.zeros((N_MODES * N_ATOMS, 3), dtype=PRECISION, device=DEVICE)
y0_gle = torch.cat((v, q, s0), dim=0)

# Quick print of initial kernel
K0 = func_gle.memory_kernel_discrete(2000).detach().cpu()
plt.plot(K0, "r")
plt.xlabel("Time Steps")
plt.ylabel("Memory Kernel K[n]")
plt.show()

# Single evaluation run (no differentiation)
T = 3000
t_eval = torch.tensor(np.arange(0, DT_PHYS * T, DT_PHYS), dtype=PRECISION, device=DEVICE)

with torch.no_grad():
    func_gle.log_temperature = True
    func_gle.Temperature_log = []
    y_GLE_full = odeint_adjoint(func_gle, y0_gle, t_eval, method="euler")
    func_gle.log_temperature = False

# slice out v and q for post-processing
v_GLE = y_GLE_full[:, :N_ATOMS, :]
q_GLE = y_GLE_full[:, N_ATOMS:2 * N_ATOMS, :]

plt.plot(func_gle.Temperature_log, "r")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (K)")
plt.show()

print("Mean Temperature (after 800 steps): ", np.mean(func_gle.Temperature_log[800:]))

r_GLE, RDF_GLE = RDF_computer(q_GLE[-1000::10, :, :])
MSD_GLE = MSD_computer(q_GLE[:, :, :])
VACF_GLE = VACF_computer(v_GLE[:, :, :])

plt.plot(r_Langevin.cpu(), RDF_Langevin.cpu(), "k", label="Langevin")
plt.plot(r_GLE.cpu(), RDF_GLE.cpu(), "r", label="Markovian GLE (ExpKernel)")
plt.xlabel("Distance (Å)")
plt.ylabel("RDF")
plt.legend()
plt.show()

plt.plot(MSD_Langevin.cpu(), "k", label="Langevin")
plt.plot(MSD_GLE.cpu(), "r", label="Markovian GLE (ExpKernel)")
plt.xlabel("Time Steps")
plt.ylabel("MSD (Å²)")
plt.legend()
plt.show()

plt.plot(VACF_Langevin.cpu(), "k", label="Langevin")
plt.plot(VACF_GLE.cpu(), "r", label="Markovian GLE (ExpKernel)")
plt.xlabel("Time Steps")
plt.ylabel("VACF")
plt.legend()
plt.show()

# =============================================================================
# 3) Run Differentiable Training (VACF matching) with option A
#    - IMPORTANT changes vs your code:
#        * NO mode switching
#        * NO flipping v_list/w_list
#        * NO RNG inside forward (noise is fixed schedule)
#        * NO logging side effects during training
# =============================================================================

# %% 3.1 Pre-process (Reference System: All-Atom)
r_AA, RDF_AA = RDF_computer(torch.tensor(X[-1000::10], dtype=PRECISION, device=DEVICE))
VACF_AA = VACF_computer(torch.tensor(V, dtype=PRECISION, device=DEVICE))

print("RDF Shape: ", RDF_AA.shape)
print("VACF Shape: ", VACF_AA.shape)

plt.plot(RDF_AA.cpu())
plt.xlabel("Distance (Å)")
plt.ylabel("RDF")
plt.show()

plt.plot(VACF_AA.cpu())
plt.xlabel("Time Steps")
plt.ylabel("VACF")
plt.show()

# %% 3.2 Training Loop
optimizer = torch.optim.SGD(func_gle.parameters(), lr=1e-4, weight_decay=1e-5)

output_dir = f"Result/{MOLECULE}/"
optimization_plots_dir = output_dir + "optimization_plots/"
final_plot_dir = output_dir + "final_plot/"
model_dict_dir = output_dir + "model_dict/"

os.makedirs(optimization_plots_dir, exist_ok=True)
os.makedirs(final_plot_dir, exist_ok=True)
os.makedirs(model_dict_dir, exist_ok=True)

# Start training from last evaluation final state (or reset to y0_gle if you prefer)
y0_train = y_GLE_full[-1].detach()

for iter in range(300):

    # small no-grad warmup to move state forward without backprop-through-everything
    T_warm = 1
    t_warm = torch.tensor(np.arange(0, DT_PHYS * T_warm, DT_PHYS), dtype=PRECISION, device=DEVICE)
    with torch.no_grad():
        y_warm = odeint_adjoint(func_gle, y0_train, t_warm, method="euler")
    y0_train = y_warm[-1].detach()

    # differentiable rollout for VACF
    T_roll = VACF_computer.td_max + 10
    t_roll = torch.tensor(np.arange(0, DT_PHYS * T_roll, DT_PHYS), dtype=PRECISION, device=DEVICE)

    # IMPORTANT: no logging during training
    func_gle.log_temperature = False

    y_curr = odeint_adjoint(func_gle, y0_train, t_roll, method="euler")

    v_curr = y_curr[:, :N_ATOMS, :]
    q_curr = y_curr[:, N_ATOMS:2 * N_ATOMS, :]

    VACF_computer.ensemble_average = True
    VACF_computer.normalize = True
    VACF_curr = VACF_computer(v_curr)

    VACF_difference = VACF_curr - VACF_AA
    loss = VACF_difference.pow(2).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # advance starting point for next iter
    y0_train = y_curr[-1].detach()

    # logging / diagnostics
    lambdas, kappas = func_gle.get_params_positive()
    Ksum = func_gle.memory_kernel_discrete(2000).detach().sum().cpu().item()

    print(iter, loss.item(), "Ksum(2000)=", Ksum)
    print("lambdas:", lambdas.detach().cpu().numpy())
    print("kappas :", kappas.detach().cpu().numpy())

    # Save numerical values
    torch.save(VACF_curr.detach().cpu(), f"{optimization_plots_dir}VACF_curr_iter_{iter}.pt")
    torch.save(VACF_AA.detach().cpu(), f"{optimization_plots_dir}VACF_AA_iter_{iter}.pt")

    torch.save(lambdas.detach().cpu(), f"{optimization_plots_dir}lambdas_iter_{iter}.pt")
    torch.save(kappas.detach().cpu(), f"{optimization_plots_dir}kappas_iter_{iter}.pt")

    K_plot = func_gle.memory_kernel_discrete(2000).detach().cpu()
    torch.save(K_plot, f"{optimization_plots_dir}memory_kernel_iter_{iter}.pt")

    # plots
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(VACF_curr.detach().cpu().numpy(), "r")
    plt.plot(VACF_AA.detach().cpu().numpy(), "k--")
    plt.ylabel("VACF")

    plt.subplot(3, 1, 2)
    plt.plot(K_plot.numpy(), "r")
    plt.ylabel("K[n]")

    plt.subplot(3, 1, 3)
    plt.plot(lambdas.detach().cpu().numpy(), "bo--", label="lambdas")
    plt.plot(kappas.detach().cpu().numpy(), "ro--", label="kappas")
    plt.legend()
    plt.xlabel("Mode index")

    plt.tight_layout()
    plt.savefig(f"{optimization_plots_dir}figure_iter_{iter}.png")
    plt.show()

    # save state dict
    state_dict = func_gle.state_dict()
    torch.save(state_dict, f"{model_dict_dir}model_state_dict.pth")

# final save
torch.save(func_gle.state_dict(), f"{model_dict_dir}model_state_dict_iter_final.pth")

# =============================================================================
# 4) Reproduce Results (optional)
# =============================================================================
# model_path = f"{model_dict_dir}model_state_dict_iter_final.pth"
# func_gle.load_state_dict(torch.load(model_path, map_location=DEVICE))
# func_gle.eval()
# print("Model successfully loaded!")

# Example: run a fresh evaluation with fixed noise schedule
# func_gle.resample_noise(seed=0)
# func_gle.log_temperature = True
# func_gle.Temperature_log = []
# with torch.no_grad():
#     y_test = odeint_adjoint(func_gle, y0_gle, t_eval, method="euler")
# func_gle.log_temperature = False
