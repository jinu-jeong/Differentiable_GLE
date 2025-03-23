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
plt.rcParams.update({'font.size': 16})

# %% Configuration
CONFIG = {
    "CO2": {"mass": 44.0, "box": [40, 40, 40], "filter_length": 200},
    "H2O": {"mass": 18.0, "box": [40, 40, 40], "filter_length": 1000},
    "Confined_H2O": {"mass": 18.0, "box": [25.174493, 29.778001, 112], "filter_length": 1000},
}

MOLECULE = "CO2"

# %% Global Variables
DEVICE = "cuda:0"
PRECISION = torch.float
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
DT = 1  # femtoseconds (fs)

# %% Load Data
X = torch.tensor(np.load(f"./{MOLECULE}/AA/pos_COM.npy"))
V = torch.tensor(np.load(f"./{MOLECULE}/AA/vel_COM.npy"))

N_ATOMS = X.shape[1]
MASS = torch.full((N_ATOMS, 1), CONFIG[MOLECULE]["mass"]).to(DEVICE)

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

# %% Initialize System
positions, velocities, box = initialize_system(
    positions=(X[-1] % CONFIG[MOLECULE]["box"][0]),
    velocities=maxwell_boltzmann_velocity(MASS, temperature=300),
    box=CONFIG[MOLECULE]["box"]
)

q, cell, v = positions, box, velocities

# %% Load Potential
tabulated_data = np.loadtxt(f"./{MOLECULE}/CG/CG_CG.pot", skiprows=3)
tabulated_data = tabulated_data[~np.isnan(tabulated_data[:, -1])]
potential_gt = Tabulated(cell, tabulated_data).to(DEVICE)

# %% 1. Demonstration with Langevin Thermostat
# Langevin thermostat cannot accurately reproduce VACF
thermostat_Langevin = Langevin_TS(gamma=0.05).to(DEVICE)
simulation_Langevin = SDE(
    potential_gt, Thermostat=thermostat_Langevin, Temp_target=300,
    timestep=DT, TIMEFACTOR=TIMEFACTOR, mass=MASS,
    non_integrand_mask=None, saver=False
)
simulation_Langevin.force_mode = True

# Initial Conditions
y0 = torch.cat((v, q))
f0 = simulation_Langevin(0., y0)

# Simulation Preparation
T = 2000
t = torch.tensor(np.arange(0, DT / TIMEFACTOR * T, DT / TIMEFACTOR)).to(DEVICE)

# Equilibration Simulation
with torch.no_grad():
    y_Langevin = odeint_adjoint(simulation_Langevin, y0, t, method="euler")

plt.plot(simulation_Langevin.Temperature_log, 'k')
plt.xlabel("Time Steps")
plt.ylabel("Temperature (K)")
plt.show()

# Post-Processors
RDF_computer = RDF_computer(cell, DEVICE)
MSD_computer = MSD_computer(1000)
VACF_computer = VACF_computer(1000)
VACF_computer.normalize = True
VACF_computer.ensemble_average = True

# Post-Processing
r_Langevin, RDF_Langevin = RDF_computer(y_Langevin[::10, N_ATOMS:])
MSD_Langevin = MSD_computer(y_Langevin[:, N_ATOMS:])
VACF_Langevin = VACF_computer(y_Langevin[:, :N_ATOMS])

# Plot Results
plt.plot(r_Langevin.cpu(), RDF_Langevin.cpu(), 'k')
plt.xlabel("Distance (Å)")
plt.ylabel("RDF")
plt.show()

plt.plot(MSD_Langevin.cpu(), 'k')
plt.xlabel("Time Steps")
plt.ylabel("MSD (Å²)")
plt.show()

plt.plot(VACF_Langevin.cpu(), 'k')
plt.xlabel("Time Steps")
plt.ylabel("VACF")
plt.show()

# %% 2. GLE-Based Simulation
class GLE_TS(torch.nn.Module):
    def __init__(self, h=None, gamma_uniform=0.001, filter_length=None, BOLTZMAN=0.001987191):
        """Generalized Langevin Equation (GLE) thermostat."""
        super(GLE_TS, self).__init__()
        
        self.BOLTZMAN = BOLTZMAN
        self.filter_length = filter_length

        # Initialize h as a learnable parameter
        dummy = torch.full((self.filter_length,), gamma_uniform)
        self.h = torch.nn.Parameter(dummy[None, None, :])

        # Memory kernel and history placeholders
        self.v_list = None
        self.w_list = None
        self.w_history = []
        self.mode = "forward"

    def construct_memory(self, T, mass, dt):
        """Construct the memory kernel for the GLE thermostat."""
        h_padded = torch.nn.functional.pad(self.h, (0, self.h.size(2) - 1)).detach()
        self.theoretical_RACF = torch.nn.functional.conv1d(h_padded, self.h)
        self.memory_kernel = self.theoretical_RACF * mass[0, 0].item() * dt / (self.BOLTZMAN * T)
        self.memory_kernel_trapezoidal = self.memory_kernel.clone()

    def get_v_and_sample_w(self, v, dt):
        """Updates velocity and noise lists."""
        if self.v_list is None:
            num_atoms = v.numel() // 3
            self.v_list = torch.zeros(num_atoms * 3, 1, self.filter_length * 2 + 1, device=v.device)
            self.w_list = torch.zeros(num_atoms * 3, 1, self.filter_length * 2 + 1, device=v.device)

        if self.mode == "forward":
            self.v_list = torch.roll(self.v_list, -1, dims=2)
            self.v_list[:, :, -1] = v.view(-1, 1).clone().detach()
            self.w_list = torch.roll(self.w_list, -1, dims=2)
            self.w_list[:, :, -1] = torch.randn_like(v, device=v.device).view(-1, 1)
        else:
            self.v_list[:, :, -1] = torch.zeros_like(v, device=v.device).view(-1, 1)
            self.v_list = torch.roll(self.v_list, 1, dims=2)
            self.w_list[:, :, -1] = torch.zeros_like(v, device=v.device).view(-1, 1)
            self.w_list = torch.roll(self.w_list, 1, dims=2)

    def forward(self, v, T, dt, mass, t):
        """Compute velocity update using GLE thermostat."""
        self.construct_memory(T, mass, dt)
        self.get_v_and_sample_w(v, dt)

        friction = -torch.nn.functional.conv1d(self.v_list[:, :, -self.filter_length:], self.memory_kernel.flip(2))
        random = torch.nn.functional.conv1d(self.w_list[:, :, -self.filter_length:], self.h.flip(2).detach())
        delta_v_langevin = friction.view(v.shape) + random.view(v.shape)

        if len(self.w_history) == (self.filter_length * 2):
            self.w_history.pop(0)
        self.w_history.append(random.clone().detach().cpu().numpy().flatten())

        return delta_v_langevin

    def verify_filter_convolution(self):
        """Verify filter convolution by computing the autocorrelation function."""
        data = np.array(self.w_history)
        print(f"Data Shape: {data.shape}")

        RACF = []
        time_delta = self.filter_length + 1
        t0_list = range(0, len(data) - time_delta, 10)

        for t0 in t0_list:
            racf = (data[t0:t0 + time_delta] * data[t0]).mean(1)
            RACF.append(racf)

        RACF = np.mean(RACF, axis=0)

        # Plot Results
        plt.figure(figsize=(8, 4))
        plt.plot(self.h.clone().detach().cpu().flatten(), 'bo--')
        plt.title("Filter")
        plt.xlabel("Filter Index")
        plt.ylabel("Filter Value")
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(self.memory_kernel.clone().detach().cpu().flatten(), 'bo--')
        plt.title("Memory Kernel")
        plt.xlabel("Time Steps")
        plt.ylabel("Memory Kernel Value")
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(self.theoretical_RACF.clone().detach().cpu().flatten(), 'bo--', label="Theoretical")
        plt.plot(RACF, 'ro--', label="Computed")
        plt.title("Theoretical RACF")
        plt.xlabel("Time Steps")
        plt.ylabel("RACF")
        plt.legend()
        plt.show()

# Define GLE Thermostat
Thermostat_GLE = GLE_TS(
    gamma_uniform=0.001, 
    filter_length=CONFIG[MOLECULE]["filter_length"]
).to(DEVICE)

# Define Differential Equation
func = SDE(
    potential_gt, 
    Thermostat=Thermostat_GLE, 
    Temp_target=300, 
    timestep=DT, 
    TIMEFACTOR=TIMEFACTOR, 
    mass=MASS, 
    non_integrand_mask=None, 
    saver=False
)
func.force_mode = True

# Set Initial Conditions
q, cell, v = positions, box, velocities
y0 = torch.cat((v * 0, q))  # Zero initial velocities
f0 = func(0., y0)

# Print GLE Properties
print("Filter: ", Thermostat_GLE.h)
print("Memory Kernel: ", Thermostat_GLE.memory_kernel)
print("Mean Gamma: ", Thermostat_GLE.memory_kernel.sum().item())
print("Relaxation Time: ", 
      Thermostat_GLE.memory_kernel.sum().item() / Thermostat_GLE.memory_kernel.flatten()[0].item())

# Verify Filter Convolution
Thermostat_GLE.verify_filter_convolution()

# Single Simulation Run (No Differentiability)
T = 3000  # Simulation time
t = torch.tensor(np.arange(0, DT / TIMEFACTOR * T, DT / TIMEFACTOR)).to(DEVICE)
y0 = torch.cat((v, q))

# Run Simulation with GLE Thermostat
with torch.no_grad():
    y_GLE = odeint_adjoint(func, y0, t, method="euler")

# Store Final State
y0 = y_GLE[-1]

# Plot Temperature Log
plt.plot(func.Temperature_log[-T:], 'r')
plt.xlabel("Time Steps")
plt.ylabel("Temperature (K)")
plt.show()

# Print Mean Temperature After Equilibration
print("Mean Temperature: ", np.mean(func.Temperature_log[800:]))

# Verify GLE Filter Convolution Again
Thermostat_GLE.verify_filter_convolution()

# Post-Processing
r_GLE, RDF_GLE = RDF_computer(y_GLE[-1000::10, N_ATOMS:])
MSD_GLE = MSD_computer(y_GLE[:, N_ATOMS:])
VACF_GLE = VACF_computer(y_GLE[:, :N_ATOMS])

# Plot Comparisons
plt.plot(r_Langevin.cpu(), RDF_Langevin.cpu(), 'k', label="Langevin")
plt.plot(r_GLE.cpu(), RDF_GLE.cpu(), 'r', label="GLE")
plt.xlabel("Distance (Å)")
plt.ylabel("RDF")
plt.legend()
plt.show()

plt.plot(MSD_Langevin.cpu(), 'k', label="Langevin")
plt.plot(MSD_GLE.cpu(), 'r', label="GLE")
plt.xlabel("Time Steps")
plt.ylabel("MSD (Å²)")
plt.legend()
plt.show()

plt.plot(VACF_Langevin.cpu(), 'k', label="Langevin")
plt.plot(VACF_GLE.cpu(), 'r', label="GLE")
plt.xlabel("Time Steps")
plt.ylabel("VACF")
plt.legend()
plt.show()

# %% 3. Run Differentiable GLE
## 3.1 Pre-process
# Reference System (All-Atom)
r_AA, RDF_AA = RDF_computer(torch.tensor(X[-1000::10]).to(DEVICE))
VACF_AA = VACF_computer(torch.tensor(V).to(DEVICE))

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



# %% 3.1 Training Loop for Dynamics Optimization (Optional if Trained)
# Dynamics Optimization Setup
optimizer = torch.optim.Adam(func.parameters(), lr=3e-4, weight_decay=1e-6, betas=[0.1, 0.1])

# Create Necessary Directories
output_dir = f"Result/{MOLECULE}/"
optimization_plots_dir = output_dir + "optimization_plots/"
final_plot_dir = output_dir + "final_plot/"
model_dict_dir = output_dir + "model_dict/"

os.makedirs(optimization_plots_dir, exist_ok=True)
os.makedirs(final_plot_dir, exist_ok=True)
os.makedirs(model_dict_dir, exist_ok=True)


# We recommend transfer learning for confined water, as it takes too much effort if learning from scratch.
if MOLECULE=="Confined_H2O":
    model_path = "H2O/model_dict/model_state_dict_iter_final.pth"
    func.load_state_dict(torch.load(model_path, map_location=DEVICE))

# Optimization core
for iter in range(300):
    # Mini-Batch Optimization
    for minibatch in range(1):
        T = max([Thermostat_GLE.filter_length, VACF_computer.td_max]) + 10
        t = torch.tensor(np.arange(0, DT / TIMEFACTOR * T, DT / TIMEFACTOR)).to(DEVICE)
        
        y_curr = odeint_adjoint(func, y0, t, method="euler")
        y0 = y_curr[Thermostat_GLE.filter_length + 1].detach()

        # Compute VACF
        VACF_curr = VACF_computer(y_curr[:, :N_ATOMS])

        # Compute Loss
        VACF_difference = VACF_curr - VACF_AA
        loss = VACF_difference.pow(2).sum()

        # Backpropagation
        optimizer.zero_grad()
        Thermostat_GLE.mode = "forward"
        loss.backward(retain_graph=True)

        # Flip Velocity and Noise History for Consistency
        func.Thermostat.v_list[:, :, -Thermostat_GLE.filter_length:] = \
            func.Thermostat.v_list[:, :, -Thermostat_GLE.filter_length:].flip(2)
        func.Thermostat.w_list[:, :, -Thermostat_GLE.filter_length:] = \
            func.Thermostat.w_list[:, :, -Thermostat_GLE.filter_length:].flip(2)

    # Update Parameters
    optimizer.step()

    # Logging
    memory_kernel_sum = Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten().sum()
    print(f"Iteration {iter}: Loss = {loss.item()}, Memory Kernel Sum = {memory_kernel_sum}")
    print("Consistency Check: ", torch.equal(
        y_curr[Thermostat_GLE.filter_length, :N_ATOMS], 
        func.Thermostat.v_list[:, :, -1].view(-1, 3)
    ))

    # Save Numerical Values
    torch.save(VACF_curr.detach().cpu(), f"{optimization_plots_dir}VACF_curr_iter_{iter}.pt")
    torch.save(VACF_AA.detach().cpu(), f"{optimization_plots_dir}VACF_AA_iter_{iter}.pt")
    torch.save(Thermostat_GLE.h.clone().detach().cpu().flatten(), f"{optimization_plots_dir}filter_iter_{iter}.pt")
    torch.save(Thermostat_GLE.h.grad.clone().detach().cpu().flatten(), f"{optimization_plots_dir}filter_gradient_iter_{iter}.pt")
    torch.save(Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten(), f"{optimization_plots_dir}memory_kernel_iter_{iter}.pt")

    if iter % 10 == 0:
        plt.figure(figsize=(10, 8))

        plt.subplot(4, 1, 1)
        plt.plot(VACF_curr.detach().cpu().numpy(), 'r', label="Current VACF")
        plt.plot(VACF_AA.detach().cpu().numpy(), 'k--', label="Reference VACF")
        plt.xlabel("Time Steps")
        plt.ylabel("VACF")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(Thermostat_GLE.h.clone().detach().cpu().flatten(), 'r', label="Filter")
        plt.xlabel("Filter Index")
        plt.ylabel("Filter Value")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(Thermostat_GLE.h.grad.clone().detach().cpu().flatten(), 'g--', label="Filter Gradient")
        plt.xlabel("Filter Index")
        plt.ylabel("Gradient Value")
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten(), 'r', label="Memory Kernel")
        plt.xlabel("Time Steps")
        plt.ylabel("Memory Kernel Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Equilibration
    T = min([len(Thermostat_GLE.memory_kernel), abs(VACF_computer.td_max-len(Thermostat_GLE.memory_kernel))])
    t = torch.tensor(np.arange(0, DT / TIMEFACTOR * T, DT / TIMEFACTOR)).to(DEVICE)
    with torch.no_grad():
        y_curr = odeint_adjoint(func, y0, t, method="euler")
    y0 = y_curr[-1].detach()

# Cancel annotation below if you want to use your own model.
# torch.save(func.state_dict(), f"{model_dict_dir}model_state_dict_iter_final.pth")

# %% 3.2 Reproduce Results
# Load the Saved Model State Dictionary
model_path = f"{model_dict_dir}model_state_dict_iter_final.pth"
func.load_state_dict(torch.load(model_path, map_location=DEVICE))
func.eval()
print("Model successfully loaded!")

# Define Long Simulation Parameters
T_long = 3000  # Number of time steps
t_long = torch.tensor(np.arange(0, DT / TIMEFACTOR * T_long, DT / TIMEFACTOR)).to(DEVICE)

# Set Initial Condition
y0 = torch.cat((v, q))  # Use previously initialized velocities and positions

# Run the Long-Time Simulation
with torch.no_grad():
    y_CG = odeint_adjoint(func, y0, t_long, method="euler")

# Store Final State
y0 = y_CG[-1]
print("Long-time simulation completed!")

# Compute RDF and VACF for Long Simulation
r_CG, RDF_CG = RDF_computer(y_CG[-1000::10, N_ATOMS:])
VACF_CG = VACF_computer(y_CG[:, :N_ATOMS])

# Save Results
torch.save(r_CG.cpu(), f"{final_plot_dir}Radial_distance.pt")
torch.save(RDF_AA.detach().cpu(), f"{final_plot_dir}RDF_AA.pt")
torch.save(VACF_AA.detach().cpu(), f"{final_plot_dir}VACF_AA.pt")
torch.save(RDF_CG.detach().cpu().flatten(), f"{final_plot_dir}RDF_CG.pt")
torch.save(VACF_CG.detach().cpu().flatten(), f"{final_plot_dir}VACF_CG.pt")

# Load
RDF_AA = torch.load(f"{final_plot_dir}RDF_AA.pt")
VACF_AA = torch.load(f"{final_plot_dir}VACF_AA.pt")
r_CG = torch.load(f"{final_plot_dir}Radial_distance.pt")
RDF_CG = torch.load(f"{final_plot_dir}RDF_CG.pt")
VACF_CG = torch.load(f"{final_plot_dir}VACF_CG.pt")



plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(r_CG.cpu(), RDF_CG.cpu(), 'r', label="CG RDF")
plt.plot(r_AA.cpu(), RDF_AA.cpu(), 'k--', label="AA RDF")
plt.xlabel("Radial distance [Å])")
plt.ylabel("RDF")
plt.legend()
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(VACF_CG.detach().cpu().numpy(), 'r', label="CG VACF")
plt.plot(VACF_AA.detach().cpu().numpy(), 'k--', label="AA VACF")
plt.xlabel("Time Delay [fs]")
plt.ylabel("VACF")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(Thermostat_GLE.h.clone().detach().cpu().flatten(), 'r', label="Filter")
plt.xlabel("Time Delay [fs]")
plt.ylabel("Filter Value")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten(), 'r', label="Memory Kernel")
plt.xlabel("Time Delay [fs]")
plt.ylabel("Memory Kernel")
plt.legend()
plt.tight_layout()
plt.show()


# %%
