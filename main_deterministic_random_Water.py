#%%
"""
Learning Kernel

Target system : LJ

"""
#%%
import os
os.chdir('/workspace/jinu/GLE')


import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn


from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann

X = torch.tensor(np.load("./Water/AA/dump.npy"))
V = torch.tensor(np.load("./Water/AA/vel.npy"))

global Natom, precision, device, mass, TIMEFACTOR, dt
Natom = X.shape[1]
precision = torch.float
device = "cuda:0"
mass = torch.full((Natom,1), 18.0).to(device)
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
dt = 1 #fs


system = System(Natom, nreplicas=1, precision=precision, device=device)
system.set_positions(X[-1][:,:,None] % 40.0)
system.set_box(np.array([40.0,40.0,40.0]))
system.set_velocities(maxwell_boltzmann(mass, T=300, replicas=1))

#%% Select force
from force import *

Tabulated_data = np.loadtxt("./Water/CG/"+"CG_CG.pot", skiprows=3)
nan_rows = np.isnan(Tabulated_data[:, -1])
Tabulated_data = Tabulated_data[~nan_rows]
potential_GT = Tabulated(system.box[0], Tabulated_data).to(device)

Thermostat_LE = Langevin_TS(gamma = 0.05).to(device)
#Thermostat = Langevin_TS(gamma = 0.05, Trainable=True).to(device)

from utility import *
func= SDE(potential_GT, Thermostat=Thermostat_LE, Temp_target=300, timestep=dt, TIMEFACTOR=TIMEFACTOR, mass=mass, non_integrand_mask=None, saver=False)
func.force_mode=True

#%% Define initial condition
q, cell, v = system.pos[0], system.box[0], system.vel[0]
y0 = torch.concatenate((v,q))
f0 = func(0., y0)

#%% Simulation prep
T = 2000

from torchdiffeq import odeint_adjoint
y0 = torch.concatenate((v,q))
t = (torch.tensor(np.arange(0, dt/TIMEFACTOR*T, dt/TIMEFACTOR))).to(device)

RDF = RDF_computer(cell, device)
MSD = MSD_computer(1000)
VACF = VACF_computer(500)


#%% Equilibration
with torch.no_grad():
    y_vanilla = odeint_adjoint(func, y0, t, method="euler")

plt.plot(func.Temperature_log, 'k')

#%% compute RDF

r_vanilla, RDF_vanilla = RDF(y_vanilla[::10,Natom:])
MSD_vanilla = MSD(y_vanilla[:,Natom:])
VACF_vanilla = VACF(y_vanilla[:,:Natom])



plt.plot(r_vanilla.cpu(), RDF_vanilla.cpu(), 'k')
plt.show()
plt.plot(MSD_vanilla.cpu(), 'k')
plt.show()
plt.plot(VACF_vanilla.cpu(), 'k')
plt.show()


#%%
#%%
#%%

class GLE_TS(torch.nn.Module):
    def __init__(self, h=None, gamma_uniform=0.001, filter_length=None, BOLTZMAN=0.001987191):
        super(GLE_TS, self).__init__()
        
        self.BOLTZMAN = BOLTZMAN
        self.filter_length = filter_length
        
        # Initialize h as a learnable parameter if Trainable is True
        #self.h = torch.nn.Parameter(torch.full((self.filter_length ,), gamma_uniform)[None,None,:])

        dummy = torch.full((self.filter_length ,), gamma_uniform)
        self.h = torch.nn.Parameter(dummy[None,None,:])
        

        #self.h = torch.nn.Parameter(torch.tensor([0.5, 0.0])[None,None,:])
        #self.filter_length = 2

        #dummy = torch.linspace(0, 2*np.pi, self.filter_length)
        #dummy = gamma_uniform*torch.exp(-100*dummy)
        #self.h = torch.nn.Parameter(dummy[None,None,:])

        # Placeholder for the memory kernel
        #self.construct_memory(, mass, 0.1)
        self.v_list = None
        self.w_list = None

        self.w_history = []

        self.mode = "forward"

    def construct_memory(self, T, mass, dt):
        h_padded = torch.nn.functional.pad(self.h, (0, self.h.size(2)-1)).detach()
        # TODO : remove hack: mass[0]
        self.theoretical_RACF = torch.nn.functional.conv1d(h_padded, self.h) # first time factor for sigma sqaure, second time factor for integration
        self.memory_kernel = self.theoretical_RACF * mass[0,0].item() * dt / (self.BOLTZMAN * T)
        
        self.memory_kernel_trapezoidal = self.memory_kernel.clone()
        #self.memory_kernel_trapezoidal[..., 0] *= 0.5
        #self.memory_kernel_trapezoidal[..., -1] *= 0.5 # This was zero and will be zero.

    def get_v_and_sample_w(self, v, dt):
        # initial
        if self.v_list == None:
            Natom = len(v.flatten())//3
            self.v_list = torch.zeros(Natom*3, 1, self.filter_length*2 + 1 , device=v.device)
            self.w_list = torch.zeros(Natom*3, 1, self.filter_length*2 + 1 , device=v.device)
        
        
        
        # Update velocity and noise lists
        if self.mode=="forward":
            self.v_list = torch.roll(self.v_list, -1, dims=2)
            self.v_list[:,:,-1] = v.view(-1,1).clone().detach()
            self.w_list = torch.roll(self.w_list, -1, dims=2)
            self.w_list[:,:,-1] = torch.randn_like(v, device=v.device).view(-1,1)
        else:
            self.v_list[:,:,-1] = torch.zeros_like(v, device=v.device).view(-1,1)
            self.v_list = torch.roll(self.v_list, 1, dims=2)
            self.w_list[:,:,-1] = torch.zeros_like(v, device=v.device).view(-1,1)
            self.w_list = torch.roll(self.w_list, 1, dims=2)
            
        

    def forward(self, v, T, dt, mass, t):
        # Construct the memory and update velocity and noise history
        self.construct_memory(T, mass, dt)
        self.get_v_and_sample_w(v, dt)


        friction = -torch.nn.functional.conv1d(self.v_list[:,:,-self.filter_length:], self.memory_kernel.flip(2))
        random = torch.nn.functional.conv1d(self.w_list[:,:,-self.filter_length:], self.h.flip(2).detach())
        delta_v_langevin = friction.view(v.shape) + random.view(v.shape)
        
        if len(self.w_history) == (self.filter_length*2):
            self.w_history.pop(0)
        self.w_history.append(random.clone().detach().cpu().numpy().flatten())
        
        return delta_v_langevin

    
    def verify_filter_convolution(self):
        data = np.array(self.w_history)
        print(data.shape)
        RACF = []
        td = self.filter_length+1
        t0_list = range(0, len(data)-td, 10)

        for t0 in t0_list:
            racf = (data[t0:t0+td]*data[t0]).mean(1)
            RACF.append(racf)

        RACF = np.mean(RACF, axis=0)

        plt.plot(self.h.clone().detach().cpu().flatten(), 'bo--')
        plt.title("Filter")
        plt.show()
        plt.plot(self.memory_kernel.clone().detach().cpu().flatten(), 'bo--')
        plt.title("Memory Kernel")
        plt.show()
        plt.plot(self.theoretical_RACF.clone().detach().cpu().flatten(), 'bo--')
        plt.plot(RACF, 'ro--')
        plt.title("Theoretical RACF")
        plt.show()

#%% GLE based simulation
Thermostat_GLE = GLE_TS(gamma_uniform=0.001, filter_length=1000).to(device)

func= SDE(potential_GT, Thermostat=Thermostat_GLE, Temp_target=300, timestep=dt, TIMEFACTOR=TIMEFACTOR, mass=mass, non_integrand_mask=None, saver=False)
func.force_mode=True
q, cell, v = system.pos[0], system.box[0], system.vel[0]
y0 = torch.concatenate((v*0,q))
f0 = func(0., y0)

print("Filter: ", Thermostat_GLE.h)
print("Memory kernel: ", Thermostat_GLE.memory_kernel)
print("Mean gamma: ", Thermostat_GLE.memory_kernel.sum().item())
print("Relaxation time : ", Thermostat_GLE.memory_kernel.sum().item() / Thermostat_GLE.memory_kernel.flatten()[0].item())
Thermostat_GLE.verify_filter_convolution()

#%%
T = 3000
t = (torch.tensor(np.arange(0, dt/TIMEFACTOR*T, dt/TIMEFACTOR))).to(device)
y0 = torch.concatenate((v,q))

with torch.no_grad():
    y_GLE= odeint_adjoint(func, y0, t, method="euler")

y0 = y_GLE[-1]
plt.plot(func.Temperature_log[-T:], 'r')
plt.show()
print(np.mean(func.Temperature_log[800:]))
#%%
Thermostat_GLE.verify_filter_convolution()

#%%
r_GLE, RDF_GLE = RDF(y_GLE[-1000::10,Natom:])
MSD_GLE = MSD(y_GLE[:,Natom:])
VACF_GLE = VACF(y_GLE[:,:Natom])


plt.plot(r_vanilla.cpu(), RDF_vanilla.cpu(), 'k')
plt.plot(r_GLE.cpu(), RDF_GLE.cpu(), 'r')
plt.show()
plt.plot(MSD_vanilla.cpu(), 'k')
plt.plot(MSD_GLE.cpu(), 'r')
plt.show()
plt.plot(VACF_vanilla.cpu(), 'k')
plt.plot(VACF_GLE.cpu(), 'r')
plt.show()
#%%
#%%
#%%
#%%
#%% Dynamics optimization
VACF = VACF_computer(1000)
VACF.ensemble_average = True; VACF.normalize=True
VACF_AA = VACF(torch.tensor(V).to(device))

r_AA, RDF_AA = RDF(torch.tensor(X[-1000::10,]).to(device))

#VACF_AA[50:] *= 1.5
#VACF_AA[50:] *= 1.5
print(VACF_AA.shape)
plt.plot(VACF_AA.cpu())
#torch.save(VACF_AA, "VACF_AA.pth")
#VACF_AA = torch.load("VACF_AA.pth")
#%%Dynamics optimization

optimizer = torch.optim.SGD(func.parameters(), lr=1e-4, weight_decay = 1e-6)
#optimizer = torch.optim.RMSprop(func.parameters(), lr=0.001, weight_decay = 1e-6)
#optimizer = torch.optim.SGD(func.parameters(), lr=1e-4)
#%%
VACF = VACF_computer(1000)

for iter in range(300):
    if iter == 100:
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-4, weight_decay = 1e-6)
    elif iter == 200:
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-5, weight_decay = 1e-6)

    T = 1
    t = (torch.tensor(np.arange(0, dt/TIMEFACTOR*T, dt/TIMEFACTOR))).to(device)
    with torch.no_grad():
        y_curr= odeint_adjoint(func, y0, t, method="euler")
    y0 = y_curr[-1].detach()


    for minibatch in range(1):
        T = VACF.td_max+10
        t = (torch.tensor(np.arange(0, dt/TIMEFACTOR*T, dt/TIMEFACTOR))).to(device)
        y_curr= odeint_adjoint(func, y0, t, method="euler")
        y0 = y_curr[-1].detach()

        # Post processing
        VACF.ensemble_average = True; VACF.normalize=True
        VACF_curr = VACF(y_curr[:,:Natom])


        VACF_difference = VACF_curr - VACF_AA
        VACF_sum_difference = VACF_curr.sum() - VACF_AA.sum()
        #Effective_gamma = Thermostat_GLE.h.sum()
        loss = VACF_difference.pow(2).sum() #+ Thermostat_GLE.h.abs().sum()*0.2# + torch.relu(Effective_gamma-0.5)

        optimizer.zero_grad()
        Thermostat_GLE.mode = "forward"
        loss.backward(retain_graph=True)
        #Thermostat_GLE.mode = "backward"
        #torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        func.Thermostat.v_list[:,:,-1000:] = func.Thermostat.v_list[:,:,-1000:].flip(2)
        func.Thermostat.w_list[:,:,-1000:] = func.Thermostat.w_list[:,:,-1000:].flip(2)


    optimizer.step()


    print(iter, loss.item(), Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten().sum())
    print("Consistency", torch.equal(y_curr[-2, :Natom], func.Thermostat.v_list[:,:,-1].view(-1,3)))

    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(VACF_curr.detach().cpu().numpy(), 'r')
    plt.plot(VACF_AA.detach().cpu().numpy(), 'k--')

    plt.subplot(4, 1, 2)
    filter = Thermostat_GLE.h.clone().detach().cpu().flatten()
    plt.plot(filter, 'r')

    plt.subplot(4, 1, 3)
    filter_gradient = Thermostat_GLE.h.grad.clone().detach().cpu().flatten()
    plt.plot(filter_gradient, 'g--')
    
    plt.subplot(4, 1, 4)
    plt.plot(Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten(), 'r')

    plt.savefig(f'SPCE/figure_iter_{iter}.png')
    plt.show()

    # save state dict
    state_dict = func.state_dict()
    torch.save(state_dict, 'model_state_dict.pth')

import pickle
plot_dict = {"VACF_GLE" : VACF_curr.detach().cpu().numpy(), "VACF_AA" : VACF_AA.detach().cpu().numpy(), "filter" : filter.numpy(), "filter_gradient" : filter_gradient.numpy(), "memory_kernel" : Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten().numpy()}
with open('SPCE/plot_dict.pkl', 'wb') as f:
    pickle.dump(plot_dict, f)

with open('SPCE/plot_dict.pkl', 'rb') as f:
    plot_dict = pickle.load(f)
#%%


T = 10000
t = (torch.tensor(np.arange(0, dt/TIMEFACTOR*T, dt/TIMEFACTOR))).to(device)

with torch.no_grad():
    y_curr= odeint_adjoint(func, y0, t, method="euler")

y0 = y_curr[-1].detach()

VACF_curr = VACF(y_curr[:,:Natom])
plt.plot(VACF_curr.detach().cpu().numpy(), 'r')
plt.plot(VACF_AA.detach().cpu().numpy(), 'k--')

write_xyz_dump("/oden/jjeong/SPCE.xyz", [7]*Natom, atom_types_map, y_curr[:,Natom:,:], cell)


#%%
#%%
VACF_curr = VACF(y_curr[:VACF.td_max,:Natom])
#%%

y0 = y_curr[-1].detach()








#%%
path_to_saved_dict = 'model_state_dict.pth'
if os.path.exists(path_to_saved_dict):
    func.load_state_dict(torch.load(path_to_saved_dict))

#%%
figure_directory = "/oden/jjeong/figure"
os.makedirs(figure_directory, exist_ok=True)


# %%
