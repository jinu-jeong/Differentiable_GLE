import numpy as np
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR

#%%
def gaussian_smearing(centered, sigma):
    return 1/(sigma*(2*np.pi)**0.5)*torch.exp(-0.5*(centered/sigma)**2)


#batch_size, state_size, brownian_size = 1, Natom*6, 1

class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'
    
    


    def __init__(self, potential, Thermostat, Temp_target, timestep, TIMEFACTOR, mass, non_integrand_mask, saver=False, append_dict=None):
        super().__init__()
        self.potential = potential
        self.Thermostat = Thermostat

        self.Temp_target = Temp_target
        self.dt = timestep / TIMEFACTOR
        
        self.mass = mass

        self.non_integrand_mask = non_integrand_mask
        
        self.force_mode = False
        
        if saver==True:
            self.saver = True
            self.pos_saver = []
            self.force_saver = []
        else:
            self.saver = False
        
        if append_dict != None:
            self.append_dict = append_dict
        else:
            self.append_dict = None

        self.Temperature_log = []
            

    def forward(self, t, state):
        original_state_shape = state.shape
        
        state = state.view(-1,3)
        Natom = len(state)//2
        
        with torch.set_grad_enabled(True):
            #3N
            v0 = state[:Natom].requires_grad_(True)            
            q0 = state[Natom:].requires_grad_(True)
            
            
            ######## Thermostat
            
            if self.Thermostat!=None:
                dv_dt_thermostat = self.Thermostat(v0, self.Temp_target, self.dt, self.mass, t = t)
            else:
                dv_dt_thermostat = 0
            
            ########
                        
            ######### first VV
            if self.force_mode==False:
                u0 = self.potential(q0).sum()
                f0 = -compute_grad(inputs=q0, output=u0)
            else:
                f0 = self.potential(q0)
            a0 = f0/self.mass
            
            if self.saver==True:
                self.pos_saver.append(q0.clone().detach().cpu())
                self.force_saver.append(f0.clone().detach().cpu())
            
            if self.append_dict!=None:
                write_xyz_dump(self.append_dict["file_name"], self.append_dict["atom_types"], self.append_dict["atom_types_map"], q0.unsqueeze(0), self.potential.cell, mode = "a")
            
            dqdt = v0 + 0.5*a0*self.dt
            q1 = q0 + dqdt *self.dt
            

            
            ######### second VV
            if self.force_mode==False:
                u1 = self.potential(q1).sum()
                f1 = -compute_grad(inputs=q1, output=u1)
            else:
                f1 = self.potential(q1)

            a1 = f1/self.mass
            dvdt = 0.5*(a0 + a1) + dv_dt_thermostat
            
            
            ######### Integrate only selected DOFs
            
        if self.non_integrand_mask!=None:
            dvdt[self.non_integrand_mask] = 0
            dqdt[self.non_integrand_mask] = 0
            
            
        
        KE = (self.mass*(v0.pow(2))).sum()/2
        T = kinetic_to_temp(KE, Natom)
        self.Temperature_log.append(T.item())

        return torch.concatenate((dvdt, dqdt)).view(original_state_shape)
"""
    def forward(self, t, state):
        return self.f(t, state)

    def g(self, t, state):
        
        
        return torch.zeros_like(state).unsqueeze(2).to(state.device)*0

"""
#%%


#%%
def kinetic_energy(masses, vel):
    Ekin = torch.sum(0.5 * torch.sum(vel * vel, dim=2,
                     keepdim=True) * masses, dim=1)
    return Ekin


def maxwell_boltzmann(masses, T, replicas=1):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            torch.sqrt(T * BOLTZMAN / masses) *
            torch.randn((natoms, 3)).type_as(masses)
        )

    return torch.stack(velocities, dim=0)


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos = pos + vel * dt + 0.5 * accel * dt * dt
    vel = vel + 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, mass, dt, T, device, gamma=0.1):

    coeff = torch.sqrt(2.0 * gamma / mass * BOLTZMAN * T * dt).to(device)
    csi = torch.randn_like(vel, device=device) * coeff
    delta_vel = -gamma * vel * dt + csi
    return delta_vel


def nose_hoover_thermostat(velocities, xi, Q, target_temperature, dt):
    # Assuming reduced units where Boltzmann's constant is 1. Adjust if needed.
    kB = 1.0
    N = velocities.size(0)
    kinetic_energy = 0.5 * torch.sum(velocities**2)

    # Update xi using its equation of motion
    dxi = dt * (kinetic_energy / N - kB * target_temperature) / Q
    xi += dxi

    # Update velocities considering the xi variable
    scaling_factor = torch.exp(-xi * dt)
    velocities *= scaling_factor

    return velocities, xi





def compute_grad(inputs, output, create_graph=True, retain_graph=True):
    """Compute gradient of the scalar output with respect to inputs.

    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 

    Returns:
        torch.Tensor: gradients with respect to each input component 
    """

    assert inputs.requires_grad

    gradspred, = torch.autograd.grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                                     create_graph=create_graph, retain_graph=retain_graph)

    return gradspred










def generate_nbr_list(coordinates, lattice_matrix, cutoff=6.0):
    
    lattice_matrix_diag = torch.diag(lattice_matrix).view(1, 1, -1)
    
    device = coordinates.device
    displacement = (
        coordinates[..., None, :, :] - coordinates[..., :, None, :])

    # Transform distance using lattice matrix inverse
    offsets = ((displacement+lattice_matrix_diag/2) // lattice_matrix_diag).detach()
    
    
    # Apply periodic boundary conditions
    displacement = displacement - offsets*(lattice_matrix_diag)

    # Compute squared distances and create mask for cutoff
    squared_displacement = torch.triu(displacement.pow(2).sum(-1))
    
    within_cutoff = (squared_displacement < cutoff **2) & (squared_displacement != 0)
    neighbor_indices = torch.nonzero(within_cutoff.to(torch.long), as_tuple=False)
    
    
    offsets = offsets[neighbor_indices[:, 0], neighbor_indices[:, 1], :]

    # Compute unit vectors and actual distances    
    unit_vectors = displacement[neighbor_indices[:,0], neighbor_indices[:,1]]
    magnitudes = squared_displacement[neighbor_indices[:,0], neighbor_indices[:,1]].sqrt()
    
    
    
    unit_vectors = unit_vectors / magnitudes.view(-1, 1)
    
    actual_distances = magnitudes[:, None]

    return neighbor_indices.detach(), offsets, actual_distances, -unit_vectors






class density_computer(torch.nn.Module):
    def __init__(self, cell, z_mask, axis, device, dr = 0.1):
        super(density_computer, self).__init__()
        self.cell = cell

        self.dr = dr
        self.device = device
        self.z_mask = z_mask
        self.axis = axis
        
        self.r_list = []
        self.r_list += [torch.arange(0.5*self.dr, cell[0,0] - self.dr*2, self.dr, device=self.device)]
        self.r_list += [torch.arange(0.5*self.dr, cell[1,1] - self.dr*2, self.dr, device=self.device)]
        self.r_list += [torch.arange(0.5*self.dr, cell[2,2] - self.dr*2, self.dr, device=self.device)]

    def forward(self, Traj):

        target = Traj[:,self.z_mask,self.axis]
        r_list = self.r_list[self.axis]

        for t, q in enumerate(target):
            pdist_gaussian = gaussian_smearing(
                q.unsqueeze(1)-r_list, self.dr).sum(0)*self.dr
            if t == 0:
                Pdist_gaussian = pdist_gaussian
            else:
                Pdist_gaussian += pdist_gaussian

        Pdist_gaussian /= (t+1)


        return r_list, Pdist_gaussian / Pdist_gaussian.sum()





class RDF_computer(torch.nn.Module):
    def __init__(self, cell, device, L_max=None):
        super(RDF_computer, self).__init__()
        self.cell = cell

        if L_max == None:
            self.L_max = self.cell[0, 0]/2
        else:
            self.L_max = L_max
        self.dr = 0.1
        self.device = device
        self.r_list = torch.arange(
            0.5*self.dr, self.L_max - self.dr*2, self.dr, device=self.device)


    def forward(self, Traj):
        Hist = []
        for t, q in enumerate(Traj):
            nbr_list, offsets, pdist, unit_vector = generate_nbr_list(
                q, self.cell, cutoff=self.L_max)

            pdist_gaussian = gaussian_smearing(
                pdist-self.r_list, self.dr).sum(0)*self.dr
            if t == 0:
                Pdist_gaussian = pdist_gaussian
            else:
                Pdist_gaussian += pdist_gaussian

        Pdist_gaussian /= (t+1)

        v = 4 * np.pi / 3 * ((self.r_list+0.5*self.dr) **
                             3 - (self.r_list-0.5*self.dr)**3)
        natom = len(Traj[0])
        bulk_density = (natom-1)/(torch.det(self.cell))
        gr = Pdist_gaussian/v * (torch.det(self.cell))/(natom-1)/natom*2

        return self.r_list, gr



class MSD_computer(torch.nn.Module):
    def __init__(self, td_max):
        super(MSD_computer, self).__init__()
        self.td_max = td_max

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(len(x), -1, 3)

        t0_list = list(range(0, len(x)-self.td_max, 100))

        MSD = torch.zeros(self.td_max, device=x.device)

        for t0 in t0_list:
            MSD += (x[t0:t0+self.td_max] - x[t0]).pow(2).mean((1, 2))
        MSD /= len(t0_list)
        return MSD



class VACF_computer(torch.nn.Module):
    def __init__(self, td_max, ensemble_average = True, normalize=True):
        super(VACF_computer, self).__init__()
        self.td_max = td_max
        self.ensemble_average = ensemble_average
        self.normalize = normalize

    def forward(self, v):
        if len(v.shape) == 2:
            v = v.view(len(v), -1, 3)

        t0_list = list(range(0, len(v)-self.td_max+1))
        if self.ensemble_average==False:
            t0_list = [0]

        VACF = torch.zeros(self.td_max, device=v.device)

        for t0 in t0_list[::10]:
            dummy = v[t0:t0+self.td_max].detach()
            VACF += (dummy * v[t0]).mean((1, 2))
        VACF /= len(t0_list)
        
        if self.normalize == True:
            VACF /= VACF[0].item()
        return VACF

class VACF_wang(torch.nn.Module):
    def __init__(self, td_max, normalize=True):
        super(VACF_wang, self).__init__()
        self.td_max = td_max
        self.normalize = normalize

    def forward(self, vel):
        vacf = [(vel * vel.detach()).mean()[None]]
        # can be implemented in parrallel
        vacf += [ (vel[:-t] * vel[t:].detach()).mean()[None] for t in range(1, self.td_max)]

        vacf = torch.stack(vacf).reshape(-1)


        if self.normalize==True:
            vacf = vacf / vacf[0].item()

        return vacf    

class VACF_computer_complex(torch.nn.Module):
    def __init__(self, td_max, ensemble_average = True, normalize=True):
        super(VACF_computer_complex, self).__init__()
        self.td_max = td_max
        self.ensemble_average = ensemble_average
        self.normalize = normalize

    def forward(self, v):
        if len(v.shape) == 2:
            v = v.view(len(v), -1, 3)

        VACF = (v * v[0].detach()).mean((1, 2))

        VACF /= VACF[0].item()

        return VACF

def distribute_particles_in_cubic_space(L, Natom):
    # Calculate the number of particles per dimension (assuming a cubic grid)
    num_per_dimension = int(np.ceil(Natom**(1/3)))

    # Calculate the spacing between particles
    spacing = L / num_per_dimension

    # Generate particle coordinates
    particle_coordinates = []
    for x in range(num_per_dimension):
        for y in range(num_per_dimension):
            for z in range(num_per_dimension):
                x_coord = x * spacing
                y_coord = y * spacing
                z_coord = z * spacing
                particle_coordinates.append([x_coord, y_coord, z_coord])

    # Convert the list of coordinates to a NumPy array
    particle_coordinates = np.array(particle_coordinates)

    return particle_coordinates[:Natom]




# Model construction
class CombinedNN(torch.nn.Module):
    def __init__(self, nn1, nn2):
        super(CombinedNN, self).__init__()
        self.nn1 = nn1
        self.nn2 = nn2

    def forward(self, x):
        return self.nn1(x) + self.nn2(x)


# Write the XYZ dump file
#write_xyz_dump("/oden/jjeong/dump.xyz", type_example, atom_types_map, y[::100,Natom:,:], box_example)

def write_xyz_dump(filename, atom_types, atom_types_map, trajectories, box, mode = "w"):
    box = box.diag()
    with open(filename, mode) as file:
        for timestep, positions in enumerate(trajectories):
            num_atoms = len(positions)
            file.write("ITEM: TIMESTEP\n")
            file.write(f"{timestep}\n")
            file.write("ITEM: NUMBER OF ATOMS\n")
            file.write(f"{num_atoms}\n")
            file.write("ITEM: BOX BOUNDS pp pp pp\n")
            # Assuming the box starts at 0 for simplicity; adjust if your simulation does otherwise
            file.write(f"0.0000000000000000e+00 {box[0]:.16e}\n")
            file.write(f"0.0000000000000000e+00 {box[1]:.16e}\n")
            file.write(f"0.0000000000000000e+00 {box[2]:.16e}\n")
            file.write("ITEM: ATOMS id element x y z\n")
            for atom_id, (atom_idx, position) in enumerate(zip(atom_types, positions), start=1):
                element_symbol = atom_types_map[atom_idx]
                x, y, z = position
                # Write atom data excluding forces
                file.write(f"{atom_id} {element_symbol} {x} {y} {z}\n")


atomic_masses_map = [
    1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180,  # H to Ne
    22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078,  # Na to Ca
    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,  # Sc to Zn
    69.723, 72.63, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224,  # Ga to Zr
    92.906, 95.95, 98, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71,  # Nb to Sn
    121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24,  # Sb to Nd
    145, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05,  # Pm to Yb
    174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59,  # Lu to Hg
    204.38, 207.2, 208.98, 209, 210, 222, 223, 226, 227, 232.04,  # Tl to Th
    231.04, 238.03, 237, 244, 243, 247, 247, 251, 252, 257,  # Pa to Md
    258, 259, 262, 267, 268, 271, 270, 269, 278, 281,  # No to Ds
    282, 285, 286, 289, 290, 293, 294, 294  # Rg to Og
]


atom_types_map = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]