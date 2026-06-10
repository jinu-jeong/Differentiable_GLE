import torch
from utility import *



class Tabulated(torch.nn.Module):
    def __init__(self, cell, Tabulated_data):
        super(Tabulated, self).__init__()
        self.cell = cell
        
        from scipy.interpolate import interp1d
        #f = interp1d(Tabulated_data[:,1], Tabulated_data[:,3])
        f = interp1d(Tabulated_data[:,1], Tabulated_data[:,3], kind='cubic', fill_value="extrapolate", bounds_error=False)
        
        self.force_magnitude = f
        
    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        
        force = torch.zeros_like(q, device=q.device)
        force_magnitude = torch.tensor(self.force_magnitude(pdist.detach().cpu().numpy())).to(q.device).to(q.dtype)
        force_vector = force_magnitude*unit_vector
                
        force.index_add_(0, nbr_list[:,0], force_vector)
        force.index_add_(0, nbr_list[:,1], -force_vector)
        return force.detach()
    

class Tabulated_specific(torch.nn.Module):
    def __init__(self, cell, z, Tabulated_data_list, interaction_list):
        super(Tabulated_specific, self).__init__()
        self.cell = cell
        self.z = z.to(cell.device)
        
        from scipy.interpolate import interp1d
        #f = interp1d(Tabulated_data[:,1], Tabulated_data[:,3])

        self.force_magnitude = []
        for i in range(len(Tabulated_data_list)):
            Tabulated_data = Tabulated_data_list[i]
            f = interp1d(Tabulated_data[:,1], Tabulated_data[:,3], kind='cubic', fill_value="extrapolate", bounds_error=False)
            interaction = torch.tensor(interaction_list[i]).to(cell.device)
            self.force_magnitude.append([f, interaction])
                

    def forward(self, q):

        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)


        force = torch.zeros_like(q, device=q.device)
        for f, interaction in self.force_magnitude:
            
            force_magnitude = torch.tensor(f(pdist.detach().cpu().numpy())).to(q.device).to(q.dtype)
            force_vector = force_magnitude*unit_vector

            atom_types_pair = self.z[nbr_list]
            mask = ((atom_types_pair == interaction) | (atom_types_pair == interaction.flip(0))).all(dim=1)

  

            force.index_add_(0, nbr_list[mask,0], force_vector[mask])
            force.index_add_(0, nbr_list[mask,1], -force_vector[mask])
        
        
        return force.detach()

class LJ(torch.nn.Module):
    def __init__(self, cell, sigma=3.405 , epsilon=0.0103, Trainable = False):
        super(LJ, self).__init__()
        self.cell = cell # This is box size --> used for force calculation in PBC environment
        if Trainable==True:
            self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
            self.epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))
        else:
            self.sigma = sigma
            self.epsilon = epsilon
        
    def LJ(self, r):
        return 4 *  self.epsilon * ((self.sigma/r)**12 - (self.sigma/r)**6 )

    def LJ_force(self, r):
        # Calculate the negative derivative of the potential energy with respect to r
        dV_dr = 24 *  self.epsilon * (2 * (self.sigma**12) / (r**13) - (self.sigma**6) / (r**7))
        return dV_dr

    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        return self.LJ(pdist)


class Langevin_TS(torch.nn.Module):
    def __init__(self, gamma, BOLTZMAN =  0.001987191, Trainable=False):
        super(Langevin_TS, self).__init__()
        if Trainable==False:
            self.gamma = gamma
        else:
            self.gamma = torch.nn.Parameter(torch.Tensor([gamma]))

        self.BOLTZMAN = BOLTZMAN

    def forward(self, v, T, dt, mass, t):
        friction = -v*self.gamma
        Langevin_coeff = torch.sqrt(2.0 * self.gamma / mass * self.BOLTZMAN * T * dt)
        random = torch.randn_like(v, device=v.device) * Langevin_coeff/dt
        delta_v_langevin = friction + random
        return delta_v_langevin


class GLE_TS_AuxVariable(torch.nn.Module):
    """GLE thermostat using auxiliary variables for Markov extension.

    Converts K(t) = Σ cₖ exp(-λₖ t) into extended system with aux variables sₖ:
        ds_k/dt = -λₖ s_k + √(2 λₖ) ξₖ(t)
        friction = Σ cₖ s_k

    For oscillatory modes, use complex pairs or damped harmonic form.
    This formulation is compatible with torchsde white-noise integration.
    """
    def __init__(self, lambdas, c_coeffs, BOLTZMAN=0.001987191, Trainable=False):
        """
        Parameters
        ----------
        lambdas : torch.Tensor or list of float
            Decay rates λₖ (shape: (n_modes,))
        c_coeffs : torch.Tensor or list of float
            Amplitudes cₖ (shape: (n_modes,))
        BOLTZMAN : float
            Boltzmann constant
        Trainable : bool
            If True, lambdas and c_coeffs become learnable parameters.
        """
        super(GLE_TS_AuxVariable, self).__init__()
        self.BOLTZMAN = BOLTZMAN
        self.n_modes = len(lambdas)

        if Trainable:
            self.lambdas = torch.nn.Parameter(torch.tensor(lambdas, dtype=torch.float))
            self.c_coeffs = torch.nn.Parameter(torch.tensor(c_coeffs, dtype=torch.float))
        else:
            self.register_buffer('lambdas', torch.tensor(lambdas, dtype=torch.float))
            self.register_buffer('c_coeffs', torch.tensor(c_coeffs, dtype=torch.float))

    def forward(self, v, s_dict, T, dt, mass, t):
        """Compute friction force from auxiliary variables.

        Parameters
        ----------
        v : torch.Tensor
            Velocity (N_atoms, 3)
        s_dict : dict
            Dictionary of auxiliary variables {k: s_k} where s_k shape (N_atoms, 3)
        T : float
            Temperature
        dt : float
            Time step
        mass : torch.Tensor
            Mass (N_atoms, 1)
        t : float
            Current time

        Returns
        -------
        friction : torch.Tensor
            Friction force (N_atoms, 3)
        ds_dict : dict
            Time derivatives {k: ds_k/dt}
        noise_dict : dict
            Noise terms {k: noise_k} for stochastic integration
        """
        friction = torch.zeros_like(v)
        ds_dict = {}
        noise_dict = {}

        for k in range(self.n_modes):
            if k in s_dict:
                s_k = s_dict[k]
                lambda_k = self.lambdas[k]
                c_k = self.c_coeffs[k]

                # Friction contribution
                friction = friction + c_k * s_k

                # ds_k/dt = -λ_k * s_k
                ds_dict[k] = -lambda_k * s_k

                # Noise amplitude: √(2 λ_k c_k² c_v T / m) where c_v = 3k_B/2 per atom
                # Simplified: noise ∝ √(λ_k)
                noise_amp = torch.sqrt(2.0 * lambda_k * self.BOLTZMAN * T / mass)
                noise_dict[k] = noise_amp

        return friction, ds_dict, noise_dict

    def initialize_aux_variables(self, v, device):
        """Initialize auxiliary variables from velocity.

        Returns
        -------
        s_dict : dict
            {k: torch.zeros_like(v)} for each mode
        """
        s_dict = {k: torch.zeros_like(v, device=device) for k in range(self.n_modes)}
        return s_dict



