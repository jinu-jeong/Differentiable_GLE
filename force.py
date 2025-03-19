import torch
import torch.nn as nn
from utility import *
import matplotlib.pyplot as plt



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



class GLE_TS(torch.nn.Module):
    def __init__(self, h=None, gamma_uniform=0.1, filter_length=None, BOLTZMAN=0.001987191):
        super(GLE_TS, self).__init__()
        
        self.BOLTZMAN = BOLTZMAN
        
        # Initialize h as a learnable parameter if Trainable is True
        if filter_length==None:
            filter_length = np.ceil(0.05 / gamma_uniform)//1
        self.filter_length = filter_length

        self.h = torch.nn.Parameter(torch.full((self.filter_length ,), gamma_uniform)[None,None,:])
        
        # Placeholder for the memory kernel
        self.memory_kernel = None
        self.v_list = None
        self.w_list = None


    def construct_memory(self, T, mass):
        h_padded = torch.nn.functional.pad(self.h, (0, self.h.size(2)-1))
        # TODO : remove hack: mass[0]
        self.memory_kernel = torch.nn.functional.conv1d(h_padded, self.h) * (mass[0,0].item() * self.BOLTZMAN * T)
    
    def get_v_and_sample_w(self, v, dt):
        # initial
        if self.v_list == None:
            Natom = len(v.flatten())//3
            self.v_list = torch.zeros(Natom*3, 1, self.filter_length , device=v.device)
            self.w_list = torch.zeros(Natom*3, 1, self.filter_length , device=v.device)
        
        
        
        # Update velocity and noise lists
        self.v_list = torch.roll(self.v_list, -1, dims=2)
        self.v_list[:,:,-1] = v.view(-1,1)
        self.w_list = torch.roll(self.w_list, -1, dims=2)
        self.w_list[:,:,-1] = torch.randn_like(v, device=v.device).view(-1,1)
        

    def forward(self, v, T, dt, mass):
        # Construct the memory and update velocity and noise history
        self.construct_memory(T, mass)
        self.get_v_and_sample_w(v, dt)

        # Calculate dissipative force using the convolution of memory kernel with velocity history
        # flip(2) is for true convolution operation. In NN libraries, convolution is actually cross-correlation.
        friction = -torch.nn.functional.conv1d(self.v_list, self.memory_kernel.flip(2)).flatten()
        
        # Random force generator part of the Langevin force
        Langevin_coeff = 1

        random = torch.nn.functional.conv1d(self.w_list, self.h.flip(2)).flatten() * Langevin_coeff




        #Langevin_coeff = self.memory_kernel.item()**0.5 * torch.sqrt(2.0 / mass * self.BOLTZMAN * T * dt)
        #random = torch.randn_like(v, device=v.device) * Langevin_coeff/dt
        
        # Return the change in velocity due to Langevin dynamics
        delta_v_langevin = friction.view(v.shape)  + random.view(v.shape)
        return delta_v_langevin



#%%
class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return torch.nn.functional.softplus(x) - self.shift
    
class SchNet(nn.Module):
    def __init__(self, z, cell, n_gaussians, hidden_channels, n_filters, num_interactions, r_cut):
        super(SchNet, self).__init__()
        self.z = z.long()
        self.cell = cell.to(torch.float32)
       
        
        num_atom_types = 200
        self.embedding = nn.Embedding(num_atom_types, hidden_channels)



        self.GS_layer = GaussianSmearing(start=0.0, stop=r_cut, n_gaussians = n_gaussians)

        self.interactions = nn.ModuleList([InteractionBlock(n_gaussians, hidden_channels, n_filters, r_cut) for _ in range(num_interactions)])
        
        self.output_layer1 = nn.Linear(hidden_channels, hidden_channels//2)
        self.output_layer1_activation = ShiftedSoftplus()
        self.output_layer2 = nn.Linear(hidden_channels//2, 1)
        
        self.r_cut = r_cut
        
        
    def forward(self, pos):
        z = self.z

        # Generate graph data
        neighbor_indices, offsets, distances, unit_vectors = generate_nbr_list(pos, self.cell, self.r_cut)
        
        
        x = self.embedding(z)


        edge_index = neighbor_indices.t().contiguous()
        edge_weight = distances  # or any other edge features
        edge_attr = self.GS_layer(edge_weight)
        

        
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)
        
        x = self.output_layer1(x)
        x = self.output_layer1_activation(x)
        x = self.output_layer2(x)

        return x.sum()
    
class InteractionBlock(nn.Module):
    def __init__(self, n_gaussians, hidden_channels, n_filters, r_cut):
        super(InteractionBlock, self).__init__()
        

        ## convolution
        self.mlp = nn.Sequential(
            nn.Linear(n_gaussians, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters),
            )
        
        self.conv = SchConv(hidden_channels, hidden_channels, n_filters, self.mlp, r_cut)

        self.act = ShiftedSoftplus()

        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchConv(nn.Module):
    def __init__(self, channel_in, channel_out, n_filters, mlp, r_cut) -> None:
        super(SchConv, self).__init__()

        self.mlp = mlp
        self.lin1 = nn.Linear(channel_in, n_filters, bias=False)
        self.lin2 = nn.Linear(n_filters, channel_out)
        self.r_cut = r_cut

        

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.r_cut) + 1.0)
    
        ## node info processing
        x = self.lin1(x)

        # edge info processing (convolution)
        row, col = edge_index
        edge_messages1 = (self.mlp(edge_attr)) * x[row]
        edge_messages2 = (self.mlp(edge_attr)) * x[col]

        # message 
        aggr_messages = torch.zeros_like(x).scatter_add(0, col.unsqueeze(-1).expand_as(edge_messages1), edge_messages1)
        aggr_messages += torch.zeros_like(x).scatter_add(0, row.unsqueeze(-1).expand_as(edge_messages2), edge_messages2)


        
        return self.lin2(aggr_messages)
    


class GaussianSmearing(nn.Module):


    def __init__(self, start, stop, n_gaussians, width=None, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):

        coeff = -0.5 / torch.pow(self.width, 2)
        diff = distances - self.offsets
        gauss = torch.exp(coeff * torch.pow(diff, 2))
        
        return gauss
    

#%%
class LJ_repulsive(torch.nn.Module):
    def __init__(self, z, cell, sigma, epsilon, r_cut, Trainable = False, device=None):
        super(LJ_repulsive, self).__init__()
        self.z = z
        self.cell = cell # This is box size --> used for force calculation in PBC environment
        self.r_cut = r_cut
        if Trainable==True:
            self.log_sigma = torch.nn.Parameter(torch.Tensor([sigma]))
            self.log_epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))
        else:
            assert device!=None
            self.log_sigma = torch.Tensor([sigma]).to(device)
            self.log_epsilon = torch.Tensor([epsilon]).to(device)
        
        #num_atom_types = 200
        #self.embedding = nn.Embedding(num_atom_types, 2)
        
    def LJ_repulsive(self, r):
        
        return 4 *  torch.exp(self.log_epsilon) * (torch.exp(self.log_sigma)/r)**12


    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell, self.r_cut)
        return self.LJ_repulsive(pdist).sum()



