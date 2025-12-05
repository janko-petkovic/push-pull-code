
'''
Contains the model function and the two methods to generate the respective
parameters and parameter names
'''

import torch

from . import BaseModel


class StupidModel(BaseModel):
    '''
    The stimulus contribution is as follows
    
    K_stx = Ks * exp(tau) * sigmoid(sigma * sum_i{f(x-2*i*x_min)})
    
    with i going from 0 to nss-1, and x_min is x_min
    '''
    def __init__(self):
        super().__init__()


    def __str__(self):
        return 'StupidModel'


    def forward(self, nss_weights, t_mesh, x_mesh, p):
        '''
        The p vector is structured as so:
        - p[0:3] -> K : SoONblast, tau, lambda
        - p[3:3+len(x)] -> K : ioONblast
        - p[3+len(x):3+2*len(x)-1] -> N : ioNblast NOTICE THE LAST ONE IS MISSING
        - p[-1] : Pi
        '''

        p = torch.pow(10,p)

        # Get useful values
        x_min = x_mesh[0].min().item()
        nss = int(nss_weights[torch.where(x_mesh==0)][0].item())
        n_spines = len(x_mesh[0])
        n_t_points = len(t_mesh)

        # Meshes corresponding to stimulated coordinates
        t_mesh_stim = t_mesh[torch.where(t_mesh>=0)]
        x_mesh_stim = x_mesh[torch.where(t_mesh>=0)]

        # Basal K_b for everyone
        K_tx = p[3:3+n_spines].tile((n_t_points, 1)).to(torch.float64)

        # Basal N_b for everyone
        N_tx = p[3+n_spines:3+2*n_spines-1].tile(n_t_points, 1)
        N_tx = torch.cat((N_tx, torch.ones(n_t_points,1)), dim=1).to(torch.float64)


        # # Gaussian contributions
        # K
        # compute the sum of f
        K_sum_of_f_tx = torch.zeros_like(t_mesh_stim)
        for i in range(nss):
            K_sum_of_f_tx += torch.exp(-(x_mesh_stim-2*i*x_min)**2/p[2])
        
        # Sigmoid of the sum of f (previous handle for the non linearity)
        # K_sig_tx = K_sum_of_f_tx

        # Decay in time and multiplication for Ks
        K_stx = p[0]*torch.exp(-t_mesh_stim/p[1])*K_sum_of_f_tx

        # Total K and N in space and time
        K_tx[torch.where(t_mesh>=0)] = K_tx[torch.where(t_mesh>=0)] + K_stx 
        N_tx[torch.where(t_mesh>=0)] = N_tx[torch.where(t_mesh>=0)]
        
        
        # The quantity A has to account for the number of stimulations
        alpha_tx = K_tx/N_tx
        weighted_alpha_tx = alpha_tx * nss_weights

        A_t = weighted_alpha_tx.sum(axis=1)

        # recover the correct dimension
        A_t, _ = torch.meshgrid(A_t, x_mesh[0], indexing='ij')

        return p[-1]*alpha_tx/(1 + A_t)



    def generate_parameter_names(self, t, x):
        '''Creates the parameter names for the local model'''

        # Start with the global parameters
        param_names = [
            r'$Ks/\Omega \, N_{b,n}$', r'$\tau_K$', r'$\lambda_K$',
        ]

        # Then each spine starting point for K
        param_names += [
            f'KboONblast_{i:.3}' for i in x
        ]

        # Then each spine starting point for N (remember that N_last = 1!)
        param_names += [
            f'NboNblast_{i:.3}' for i in x
        ]

        # Finally Pi
        param_names[-1] = 'Pi'

        return param_names



    def generate_parameter_guess(self, t, x, p_guess=10.):

        p_names = self.generate_parameter_names(t, x)

        params = [p_guess]*(len(p_names))
        params = torch.log10(torch.tensor(params))

        return params


    def generate_boundary_conditions(self, t, x, p_optim):
        parnames = self.generate_parameter_names(t, x)
        lb = [-3]*len(parnames)
        ub = [3]*len(parnames)

        lb[0] = -1
        ub[0] = 2
        lb[1] = 0
        ub[1] = 1
        lb[2] = 0
        ub[2] = 2

        for idx in range(len(x)):
            lb[3+idx] = 1
            ub[3+idx] = 2

        for idx in range(len(x)-1):
            lb[3+len(x)+idx] = -1
            ub[3+len(x)+idx] = 1

        lb[-1] = 5
        ub[-1] = 7

        # Id don't what array is p_optim going to be
        if hasattr(p_optim, '__len__'):
            eps = 0.01

            # Tau K
            lb[1] = p_optim[1] - eps
            ub[1] = p_optim[1] + eps

            # Sigma K
            lb[2] = p_optim[2] - eps
            ub[2] = p_optim[2] + eps

        return lb, ub


if __name__ == '__main__':
    model = LocalModel()

    x = torch.arange(20, dtype=torch.float64)
    t = torch.arange(10, dtype=torch.float64)
    p_guess = model.generate_parameter_guess(t, x)
    p_names = model.generate_parameter_names(t, x)

    if len(p_guess) != len(p_names):
        raise RuntimeError('Something wrong with par names and par guess')

    print('All good!')