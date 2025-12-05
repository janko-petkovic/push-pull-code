
'''
Contains the model function and the two methods to generate the respective
parameters and parameter names
'''

import torch
from math import log10

from . import BaseModel


class LocalGaussModel(BaseModel):
    '''
    The stimulus contribution belongs to this class
    
    K_stx = Ks * exp(tau) * non_lin(lambda * sum_i{f(x-2*i*x_min)})
    
    with i going from 0 to nss-1, and x_min is x_min.
    In this case we do not use the non linearity and use f = gaussian

    K_stx = Ks * exp(tau) * sum_i{-(x-2*i*x_min)**2/sigma**2}
    '''

    def __init__(self):
        super().__init__()


    def __str__(self):
        return 'LocalGaussModel'


    ######################
    # Single model fitting
    ######################

    def forward(self, nss_weights, t_mesh, x_mesh, p):
        '''
        The p vector is structured as so:
        - p[0:3] -> K : SoONblast, tau, lambda
        - p[3:6] -> N : SoNblast, tau, lambda
        - p[6:6+len(x)] -> K : ioONblast
        - p[6+len(x):6+2*len(x)-1] -> N : ioNblast NOTICE THE LAST ONE IS MISSING
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
        K_tx = p[6:6+n_spines].tile((n_t_points, 1)).to(torch.float64)

        # Basal N_b for everyone
        N_tx = p[6+n_spines:6+2*n_spines-1].tile(n_t_points, 1)
        N_tx = torch.cat((N_tx, torch.ones(n_t_points,1)), dim=1).to(torch.float64)


        # # Gaussian contributions
        # K
        # compute the sum of f
        K_sum_of_f_tx = torch.zeros_like(t_mesh_stim)
        for i in range(nss):
            K_sum_of_f_tx += torch.exp(-(x_mesh_stim-2*i*x_min)**2/p[2]**2)
        
        # Sigmoid of the sum of f (previous handle for the non linearity)
        # K_sig_tx = K_sum_of_f_tx

        # Decay in time and multiplication for Ks
        # K_stx = p[0]*torch.exp(-t_mesh_stim/p[1])*K_sig_tx
        K_stx = p[0]*torch.exp(-t_mesh_stim/p[1])*K_sum_of_f_tx


        # N
        # compute the sum of f
        N_sum_of_f_tx = torch.zeros_like(t_mesh_stim)
        for i in range(nss):
            N_sum_of_f_tx += torch.exp(-(x_mesh_stim-2*i*x_min)**2/p[5]**2)
        
        # Sigmoid of the sum of f (previous handle for the non linearity)
        # N_sig_tx = N_sum_of_f_tx

        # Decay in time and multiplication for Ns
        # N_stx = p[4]*torch.exp(-t_mesh_stim/p[5])*N_sig_tx
        N_stx = p[3]*torch.exp(-t_mesh_stim/p[4])*N_sum_of_f_tx



        # Total K and N in space and time
        K_tx[torch.where(t_mesh>=0)] = K_tx[torch.where(t_mesh>=0)] + K_stx 
        N_tx[torch.where(t_mesh>=0)] = N_tx[torch.where(t_mesh>=0)] + N_stx
        
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
            'KsoONbl', 'tau_K', 'sigma_K',
            'NsoNbl', 'tau_N', 'sigma_N',
        ]

        # Then each spine starting point for K
        param_names += [
            f'KboONbl_{i:.3}' for i in x
        ]

        # Then each spine starting point for N (remember that N_last = 1!)
        param_names += [
            f'NboNbl_{i:.3}' for i in x
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
        '''
        Only part of the p_optim vector will be utilized by the routine
        but its easier this way, and also less error prone.

        The p vector is structured as so:
        - p[0:3] -> K : SoONblast, tau, lambda
        - p[3:6] -> N : SoNblast, tau, lambda
        - p[6:6+len(x)] -> K : ioONblast
        - p[6+len(x):6+2*len(x)-1] -> N : ioNblast NOTICE THE LAST ONE IS MISSING
        - p[-1] : Pi
        '''

        parnames = self.generate_parameter_names(t, x)
        lb = [-3]*len(parnames)
        ub = [3]*len(parnames)

        lb[0] = log10(0.02)
        ub[0] = log10(1)
        lb[1] = 0
        ub[1] = 2
        lb[2] = 0
        ub[2] = 1
        
        lb[3] = -1
        ub[3] = 1
        lb[4] = -1
        ub[4] = 1
        lb[5] = 0
        ub[5] = 2

        for idx in range(len(x)):
            lb[6+idx] = log10(0.1)
            ub[6+idx] = log10(0.5)

        for idx in range(len(x)-1):
            lb[6+len(x)+idx] = log10(0.42)
            ub[6+len(x)+idx] = log10(2.35)

        lb[-1] = 5
        ub[-1] = 7

        if hasattr(p_optim, '__len__'):
            eps = 0.01
            # Tau K
            lb[1] = p_optim[1]-eps
            ub[1] = p_optim[1]+eps

            # Lambda K
            lb[2] = p_optim[2]-eps
            ub[2] = p_optim[2]+eps

            # Tau N
            lb[4] = p_optim[4]-eps
            ub[4] = p_optim[4]+eps

            # Lambda N
            lb[5] = p_optim[5]-eps
            ub[5] = p_optim[5]+eps


        return lb, ub


    
    #####################
    # Interface for multi
    #####################

    def generate_parameter_names_for_multi(self, t, x):
        
        par_names = self.generate_parameter_names(t, x)

        glob_pn = [par_names[1],par_names[2],par_names[4],par_names[5]]
        spec_pn = [par_names[0],par_names[3]] + par_names[6:]

        return glob_pn, spec_pn



    def generate_boundary_conditions_for_multi(self, t, x):
        '''
        Returns in order glb, gub, slb, gub
        '''
        lb, ub = self.generate_boundary_conditions(t, x, None)

        glb = [lb[1], lb[2], lb[4], lb[5]]
        slb = [lb[0], lb[3]] + lb[6:]

        gub = [ub[1], ub[2], ub[4], ub[5]]
        sub = [ub[0], ub[3]] + ub[6:]

        return glb, gub, slb, sub



    def parse_p_from_multi(self, offset, nss_weights, p_multi):
        '''
        Remember that p_multi is a torch with a gradient, we have to 
        keep all torch while building parsed_p
        '''

        n_spines = len(nss_weights.T)
        glob_p = p_multi[:4]
        spec_p_span = 2*n_spines + 2
        spec_p = p_multi[4 + offset : 4 + offset + spec_p_span]
        offset += spec_p_span


        parsed_p = torch.empty(4 + spec_p_span)
        parsed_p[0] = spec_p[0]
        parsed_p[1:3] = glob_p[:2]
        parsed_p[3] = spec_p[1]
        parsed_p[4:6] = glob_p[2:]
        parsed_p[6:] = spec_p[2:]

        return parsed_p, offset



    def generate_multi_profile_idxes(self, multi_result):
        profile_idxes =[0]*len(multi_result.optimize_result.as_list()[0].x)
        profile_idxes[0] = 1
        profile_idxes[1] = 1
        profile_idxes[2] = 1
        profile_idxes[3] = 1

        return profile_idxes



    ############
    # Validation
    ############

    def validation_forward(self, t_mesh, x_mesh, x_stim, p_dict):
        '''
        Parameters
        ----------        
        t_mesh : array (t_points, n_spines)

        x_mesh : array (t_points, n_spines)

        x_stim : array (number of stimulations)
            Contains the positions of the stimulations

        p : dict
            Dictionary with the necessary parameters.
            Remember that is has to contain the two lists of Kb and Nb
            each long (n_spines)

        Returns
        -------
        array (t_pints, n_spines)
            Bidimensional array with spine sizes in time and space
        '''

        # Alpha computation
        # Basal condition for everyone
        Kb_tx = torch.tile(p_dict['Kbs'], (len(t_mesh), 1))
        Nb_tx = torch.tile(p_dict['Nbs'], (len(t_mesh), 1))

        # compute the sum of f
        K_sum_of_f_tx = torch.zeros_like(x_mesh)
        N_sum_of_f_tx = torch.zeros_like(x_mesh)

        for xs in x_stim:
            K_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['lambda_K']**2)
            N_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['lambda_N']**2)
        
        Ks_tx = p_dict['Ks']*torch.exp(-t_mesh/p_dict['tau_K'])*K_sum_of_f_tx
        Ns_tx = p_dict['Ns']*torch.exp(-t_mesh/p_dict['tau_N'])*N_sum_of_f_tx

        # Add basal and stimulus contributions
        K_tx = Kb_tx + Ks_tx
        N_tx = Nb_tx + Ns_tx

        # Alpha at each spine for each time point
        alpha_tx = K_tx / N_tx

        # One alpha sum for each timepoint
        A_t = alpha_tx.sum(axis=1)

        # recover the correct dimension
        A_t = torch.tile(A_t, (len(x_mesh[0]),1)).T

        sizes_tx = p_dict['Pi']*alpha_tx/(1 + A_t)

        # This is a utility for better comparison
        alphab_tx = (Kb_tx / Nb_tx)
        Ab_t = alphab_tx.sum(axis=1)
        Ab_t = torch.tile(Ab_t, (len(x_mesh[0]),1)).T

        sizesb_tx =  p_dict['Pi']*alphab_tx/(1 + Ab_t)

        return sizes_tx, sizesb_tx 



if __name__ == '__main__':
    model = LocalModel()

    x = torch.arange(20, dtype=torch.float64)
    t = torch.arange(10, dtype=torch.float64)
    p_guess = model.generate_parameter_guess(t, x)
    p_names = model.generate_parameter_names(t, x)

    if len(p_guess) != len(p_names):
        raise RuntimeError('Something wrong with par names and par guess')

    v_p_names = model.generate_validation_parameter_names(t, x)
    v_p_guess = model.generate_validation_parameter_guess(t, x)

    t_mesh, x_mesh = torch.meshgrid(t,x, indexing='ij')
    nssw = model.generate_nss_weights(1, t, x)

    pred = model(nssw, t_mesh, x_mesh, p_guess)

    pred_valid = model.validation_forward(nssw, t_mesh, x_mesh, 
        model.extract_p_validation(p_guess), v_p_guess)

    print('All good!')
    print(((pred - pred_valid)**2).sum())