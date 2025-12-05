'''
Worst name but for now I remember what I wanted.
Differs from LocalGaussModel as it has a wider range of parameters,
namely two additional ones:
- omega over omega_last
- len(Kb) == len(Nb) (we do not drop the last one)
'''

import torch
from math import log10

from . import BaseModel


class LocalGaussModelTilde(BaseModel):
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
        return 'LocalGaussModelTilde'


    ######################
    # Single model fitting
    ######################

    def forward(self, nss_weights, t_mesh, x_mesh, p):
        '''
        The p vector is structured as so:
        - p[0:3] -> K : SoOlNbl, tau, lambda
        - p[3:6] -> N : SoNbl, tau, lambda
        - p[6:6+len(x)] -> K : ioONblast
        - p[6+len(x):6+2*len(x)] -> N : ioNblast
        - p[-2] : OoOl
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

        # Basal N_b for everyone (note that we are not adding the last
        # 1 but taking directly n_spines elements)
        N_tx = p[6+n_spines:6+2*n_spines].tile(n_t_points, 1)


        # # Gaussian contributions
        # K
        # compute the sum of f
        K_sum_of_f_tx = torch.zeros_like(t_mesh_stim)
        for i in range(nss):
            K_sum_of_f_tx += torch.exp(-(x_mesh_stim-2*i*x_min)**2/p[2]**2)
            # K_sum_of_f_tx += torch.exp((x_mesh_stim-2*i*x_min).abs()/p[2])
        
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
            # N_sum_of_f_tx += torch.exp((x_mesh_stim-2*i*x_min).abs()/p[5])
        
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

        return p[-1]*alpha_tx/(p[-2] + A_t)



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

        # Then each spine starting point for N
        param_names += [
            f'NboNbl_{i:.3}' for i in x
        ]

        # Omega over Omega last
        param_names += ['OoOl']

        # Finally Pi
        param_names += ['Pi']

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

        Fitting diary
        -------------
        Iteration 1:
            Parameters from literature and Jean finding that proteins are
            10-30 times more abundant in the spine. I assume a dendrite
            rougly long 100 um 
            1.1 : vanilla
            1.2 : say CaM is limiting
            1.3 : Ks and Ns want to be smaller
            1.4 : Ns wants to be even smaller, Kb Nb more spread
                        Kb = [log10(0.001),log10(0.2)]
                        Nb = [log10(0.1), log10(10)]

        Iteration 2:
            Turns out ratio is roughly 1, so chi=[0.8, 1.2] (Kanaan and Surbh).
            From the simulations I always end up using a 1000 um long dendrite,
            so lets try that. Interestingly, longet dendrite seems to account
            for some of the trends in the previous fit trials (e.g., Kb ending
            on the lower bound)
            2.1 : vanilla
            2.2 : try higher ub for both kb and nb (only nb with student)
            2.3 : higher both kb and nb
            2.4 : wider nb
            2.5 : wider nb and kb
            2.6 : slightly higher kb

            Dropping 7 clustered
            2.2 : Kb wants to be bigger, maybe because the dendritic length is 
                  overestimated. Increase KboONbl ub to double
            2.3 : Kbs are still slightly constrained by the upper bound and Nbs 
                  by the lower one. I will adjust both. See what happens with
                  KsoONbl (for now, always lying on the lower bound)
                  - result: we are back to the old results, we lose the goda2022
                    behaviour. Also the basal distributions are shit now, Nb is
                    now upper bounded -> go to old Nb bounds, try only higher
                    Kb ub
            # 2.4 : Only Kbs higher ub, the rest is basal.

            Dropping 15:
            2.1 : bad
            2.2 : higher Kb Nb ubs: almost good, Nb is squeezed right
            2.2 : even higher Nb ubs
            2.3 : even higher Nb ubs
            2.4 : higher Nb Kb ubs
            2.5 : vanilla + low Ks
            2.6 : 10x Kb ub + 10x Ks both

        Iteration 3
            Kbs and Nbs are taken from 2.5 and 2.6
            3.1 : Ks and Ns 1/10 (drop 15)
            3.2 : Ns lower 0.5 (drop 15)
            3.3 : Nbs higher 1.5x (drop 15)
        '''

        parnames = self.generate_parameter_names(t, x)
        lb = [-3]*len(parnames)
        ub = [3]*len(parnames)

        # KsoOlNbl, tau_k, lambda_K
        lb[0] = log10(0.006)
        ub[0] = log10(0.16)

        lb[1] = 0
        ub[1] = 2
        lb[2] = 0
        ub[2] = log10(30)
        
        # NsoNbl, tau_N, lambda_N
        lb[3] = log10(2.92)
        ub[3] = log10(17)

        lb[4] = 0
        ub[4] = 2
        lb[5] = 0
        ub[5] = log10(30)

        # KboOlNbl
        for idx in range(len(x)):
            lb[6+idx] = log10(0.0002)
            ub[6+idx] = log10(0.005)

        # NboNbl
        for idx in range(len(x)):
            lb[6+len(x)+idx] = log10(0.28)
            ub[6+len(x)+idx] = log10(3.5)

        # OoOl
        lb[-2] = log10(0.4)
        ub[-2] = log10(2.3)


        # Pi
        lb[-1] = 6
        ub[-1] = 9

        
        return lb, ub


    
    #####################
    # Interface for multi
    #####################

    def generate_parameter_names_for_multi(self, t, x, is_last=False):
        '''
        Parameters
        ----------
        t : numpy array
            Time points

        x : numpy array
            Spine positions

        is_last : bool
            If we are generating for the last experiment
        '''

        par_names = self.generate_parameter_names(t, x)

        # we want to drop omega and the last Kb for the last experiment
        if is_last:
            glob_pn = par_names[:6]

            # Drop last NboNbl and OoOl (they are 1 without gradient)
            spec_pn = par_names[6:-3]
            # Keep Pi tough
            spec_pn += [par_names[-1]]

        else:
            glob_pn = par_names[:6]
            spec_pn = par_names[6:]

        return glob_pn, spec_pn



    def generate_boundary_conditions_for_multi(self, t, x, is_last=False):
        '''
        Parameters
        ----------
        t : numpy array
            Time points
        x : numpy array
            Spine positions
        is_last : bool
            If we are generating for the last experiment

        Returns
        -------
        tuple 
            In order: glb, gub, slb, sub
        '''
        lb, ub = self.generate_boundary_conditions(t, x, None)

        if is_last:
            glb = lb[:6]
            gub = ub[:6]

            # Same logic followed to generate par names
            slb = lb[6:-3]
            slb += [lb[-1]]
            sub = ub[6:-3]
            sub += [ub[-1]]

        else:
            glb = lb[:6]
            gub = ub[:6]
            
            slb = lb[6:]
            sub = ub[6:]

        return glb, gub, slb, sub



    def parse_p_from_multi(self, offset, nss_weights, p_multi, is_last=False):
        '''
        Weird function, best solution I found for now

        Parameters
        ----------
        offset : int
            Where to start reading the specific parameters from in the p_multi
            array
        nss_weights : array (t_points, n_spines)
            The nss_weights for all the experiments
        p_multi : array
            The array with all the parameters
        is_last : bool
            If we are in the last experiment

        Returns
        -------
        parsed_p : numpy array (6 + 2*n_spines + 2)
            Array of parsed_parameters 
            
        offset : int 
            Updated offset value

        Note
        ----
        Remember that p_multi is a torch with a gradient, we have to 
        keep all torch while building parsed_p
        '''
        n_spines = len(nss_weights.T)

        # Generate torch container and assign after (mantain gradients)
        parsed_p = torch.ones(6 + 2*n_spines + 2).to(torch.float64)

        # Read the global parameters
        parsed_p[:6] = p_multi[:6]

        # And now the specific parameters
        if is_last:
            spec_p_span = 2*n_spines
            spec_p = p_multi[6 + offset : 6 + offset + spec_p_span]
            
            # We leave a space for Kblast and Omegalast
            parsed_p[6:-3] = spec_p[:-1]
            parsed_p[-1] = spec_p[-1]
            

        else:
            spec_p_span = 2*n_spines + 2
            spec_p = p_multi[6 + offset : 6 + offset + spec_p_span]

            parsed_p[6:] = spec_p

        offset += spec_p_span

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

    
    def validation_forward(self, t_mesh, x_mesh, x_stim, p_dict) -> tuple:
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
        tuple (sizes_tx, basal_sizes_tx)
            Two matrices [timepoints, n_spines]
        '''

        # Alpha computation
        # Basal condition for everyone
        Kb_tx = torch.tile(p_dict['Kbs'], (len(t_mesh), 1))
        Nb_tx = torch.tile(p_dict['Nbs'], (len(t_mesh), 1))

        # compute the sum of f
        K_sum_of_f_tx = torch.zeros_like(x_mesh)
        N_sum_of_f_tx = torch.zeros_like(x_mesh)

        for xs in x_stim:
            K_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['sigma_K']**2)
            N_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['sigma_N']**2)
        
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

        sizes_tx = p_dict['Pi']*alpha_tx/(p_dict['Omega'] + A_t)

        # This is a utility for better comparison
        alphab_tx = (Kb_tx / Nb_tx)
        Ab_t = alphab_tx.sum(axis=1)
        Ab_t = torch.tile(Ab_t, (len(x_mesh[0]),1)).T

        sizesb_tx =  p_dict['Pi']*alphab_tx/(p_dict['Omega'] + Ab_t)

        return sizes_tx, sizesb_tx 


    
    def validation_forward_r_delta_r(self, t_mesh, x_mesh, x_stim, p_dict) -> tuple:

        '''
        Generate the parameters that Simulation will use to construct the
        non responders' function in the phase space.

        Parameters
        ----------        
        t : int > 2

        xs : array (n_spines)

        x_stim : array (number of stimulations)
            Contains the positions of the stimulations

        p : dict
            Dictionary with the necessary parameters.
            Remember that is has to contain the two lists of Kb and Nb
            each long (n_spines)

        Returns
        -------
        tuple (Rb_t, deltaR_t)
            Basal and variation of the remaineder coefficient R
        '''

        # Alpha computation
        # Basal condition for everyone
        Kb_tx = torch.tile(p_dict['Kbs'], (len(t_mesh), 1))
        Nb_tx = torch.tile(p_dict['Nbs'], (len(t_mesh), 1))

        # compute the sum of f
        K_sum_of_f_tx = torch.zeros_like(x_mesh)
        N_sum_of_f_tx = torch.zeros_like(x_mesh)

        for xs in x_stim:
            K_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['sigma_K']**2)
            N_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['sigma_N']**2)
        
        Ks_tx = p_dict['Ks']*torch.exp(-t_mesh/p_dict['tau_K'])*K_sum_of_f_tx
        Ns_tx = p_dict['Ns']*torch.exp(-t_mesh/p_dict['tau_N'])*N_sum_of_f_tx


        # Add basal and stimulus contributions
        K_tx = Kb_tx + Ks_tx
        N_tx = Nb_tx + Ns_tx

        # Alpha and alpha basal 
        alpha_tx = K_tx / N_tx
        alphab_tx = Kb_tx / Nb_tx

        # Mask for R
        mask = torch.ones(len(x_mesh[0]), dtype=int)
        mask[x_stim] = 0

        R_t = p_dict['Omega'] + alpha_tx[:,mask].sum(dim=1)
        Rb_t = p_dict['Omega'] + alphab_tx[:,mask].sum(dim=1)


        deltaR_t = R_t - Rb_t


        return Rb_t, deltaR_t 



    def validation_forward_return_all(
        self, t_mesh, x_mesh, x_stim, p_dict
        ) -> tuple:
        '''
        Utility function used to generate the figures for the model cartoon.
        Works like validation_forward, but returns all the fields
        
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
        tuple (sizes_tx, basal_sizes_tx)
            Two matrices [timepoints, n_spines]
        '''

        # Alpha computation
        # Basal condition for everyone
        Kb_tx = torch.tile(p_dict['Kbs'], (len(t_mesh), 1))
        Nb_tx = torch.tile(p_dict['Nbs'], (len(t_mesh), 1))

        # compute the sum of f
        K_sum_of_f_tx = torch.zeros_like(x_mesh)
        N_sum_of_f_tx = torch.zeros_like(x_mesh)

        for xs in x_stim:
            K_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['sigma_K']**2)
            N_sum_of_f_tx += torch.exp(-(x_mesh-xs)**2/p_dict['sigma_N']**2)
        
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

        sizes_tx = p_dict['Pi']*alpha_tx/(p_dict['Omega'] + A_t)

        # This is a utility for better comparison
        alphab_tx = (Kb_tx / Nb_tx)
        Ab_t = alphab_tx.sum(axis=1)
        Ab_t = torch.tile(Ab_t, (len(x_mesh[0]),1)).T

        sizesb_tx =  p_dict['Pi']*alphab_tx/(p_dict['Omega'] + Ab_t)

        return sizes_tx, sizesb_tx, K_tx, Kb_tx, N_tx, Nb_tx 



if __name__ == '__main__':
    model = LocalGaussModelTilde()

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
