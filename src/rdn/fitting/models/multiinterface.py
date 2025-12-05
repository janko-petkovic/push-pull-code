
'''
Contains the model function and the two methods to generate the respective
parameters and parameter names
'''

import torch

from . import BaseModel


class MultiInterface():

    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model


    def __str__(self):
        return f'Multi_{self.model}'


    def __call__(self, concatenate, nsss_weights, mesh_pairs, p_multi):
        return self.forward(concatenate, nsss_weights, mesh_pairs, p_multi)


    def is_multi(self):
        return True


    def generate_nsss_weights(self, nsss, tx_pairs):
        
        nsss_weights = []

        for nss, (t,x) in zip(nsss, tx_pairs):
            nsss_weights.append(self.model.generate_nss_weights(nss, t, x))

        return nsss_weights



    def forward(self, concatenate_output, nsss_weights, mesh_pairs, p):

        offset = 0
        is_last = False
        predictions = []

        # Dunno a better way
        for idx, (nss_weights, (t_mesh, x_mesh)) in \
            enumerate(zip(nsss_weights, mesh_pairs)):

            # Check if this is the last experiment
            is_last = (idx == len(nsss_weights)-1)

            # Create the data specific parameter vector   
            parsed_p, offset = self.model.parse_p_from_multi(offset, nss_weights, p, is_last=is_last)

            # Obtain the predictions
            predictions.append(self.model(nss_weights, t_mesh, x_mesh, parsed_p))

        if concatenate_output:
            return torch.concatenate(predictions, axis=1)
        else:
            return predictions



    def generate_parameter_names(self, nsss, tx_pairs):

        spec_p_names = []
        is_last = False
        
        for idx, (nss, (t,x)) in enumerate(zip(nsss, tx_pairs)):

            # Check if this is the last experiment
            is_last = (idx == len(nsss)-1)

            glob_pn, spec_pn = self.model.generate_parameter_names_for_multi(t, x, is_last=is_last)
            spec_pn = [f'{nss}_{spn}' for spn in spec_pn]
            spec_p_names += spec_pn
    
        # Dirty but effective
        full_p_names = glob_pn + spec_p_names

        return full_p_names


    def generate_parameter_guess(self, nsss, tx_pairs, p_guess=10.):

        p_names = self.generate_parameter_names(nsss, tx_pairs)

        params = [p_guess]*(len(p_names))
        params = torch.log10(torch.tensor(params))

        return params



    def generate_boundary_conditions(self, tx_pairs):
        
        spec_lbs = []
        spec_ubs = []
        is_last = False

        for idx, (t, x) in enumerate(tx_pairs):

            # Check if this is the last experiment
            is_last = (idx == len(tx_pairs)-1)

            glb, gub, slb, sub = self.model.generate_boundary_conditions_for_multi(t, x, is_last=is_last)
            spec_lbs += slb
            spec_ubs += sub

        # Again dirty but effective
        lbs = glb + spec_lbs
        ubs = gub + spec_ubs

        return lbs, ubs


    def generate_profile_idxes(self, result):
        profile_idxes = self.model.generate_multi_profile_idxes(result)

        return profile_idxes
