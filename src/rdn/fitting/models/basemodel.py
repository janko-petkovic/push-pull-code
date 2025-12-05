'''
Abstract base model class
'''

from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):

    def __init__(self):
        pass

    def __call__(self, nss_weights, t_mesh, x_mesh, p):
        return self.forward(nss_weights, t_mesh, x_mesh, p)

    def generate_nss_weights(self, nss, t, x) -> torch.Tensor:
        nss_weights = torch.ones(len(x))*nss
        nss_weights = torch.where(x<0, 2*nss-2, nss_weights)
        nss_weights = torch.where(x>0, 2, nss_weights)
        nss_weights = torch.tile(nss_weights, (len(t), 1))

        return nss_weights


    @abstractmethod
    def forward(self, nss_weights, t_mesh, x_mesh, p):
        pass


    @abstractmethod
    def generate_parameter_names(self, t, x):
        pass

    @abstractmethod
    def generate_parameter_guess(self, t, x, p_guess=10.):
        pass

    @abstractmethod
    def generate_boundary_conditions(self, t, x, p_optim):
        pass

    # Interface for multi
    @abstractmethod
    def generate_parameter_names_for_multi(self, t, x):
        pass

    @abstractmethod
    def generate_boundary_conditions_for_multi(self, t, x):
        '''
        Returns in order blb, gub, slb, gub
        '''
        pass

    @abstractmethod
    def parse_p_from_multi(self, offset, nss_weights, p_multi):
        pass

    @abstractmethod
    def generate_multi_profile_idxes(self, multi_result):
        pass



