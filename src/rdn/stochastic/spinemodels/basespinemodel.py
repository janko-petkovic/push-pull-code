'''
Base spine model class implementation.
The spine model contains information on how the protein exchange between
spine and dendrite occurs
'''

import numpy as np
from abc import ABC, abstractmethod

class BaseSpineModel(ABC):
    def __init__(self,
                 k_ins: np.ndarray,
                 k_outs: np.ndarray) -> None:

        self.k_ins = k_ins
        self.k_outs = k_outs

    @abstractmethod
    def compute_pins(self, p_outs_mf):
        pass

    @abstractmethod
    def compute_k_ins_t(self):
        pass

    @abstractmethod
    def generate_exiting(self):
        pass
                 


