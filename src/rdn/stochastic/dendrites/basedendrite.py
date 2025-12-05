'''
Implementation for the abstract dendrite class.
For now we assume that 1 space step = 1um and that we have a spine for
each um
'''

import numpy as np
from typing import Type
from abc import ABC, abstractmethod

from ..spinemodels import BaseSpineModel

class BaseDendrite(ABC):
    def __init__(self,
                 spine_model: Type[BaseSpineModel], # the model describing the spine behaviour
                 p_outs: np.ndarray) -> None:
        
        self.spine_model = spine_model
        self.p_outs = p_outs
        self.p_ins = spine_model.compute_pins(p_outs)

    @abstractmethod
    def step(self):
        pass 

    @abstractmethod
    def simulate(self):
        pass


                 
