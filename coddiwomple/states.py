"""
States Module
"""
#####Imports#####
import numpy as np
import copy
from openmmtools.states import SamplerState
from simtk import unit


class ParticleState():
    """
    Generalized ParticleState object to hold a particle state x and potentially other attributes
    """
    def __init__(self, positions, **kwargs):
        """
        generalized init method

            arguments
                positions : np.ndarray
                    current positions of the state's latent variables
        """
        pass

class PDFState():
    """
    Generalized PDFState object that defines a parametrizable probability distribution function
    """
    def __init__(self, **kwargs):
        pass

    def reduced_potential(self, particle_state):
        """
        compute the 'reduced potential energy' of a ParticleState

        arguments
            particle_state : ParticleState
                state of the particle

        returns
            reduced_potential : float
                reduced potential
        """
        pass

    def set_parameters(self, parameters):
        """
        update the pdf_state parameters in place

        arguments
            parameters : np.array or float or dict
                generalizable parameterization object
        """
        pass

    def get_parameters(self):
        """
        return the current parameters

        return
            returnable_dict : dict
                dictionary of the current parameter names and values {str: float}
        """
        pass
