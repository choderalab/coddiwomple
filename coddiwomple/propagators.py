"""
Propagator Module
"""
#####Imports#####
import logging
import numpy as np
from scipy.special import logsumexp

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("propagators")
_logger.setLevel(logging.DEBUG)

class Propagator():
    """
    Generalized propagator object.
    """
    def __init__(self, **kwargs):
        """
        Generalized Init Method.
        """
        _logger.debug(f"successfully executed {self.__class__.__name__} dummy init.")

    def apply(self, particle_state, **kwargs):
        """
        Dummy MCMC move apply method.

        arguments
            particle_state : coddiwomple.states.ParticleState
                the particle state to which we apply the move

        returns
            particle_state : coddiwomple.states.ParticleState
                the particle state to which we apply the move;
                this is modified in place.
            proposal_work : float
                -log(weight)
        """
        pass
