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
        initialize a propagator
        """
        pass

    def propagate(self, particle_state, pdf_state, num_iterations, **kwargs):
        """
        apply an equipped 'MCMCMove' (possibly determined by an invariant distribution defined by pdf_state (a PDFState)) to a particle_state (i.e. a ParticleState)

        arguments
            particle_state : coddiwomple.states.ParticleState
                the particle state to which we apply the move
            pdf_state : coddiwomple.states.PDFState
                the pdf_state that (may) govern the self._move;
                the pdf_state should be parametrized properly here
            num_iterations : int
                the number of times to sequentially run the self._move object

        returns
            particle_state : coddiwomple.states.ParticleState
                the particle state to which we apply the move
                WARNING : the particle_state is updated in place
            proposal_work : float
                -log(weight)
        """
        proposal_work = 0.
        for i in range(num_iterations):
            proposal_work += self.apply(particle_state = particle_state, pdf_state = pdf_state, **kwargs)

        return particle_state, proposal_work
