"""
Distribution Factory Module
"""
#####Imports#####
import torch
import numpy as np


class TargetFactory():
    """
    Manage Target probability distributions.
    """
    def __init__(self, pdf_state, **kwargs):
        """
        Initialize the pdf_state

        arguments
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function

        parameters
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function

        """
        self.pdf_state = pdf_state




class ProposalFactory():
    """
    Manage Proposal probability distributions
    """
    def __init__(self, pdf_state, **kwargs):
        """
        Initialize the pdf_state

        arguments
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function

        parameters
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function

        """
        self.pdf_state = pdf_state

    def generate_initial_sample(self, particle, intial_pdf_state = None, **kwargs):
        """
        Pull an initial i.i.d sample from the pdf_state

        arguments
            particle : coddiwomple.particles.Particle
                the particle for which to generate an initial sample
            initial_pdf_state : coddiwomple.states.PDFState
                inital pdf state from which to pull an i.i.d sample (to which an importance weight will be computed)

        return
            incremental_work : float
                the -log(weight_0)
        """
        assert particle._state is None, f"the particle state is not None; it looks like the particle has already been Initialized"
        if initial_pdf_state is None:
            generation_pdf = self.pdf_state
        else:
            generation_pdf = initial_pdf_state

        returnable_particle_state = self._generate_initial_sample(generation_pdf, **kwargs)
        state_reduced_potential = self.pdf_state.reduced_potential(returnable_particle_state)
        generation_reduced_potential = self.pdf_state.reduced_potential(returnable_particle_state)
        initial_work = state_reduced_potential - generation_reduced_potential

        particle.update_state(returnable_particle_state) #update the state
        particle.set_new_state_work(initial_work) #update the state work
        particle.update_work(particle.get_last_state_work()) #update the total work
        particle.update_proposal_works(0.) #there is no proposal work in the first iteration
        particle.set_new_state_works(state_reduced_potential) #create a new state work increment with the current reduced potential

        return

    def compute_incremental_weight(self, particle, **kwargs):
        """
        compute an incremental work where work_(inc,t) = u_t(x_t) - u_(t-1)(x_(t-1)) + proposal_work_t;
        where proposal_work_t = log(K_t(x_t | x_(t-1))) - log(L_(t-1)(x_(t-1) | x_t)).

            - x_t is the current state of the particle for which a 'reduced potential' (u_t(x_t)) is to be computed
            - u_(t-1)(x_(t-1)) is the reduced potential of the previous state at the previously defined PDFState;
                this quantity is held by the particle from the previous iteration (i.e. Particle.get_last_state_work())
            - proposal_work_t is the log ratio of forward to reverse proposal kernels at time t and t-1 (by convention), respectively
                this quantity is held by the partile from the propagation at the current (t) iteration (i.e. Partice.get_last_proposal_work())

        arguments
            particle : coddiwomple.particles.Particle
                the particle for which to generate an initial sample

        returns
            incremental_work : float
                the incremental work of the particle at time t
        """



    def _generate_initial_sample(self):
        """
        Dummy sample initialization
        """
        pass
