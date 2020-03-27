"""
Distribution Factory Module
"""
#####Imports#####
import torch
import numpy as np
import logging
import os

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("distribution_factories")
_logger.setLevel(logging.DEBUG)


class DistributionFactory():
    """
    Manage a PDFState.
    """
    def __init__(self, pdf_state, parameter_sequence, **kwargs):
        """
        Initialize the pdf_state

        arguments
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function
            parameter_sequence : list
                sequence to parametrize the pdf_state


        parameters
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function
            parameter_sequence : iterable
                sequence to parametrize the pdf_state
        """
        self.pdf_state = pdf_state
        self.parameter_sequence = parameter_sequence
        _logger.debug(f"equipped pdf state and parameter sequence")

    def update_pdf(self, sequence_index):
        """
        update the PDF with a new sequence_index

        arguments
            sequence_index : int
                the index pointing to the new sequence parameters
        """
        parameters = self.parameter_sequence[sequence_index]
        self.pdf_state.set_parameters(parameters)


    def update_parameter_sequence(self, new_parameters):
        """
        modify the parameter sequence in place

        arguments
            new_parameters : dict or np.ndarray
                the new parameters to create
        """
        new_parameters_type, parameter_sequence_type = type(new_parameters), type(self.parameter_sequence[0])
        assert new_parameters_type == parameter_sequence_type, f"the new_parameters type ({new_parameters_type}) is not the same as the parameter sequence element type ({parameter_sequence_type})"
        self.parameter_sequence.append(new_parameters)


class TargetFactory(DistributionFactory):
    """
    Manage Target probability distributions.
    """
    def __init__(self, pdf_state, parameter_sequence, termination_parameters = None, **kwargs):
        """
        Initialize the pdf_state

        arguments
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function
            parameter_sequence : list
                sequence to parametrize the pdf_state
            termination_parameters : type(parameter_sequence[0]), default None
                parameters to trigger a termination
                if None, then the termination parameters will be defined as parameter_sequence[-1]

        attributes
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function
            parameter_sequence : iterable
                sequence to parametrize the pdf_state
        """
        super().__init__(pdf_state, parameter_sequence)
        if termination_parameters is None:
            self.termination_parameters = self.parameter_sequence[-1]
        else:
            termination_parameters_type, parameter_type = type(termination_parameters), type(self.parameter_sequence[0])
            assert termination_parameters_type == parameter_type, f"The termination parameters type ({termination_parameters_type}) do not match the sequence parameters type ({parameter_type})"
            self.termination_parameters = termination_parameters
        _logger.debug(f"set termination parameters as: {self.termination_parameters}")

    def compute_incremental_work(self, particle, neglect_proposal_work = False, **kwargs):
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
            neglect_proposal_work : bool, default False
                whether to remove the proposal work contribution from the incremental work

        returns
            incremental_work : float
                the incremental work of the particle at time t
        """
        #first, update the pdf state
        iteration = particle.iteration
        self.update_pdf(iteration)

        #then compute the reduced potential of the current particle state at the current target distribution
        u_t = self.pdf_state.reduced_potential(particle.state)

        #then compute the work
        state_work = u_t - particle.auxiliary_work
        proposal_work = particle.proposal_work[-1] if neglect_proposal_work else 0.
        incremental_work = state_work + proposal_work

        return incremental_work


    def terminate(self, particle):
        """
        Decide whether to terminate the particle from the SMC algorithm

        arguments
            particle : coddiwomple.particles.Particle
                particle to query

        returns
            terminate_bool : bool
                whether to terminate the particle
        """
        current_particle_parameters = self.parameter_sequence[len(particle.incremental_works)]
        terminate_bool = True if current_particle_parameters == self.termination_parameters else False
        return terminate_bool


class ProposalFactory(DistributionFactory):
    """
    Manage Proposal probability distributions
    """
    def __init__(self, pdf_state, parameter_sequence, propagator, **kwargs):
        """
        Initialize the pdf_state

        arguments
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function
            parameter_sequence : list
                sequence to parametrize the pdf_state
            propagator : coddiwomple.propagators.Propagator
                the propagator of dynamics


        attributes
            pdf_state : coddiwomple.states.PDFState
                the generalized PDFState object representing a parametrizable probability distribution function
            parameter_sequence : iterable
                sequence to parametrize the pdf_state
        """
        super(ProposalFactory, self).__init__(pdf_state, parameter_sequence)

        self._propagator = propagator
        _logger.debug(f"successfully equipped propagator: {self._propagator}")


    def equip_initial_sample(self, particle, initial_particle_state, generation_pdf = None, **kwargs):
        """
        Equip a particle with an initial particle state.
            1. update particle state
            2. update proposal work
            3. update auxiliary_work

        arguments
            particle : coddiwomple.particles.Particle
                the particle for which to generate an initial sample
            initial_particle_state : coddiwomple.states.ParticleState
                the initial particle state to equip
            generation_pdf : coddiwomple.states.PDFState, default None
                the pdf that generates the initial particle state;
                if None, the generation pdf is taken to be self.pdf_state.

        return
            initial_work : float
                the -log(weight_0)
        """
        assert particle.state is None, f"the particle state is not None; it looks like the particle has already been Initialized"
        if generation_pdf is None:
            self.update_pdf(0)
            generation_pdf = self.pdf_state
        else:
            generation_pdf = initial_pdf_state

        state_reduced_potential = self.pdf_state.reduced_potential(initial_particle_state)
        generation_reduced_potential = generation_pdf.reduced_potential(initial_particle_state)
        initial_work = state_reduced_potential - generation_reduced_potential

        particle.update_state(initial_particle_state) #update the state
        particle.update_proposal_work(0.) #the initial work has no proposal work contribution
        #particle.update_work(initial_work) #we don't actually want to do this until the resampler says its ok
        particle.update_auxiliary_work(-state_reduced_potential) #we need this variable for the next increment
        return initial_work

    def propagate(self, particle, **kwargs):
        """
        Propagate a particle's state and update in place

        arguments
            particle : coddiwomple.particles.ParticleState
                the particle to propagate
        """
        #first, update the pdf state
        iteration = len(particle.incremental_works)
        self.update_pdf(iteration)
        _logger.debug(f"propagating at iteration: {iteration}...")

        #then, propagate and update the state/proposal_work
        state, proposal_work = self._propagator.apply(self.pdf_state, particle.state, **kwargs)
        particle.update_state(state)
        particle.update_proposal_work(proposal_work)

    def propagate(self, particle, num_applications = 1, **kwargs):
        """
        apply an equipped 'MCMCMove' (possibly determined by an invariant distribution defined by pdf_state (a PDFState)) to a particle_state (i.e. a ParticleState)

        arguments
            particle : coddiwomple.particles.Particle
                the particle to propagate
            num_applications : int, default 1
                the number of times to sequentially apply

        returns
            particle : coddiwomple.particles.Particle
                the updated particle
            proposal_work : float
                -log(weight)
        """
        #first, update the pdf state
        iteration = particle.iteration
        self.update_pdf(iteration)
        _logger.debug(f"propagating at iteration: {iteration}...")

        _logger.debug(f"propagating {num_applications}...")
        proposal_work = 0.
        particle_state = particle.state
        for i in range(num_applications):
            _logger.debug(f"applying move iteration {i+1}/{num_applications}...")
            particle_state, work = self._propagator.apply(self.pdf_state, particle_state, **kwargs)
            proposal_work += work
        _logger.debug(f"total accumulated proposal work: {proposal_work}")

        particle.update_state(particle_state)
        particle.update_proposal_work(proposal_work)

        return particle, proposal_work
