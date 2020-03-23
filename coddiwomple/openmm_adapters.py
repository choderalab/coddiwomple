"""
OpenMM Adapter Module:
This module is a consolidation of standard SMC targets, proposals, etc. specifically for openMM interoperability.
"""

#####Imports#####
from coddiwomple.states import ParticleState, PDFState
from coddiwomple.targets import TargetFactory
from coddiwomple.proposals import ProposalFactory
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState, IComposableStates
from coddiwomple.propagators import MCMCMove
from openmmtools import cache, utils
from perses.dispersed.utils import check_platform, configure_platform #TODO: make sure this is functional and supports mixed precision
from simtk import unit
import simtk.openmm as openmm

#define the cache
cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())


#ParticleState Adapter
class OpenMMParticleState(ParticleState, SamplerState):
    """
    ParticleState for openmmtools.states.SamplerState
    """
    def __init__(self, positions, velocities = None, box_vectors = None, **kwargs):
        """
        create a SamplerState

        parameters
            positions : np.array(N,3) * unit.nanometers (or length units)
                positions of state
            velocities : np.array(N,3) * unit.nanometers / unit.picoseconds (or velocity units)
                velocities of state
            box_vectors : np.array(3,3) * unit.nanometers (or velocity units)
                current box vectors
        """
        super(OpenMMParticleState, self).__init__(positions = positions, velocities = velocities, box_vectors = box_vectors)

#PDFState Adapter
class OpenMMPDFState(PDFState, CompoundThermodynamicState):
    """
    PDFState for openmmtools.states.CompoundThermodynamicState
    """

    def __init__(self,
                 system,
                 temperature = 300 * unit.kelvin,
                 pressure = 1.0 * unit.atmosphere,
                 **kwargs):
        """
        Subclass of coddiwomple.states.PDFState that specifically handles openmmtools.states.CompoundThermodynamicStates
        Init method does the following:
            1. create a openmmtools.states.ThermodynamicState with the given system, temperature, and pressure
            2. create an openmmtools.alchemy.AlchemicalState from the given system
            3. create an openmtools.states.CompoundThermodynamicState 1 and 2

        arguments
            system : openmm.System
                system object to wrap
            temperature : float * unit.kelvin (or temperature units), default 300.0 * unit.kelvin
                temperature of the system
            pressure : float * unit.atmosphere (or pressure units), default 1.0 * unit.atmosphere
                pressure of the system

        init method is adapted from https://github.com/choderalab/openmmtools/blob/110524bc5079af77d31f5fab464edd7b668ff5ac/openmmtools/states.py#L2766-L2783
        """
        openmm_pdf_state = ThermodynamicState(system, temperature, pressure)
        alchemical_state = AlchemicalState.from_system(system, **kwargs)
        assert isinstance(alchemical_state, IComposableStates), f"alchemical state is not an instance of IComposableStates"
        self.__dict__ = openmm_pdf_state.__dict__
        self._composable_states = [alchemical_state]
        self.set_system(self._standard_system, fix_state=True)

    def set_parameters(self, parameters):
        """
        update the pdf_state parameters in place

        arguments
            parameters : dict
                dictionary of variables
        """
        assert set([param for param in self._parameters.keys() if param is not None]) == set(parameters.keys()), f"the parameter keys supplied do not match the internal parameter names"
        for key,val in parameters.items():
            setattr(self, key, val)

    def get_parameters(self):
        """
        return the current parameters

        return
            returnable_dict : dict
                dictionary of the current parameter names and values {str: float}
        """
        returnable_dict = {i:j for i, j in _composable_states[0]._parameters.items() if j is not None}
        return returnable_dict




#TargetFactory Adapter
class OpenMMTargetFactory(TargetFactory):
    """
    Adapter for TargetFactory
    """
    def __init__(self, openmm_pdf_state, **kwargs):
        """
        Initialize the OpenMMTargetFactory

        arguments
            openmm_pdf_state : OpenMMPDFState
                thermodynamic state of the targets
        """
        super(TargetFactory, self).__init__(pdf_state = openmm_pdf_state, **kwargs)


#ProposalFactory Adapter
class OpenMMProposalFactory(ProposalFactory):
    """
    Adapter for ProposalFactory
    """
    def __init__(self, openmm_pdf_state, **kwargs):
        """
        Initialize the OpenMMTargetFactory

        arguments
            openmm_pdf_state : OpenMMPDFState
                thermodynamic state of the targets
        """
        super(ProposalFactory, self).__init__(pdf_state = openmm_pdf_state, **kwargs)

#Propagator Adapter
class OpenMMBaseIntegratorMove(Propagator, openmmtools.mcmc.BaseIntegratorMove):
    """
    Generalized OpenMMTools Integrator Propagator
    """
    def __init__(self,
                 n_steps,
                 context_cache=None,
                 reassign_velocities=False,
                 n_restart_attempts=4):

        """
        call the BaseIntegratorMove init method
        """
        super(OpenMMIntegratorPropagator, self).__init__(n_steps = n_steps,
                                                         context_cache = context_cache,
                                                         reassign_velocities = reassign_velocities,
                                                         n_restart_attempts = n_restart_attempts)

     def apply(self, openmm_pdf_state, particle_state):
        """
        Propagate the state through the integrator.
        This updates the particle_state after the integration. It also logs
        benchmarking information through the utils.Timer class.

        The only reason we are subclassing this is so that we can pull the heat (i.e. the )

        arguments
            openmm_pdf_state : OpenMMPDFState
                thermodynamic state of the targets
            particle_state : OpenMMParticleState
               The state to apply the move to. This is modified.

        returns
            particle_state : OpenMMParticleState
                The state to apply the move to. This is modified.
            proposal_work : float
                log ratio of the forward to reverse proposal kernels

        See Also
        --------
        openmmtools.utils.Timer
        """
        move_name = self.__class__.__name__  # shortcut
        timer = Timer()

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create integrator.
        integrator = self._get_integrator(openmm_pdf_state)

        # Create context.
        timer.start("{}: Context request".format(move_name))
        context, integrator = context_cache.get_context(openmm_pdf_state, integrator)
        integrator.reset()
        timer.stop("{}: Context request".format(move_name))
        #logger.debug("{}: Context obtained, platform is {}".format(
        #    move_name, context.getPlatform().getName()))

        # Perform the integration.
        for attempt_counter in range(self.n_restart_attempts + 1):

            # If we reassign velocities, we can ignore the ones in particle_state.
            particle_state.apply_to_context(context, ignore_velocities=self.reassign_velocities)
            if self.reassign_velocities:
                context.setVelocitiesToTemperature(openmm_pdf_state.temperature)

            # Subclasses may implement _before_integration().
            self._before_integration(context, openmm_pdf_state)

            try:
                # Run dynamics.
                timer.start("{}: step({})".format(move_name, self.n_steps))
                integrator.step(self.n_steps)
            except Exception:
                # Catches particle positions becoming nan during integration.
                restart = True
            else:
                timer.stop("{}: step({})".format(move_name, self.n_steps))

                # We get also velocities here even if we don't need them because we
                # will recycle this State to update the sampler state object. This
                # way we won't need a second call to Context.getState().
                context_state = context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                                                 enforcePeriodicBox=openmm_pdf_state.is_periodic)

                # Check for NaNs in energies.
                potential_energy = context_state.getPotentialEnergy()
                restart = np.isnan(potential_energy.value_in_unit(potential_energy.unit))

            # Restart the move if we found NaNs.
            if restart:
                err_msg = ('Potential energy is NaN after {} attempts of integration '
                           'with move {}'.format(attempt_counter, self.__class__.__name__))

                # If we are on our last chance before crash, try to re-initialize context
                if attempt_counter == self.n_restart_attempts - 1:
                    logger.error(err_msg + ' Trying to reinitialize Context as a last-resort restart attempt...')
                    context.reinitialize()
                    particle_state.apply_to_context(context)
                    openmm_pdf_state.apply_to_context(context)
                # If we have hit the number of restart attempts, raise an exception.
                elif attempt_counter == self.n_restart_attempts:
                    # Restore the context to the state right before the integration.
                    from openmmtools.mcmc import IntegratorMoveError
                    particle_state.apply_to_context(context)
                    logger.error(err_msg)
                    raise IntegratorMoveError(err_msg, self, context)
                else:
                    logger.warning(err_msg + ' Attempting a restart...')
            else:
                break

        # Subclasses can read here info from the context to update internal statistics.
        self._after_integration(context, openmm_pdf_state)

        # Updated sampler state.
        timer.start("{}: update sampler state".format(move_name))
        # This is an optimization around the fact that Collective Variables are not a part of the State,
        # but are a part of the Context. We do this call twice to minimize duplicating information fetched from
        # the State.
        # Update everything but the collective variables from the State object
        particle_state.update_from_context(context_state, ignore_collective_variables=True)
        # Update only the collective variables from the Context
        particle_state.update_from_context(context, ignore_positions=True, ignore_velocities=True,
                                          ignore_collective_variables=False)
        timer.stop("{}: update sampler state".format(move_name))

        proposal_work = -integrator.get_heat()
        return particle_state, proposal_work

class OpenMMLangevinDynamicsMove(OpenMMBaseIntegratorMove):
    """
    Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move assigns a velocity from the Maxwell-Boltzmann distribution
    and executes a number of Maxwell-Boltzmann steps to propagate dynamics.
    This is not a *true* Monte Carlo move, in that the generation of the
    correct distribution is only exact in the limit of infinitely small
    timestep; in other words, the discretization error is assumed to be
    negligible. Use HybridMonteCarloMove instead to ensure the exact
    distribution is generated.
    .. warning::
        No Metropolization is used to ensure the correct phase space
        distribution is sampled. This means that timestep-dependent errors
        will remain uncorrected, and are amplified with larger timesteps.
        Use this move at your own risk!

    arguments
        timestep : simtk.unit.Quantity, optional
            The timestep to use for Langevin integration
            (time units, default is 1*simtk.unit.femtosecond).
        collision_rate : simtk.unit.Quantity, optional
            The collision rate with fictitious bath particles
            (1/time units, default is 10/simtk.unit.picoseconds).
        n_steps : int, optional
            The number of integration timesteps to take each time the
            move is applied (default is 1000).
        reassign_velocities : bool, optional
            If True, the velocities will be reassigned from the Maxwell-Boltzmann
            distribution at the beginning of the move (default is False).
        context_cache : openmmtools.cache.ContextCache, optional
            The ContextCache to use for Context creation. If None, the global cache
            openmmtools.cache.global_context_cache is used (default is None).
    """

    def __init__(self,
                 timestep=1.0*unit.femtosecond,
                 collision_rate=10.0/unit.picoseconds,
                 n_steps=1000,
                 reassign_velocities=False,
                 **kwargs):
        super(OpenMMLangevinDynamicsMove, self).__init__(n_steps=n_steps,
                                                         reassign_velocities=reassign_velocities,
                                                         **kwargs)

        self.timestep = timestep
        self.collision_rate = collision_rate

    def apply(self,
              openmm_pdf_state,
              particle_state):
        """
        Apply the Langevin dynamics MCMC move.
        This modifies the given particle_state. The temperature of the
        thermodynamic state is used in Langevin dynamics.

        arguments
            openmm_pdf_state : openmmtools.states.ThermodynamicState
               The thermodynamic state to use to propagate dynamics.
            particle_state : openmmtools.states.SamplerState
               The sampler state to apply the move to. This is modified.

        returns
            particle_state : OpenMMParticleState
                The state to apply the move to. This is modified.
            proposal_work : float
                log ratio of the forward to reverse proposal kernels
        """
        # Explicitly implemented just to have more specific docstring.
        particle_state, proposal_work = super(LangevinDynamicsMove, self).apply(openmm_pdf_state, particle_state)
        return particle_state, proposal_work
