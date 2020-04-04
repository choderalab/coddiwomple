"""
OpenMM Propagator Adapter Module
"""

#####Imports#####
from coddiwomple.propagators import Propagator
from openmmtools import cache, utils
from openmmtools.utils import Timer
from openmmtools import mcmc
from perses.dispersed.utils import check_platform, configure_platform #TODO: make sure this is functional and supports mixed precision
from simtk import unit
import simtk.openmm as openmm
from coddiwomple.openmm.utils import get_dummy_integrator
import os
import numpy as np
import logging
from copy import deepcopy

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("openmm_propagators")
_logger.setLevel(logging.DEBUG)

#define the cache
cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())

#Propagator Adapter
class OMMBIP(mcmc.BaseIntegratorMove, Propagator):
    """
    Generalized OpenMM Base Integrator Propagator
    """
    def __init__(self,
                 openmm_pdf_state,
                 integrator,
                 context_cache=None,
                 reassign_velocities=False,
                 n_restart_attempts=0):

        """
        call the BaseIntegratorMove init method

        arguments
            openmm_pdf_state : coddiwomple.openmm_adapters.OpenMMPDFState
                the pdf state of the propagator
            integrator : openmm.Integrator
                integrator of dynamics
            context_cache : openmmtools.cache.ContextCache, optional
                The ContextCache to use for Context creation. If None, the global cache
                openmmtools.cache.global_context_cache is used (default is None).
            reassign_velocities : bool, optional
                If True, the velocities will be reassigned from the Maxwell-Boltzmann
                distribution at the beginning of the move (default is False).
            n_restart_attempts : int, optional
                When greater than 0, if after the integration there are NaNs in energies,
                the move will restart. When the integrator has a random component, this
                may help recovering. On the last attempt, the ``Context`` is
                re-initialized in a slower process, but better than the simulation
                crashing. An IntegratorMoveError is raised after the given number of
                attempts if there are still NaNs.

        attributes
            pdf_state : coddiwomple.openmm_adapters.OpenMMPDFState
            integrator :  openmm.Integrator
            context_cache : openmmtools.cache.ContextCache
            reassign_velocities : bool
            n_restart_attempts : int or None
        """
        super().__init__(n_steps = None,
                         context_cache = context_cache,
                         reassign_velocities = reassign_velocities,
                         n_restart_attempts = n_restart_attempts)

        _logger.debug(f"successfully executed {mcmc.BaseIntegratorMove.__class__.__name__} init.")

        self.pdf_state = openmm_pdf_state

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create context and reset integrator for good measure
        self.context, self.integrator = context_cache.get_context(self.pdf_state, integrator)
        self.integrator.reset()

        _logger.debug(f"successfully equipped integrator: {self.integrator.__class__.__name__}")
        _logger.debug(f"integrator printable: {self.integrator.pretty_print()}")

    def apply(self,
              particle_state,
              n_steps = 1,
              reset_integrator = False,
              apply_pdf_to_context = False,
              **kwargs):
        """
        Propagate the state through the integrator.
        This updates the particle_state after the integration. It also logs
        benchmarking information through the utils.Timer class.

        arguments
            particle_state : OpenMMParticleState
               The state to apply the move to. This is modified.
            n_steps : int, default 1
                number of steps to apply to the integrator
            reset_integrator : bool, default False
                whether to reset the integrator
            apply_pdf_to_context : bool, default False
                whether to self.pdf_state.apply_to_context


        returns
            particle_state : OpenMMParticleState
                The state to apply the move to. This is modified.
            global_integrator_variables : dict
                dict of integrator global variables

        see also
            openmmtools.utils.Timer
        """
        move_name = self.__class__.__name__  # shortcut

        # reset the integrator
        if reset_integrator:
            self.integrator.reset()
        if apply_pdf_to_context:
            self.pdf_state.apply_to_context(self.context)

        # Perform the integration.
        for attempt_counter in range(self.n_restart_attempts + 1):
            # If we reassign velocities, we can ignore the ones in particle_state.
            particle_state.apply_to_context(self.context, ignore_velocities=self.reassign_velocities)
            if self.reassign_velocities:
                self.context.setVelocitiesToTemperature(self.pdf_state.temperature)

            # Subclasses may implement _before_integration().
            self._before_integration(particle_state,
                                     n_steps,
                                     reset_integrator,
                                     apply_pdf_to_context,
                                     **kwargs)

            try:
                for _ in range(n_steps):
                    self.integrator.step(1)
                    self._during_integration(particle_state,
                                             n_steps,
                                             reset_integrator,
                                             apply_pdf_to_context,
                                             **kwargs)
            except Exception as e:
                # Catches particle positions becoming nan during integration.
                _logger.warning(f"Exception raised: {e}")
                restart = True
            else:
                # We get also velocities here even if we don't need them because we
                # will recycle this State to update the sampler state object. This
                # way we won't need a second call to Context.getState().
                context_state = self.context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                                                 enforcePeriodicBox=self.pdf_state.is_periodic)

                # Check for NaNs in energies.
                potential_energy = context_state.getPotentialEnergy()
                restart = np.isnan(potential_energy.value_in_unit(potential_energy.unit))

            # Restart the move if we found NaNs.
            if restart:
                err_msg = ('Potential energy is NaN after {} attempts of integration '
                           'with move {}'.format(attempt_counter, self.__class__.__name__))

                # If we are on our last chance before crash, try to re-initialize context
                if attempt_counter == self.n_restart_attempts - 1:
                    _logger.error(err_msg + ' Trying to reinitialize Context as a last-resort restart attempt...')
                    self.context.reinitialize()
                    self.integrator.reset()
                    particle_state.apply_to_context(self.context)
                    self.pdf_state.apply_to_context(self.context)
                # If we have hit the number of restart attempts, raise an exception.
                elif attempt_counter == self.n_restart_attempts:
                    # Restore the context to the state right before the integration.
                    particle_state.apply_to_context(self.context)
                    _logger.error(err_msg)
                    raise mcmc.IntegratorMoveError(err_msg, self, self.context)
                else:
                    _logger.warning(err_msg + ' Attempting a restart...')
            else:
                break

        # Subclasses can read here info from the context to update internal statistics.
        self._after_integration(particle_state,
                                 n_steps,
                                 reset_integrator,
                                 apply_pdf_to_context,
                                 **kwargs)

        # Updated sampler state.
        # This is an optimization around the fact that Collective Variables are not a part of the State,
        # but are a part of the Context. We do this call twice to minimize duplicating information fetched from
        # the State.
        # Update everything but the collective variables from the State object
        particle_state.update_from_context(context_state, ignore_collective_variables=True)
        # Update only the collective variables from the Context
        particle_state.update_from_context(self.context, ignore_positions=True, ignore_velocities=True,
                                          ignore_collective_variables=False)

        global_integrator_variables = self._get_global_integrator_variables()

        return particle_state, global_integrator_variables

    def _get_integrator(self):
        pass

    def _get_global_integrator_variables(self):
        """
        return a dictionary of the self.integrator's global variables

        returns
            global_integrator_variables : dict
                {global variable name <str> : global variable value <float>}
        """
        num_global_vars = self.integrator.getNumGlobalVariables()
        global_integrator_variables = {self.integrator.getGlobalVariableName(idx): self.integrator.getGlobalVariable(idx) for idx in range(num_global_vars)}
        return global_integrator_variables

    def _get_context_parameters(self):
        """
        return a dictionary of the self.context's parameters

        returns
            context_parameters : dict
            {parameter name <str> : parameter value value <float>}
        """
        swig_parameters = self.context.getParameters()
        context_parameters = {q: swig_parameters[q] for q in swig_parameters}
        return context_parameters

    def _before_integration(self,
                            *args,
                            **kwargs):
        pass

    def _during_integration(self,
                            *args,
                            **kwargs):
        pass

    def _after_integration(self,
                           *args,
                           **kwargs):
        pass


class OMMAISP(OMMBIP):
    """
    OpenMM Annealed Importance Sampling Propagator

    This propagator is equipped with a coddiwomple.openmm.integrators.OMMLIAIS integrator or a subclass thereof.
    The purpose is to allow for the management and validation of a singly-parameterized annealing protocol (i.e. 'fractional_iteration')

    Ensure that the self.apply() step will anneal the FULL protocol (i.e. fractional_iteration = 0., 1.) for all parameters
    """
    def __init__(self,
                 openmm_pdf_state,
                 integrator,
                 record_state_work_interval = None,
                 context_cache=None,
                 reassign_velocities=False,
                 n_restart_attempts=0):
        """
        see super

        arguments:
            record_state_work_interval : int, default None
                frequency with which to record the state work
                if None, the state work is never calculated during integration
        """

        super().__init__(openmm_pdf_state,
                         integrator,
                         context_cache=context_cache,
                         reassign_velocities=reassign_velocities,
                         n_restart_attempts=n_restart_attempts)

        #and there is one more validation that has to happen...
        pdf_state_parameters = list(self.pdf_state.get_parameters().keys())
        function_parameters = self.integrator._function_parameters
        assert set(pdf_state_parameters) == set(function_parameters), f"the pdf_state parameters ({pdf_state_parameters}) is not equal to the function parameters ({function_parameters})"
        self._record_state_work_interval = record_state_work_interval
        self._state_works = {}
        self._state_works_counter = 0

    def _before_integration(self, *args, **kwargs):
        """
        make sure that the context parameters are all 0.0
        """
        context_parameters = self._get_context_parameters()
        _logger.debug(f"\tcontext_parameters before integration: {context_parameters}")
        self._current_state_works = []
        if self._record_state_work_interval is not None:
            self._current_state_works.append(0.0)

    def _during_integration(self, *args, **kwargs):
        """
        update the state work
        """
        if self._record_state_work_interval is not None:
            integrator_variables = self._get_global_integrator_variables()
            iteration = integrator_variables['iteration']
            if iteration % self._record_state_work_interval == 0:
                self._current_state_works.append(self.integrator.get_state_work())


    def _after_integration(self, *args, **kwargs):
        """
        make sure that he context parameters are all 1.0
        """
        context_parameters = self._get_context_parameters()
        _logger.debug(f"\tcontext_parameters after integration: {context_parameters}")

        integrator_variables = self._get_global_integrator_variables()
        iteration = integrator_variables['iteration']

        if self._record_state_work_interval is not None:
            if iteration % self._record_state_work_interval == 0:
                pass #do not record if the state work was recorded at the last `_during_integration` pass
            else:
                self._current_state_works.append(self.integrator.get_state_work())
        else:
            self._current_state_works.append(self.integrator.get_state_work())

        self._state_works[self._state_works_counter] = deepcopy(self._current_state_works)
        self._state_works_counter += 1



class OMMAISVP(OMMAISP):
    """
    OpenMMAIS Verbose Propagator

    OMMAISVP is a simple subclass of OMMAISP that prints context parameters and integrator variables before, during, and after the integration steps
    """
    def __init__(self,
                 openmm_pdf_state,
                 integrator,
                 record_state_work_interval = None,
                 context_cache=None,
                 reassign_velocities=False,
                 n_restart_attempts=0):

        super().__init__(openmm_pdf_state,
                         integrator,
                         record_state_work_interval = record_state_work_interval,
                         context_cache=context_cache,
                         reassign_velocities=reassign_velocities,
                         n_restart_attempts=n_restart_attempts)

    def _before_integration(self, *args, **kwargs):
        super()._before_integration(*args, **kwargs)
        integrator_variables = self._get_global_integrator_variables()
        _logger.debug(f"\tintegrator_global_variables before integration:")
        for key, val in integrator_variables.items():
            _logger.debug(f"\t\t{key}: {val}")


    def _during_integration(self, *args, **kwargs):
        super()._during_integration(*args, **kwargs)
        integrator_variables = self._get_global_integrator_variables()
        context_parameters = self._get_context_parameters()
        _logger.debug(f"\tcontext_parameters during integration: {context_parameters}")
        _logger.debug(f"\tintegrator_global_variables during integration: ")
        for key, val in integrator_variables.items():
            _logger.debug(f"\t\t{key}: {val}")

    def _after_integration(self, *args, **kwargs):
        super()._after_integration(*args, **kwargs)
        integrator_variables = self._get_global_integrator_variables()
        _logger.debug(f"\tintegrator_global_variables after integration: ")
        for key, val in integrator_variables.items():
            _logger.debug(f"\t\t{key}: {val}")


class OMMAISPR(OMMAISP):
    """
    OpenMMAISP Reportable

    OMMAISP is a simple subclass of OMMAISP that equips an OpenMMReporter object and writes a trajectory to disk at specified iterations
    """
    def __init__(self,
                 openmm_pdf_state,
                 integrator,
                 record_state_work_interval = None,
                 reporter = None,
                 trajectory_write_interval = 1,
                 context_cache=None,
                 reassign_velocities=False,
                 n_restart_attempts=0):
        """
        see super (i.e. OMMAISP)

        arguments (new):
            reporter : coddiwomple.openmm.reporter.OpenMMReporter, default None
                a reporter object to write trajectories
            trajectory_write_interval : int, default 1
                write the trajectory every trajectory_write_interval intervals
        """
        super().__init__(openmm_pdf_state = openmm_pdf_state,
                         integrator = integrator,
                         record_state_work_interval = record_state_work_interval,
                         context_cache=context_cache,
                         reassign_velocities=reassign_velocities,
                         n_restart_attempts=n_restart_attempts)

        self._write_trajectory = False if reporter is None else True
        self.reporter = reporter
        self.particle = None
        self._trajectory_write_interval = trajectory_write_interval if self._write_trajectory else None

    def _before_integration(self, *args, **kwargs):
        """
        update the particle with the particle state
        """
        from coddiwomple.particles import Particle
        super()._before_integration(*args, **kwargs)
        particle_state = args[0]
        self.particle = Particle(index = 0, iteration = 0)
        if self._write_trajectory:
            particle_state.update_from_context(self.context, ignore_velocities=True)
            self.particle.update_state(particle_state)
            self.reporter.record([self.particle])

    def _during_integration(self, *args, **kwargs):
        """
        write trajectory if we are allowed and if we satisfy the interval criterion
        """
        super()._during_integration(*args, **kwargs)
        particle_state = args[0]
        if self._write_trajectory:
            integrator_variables = self._get_global_integrator_variables()
            iteration = integrator_variables['iteration']
            n_iterations = integrator_variables['niterations']
            if iteration % self._trajectory_write_interval == 0:
                particle_state.update_from_context(self.context, ignore_velocities=True)
                if iteration == n_iterations:
                    try:
                        self.reporter.record([self.particle], save_to_disk=True)
                    except Exception as e:
                        _logger.warning(f"{e}")
                    self.reporter.reset()
                else:
                    self.reporter.record([self.particle])

    def _after_integration(self, *args, **kwargs):
        """
        write trajectory if we are allowed and if we satisfy the interval criterion
        """
        super()._after_integration(*args, **kwargs)
        particle_state = args[0]
        if self._write_trajectory:
            particle_state.update_from_context(self.context, ignore_velocities=True)
            integrator_variables = self._get_global_integrator_variables()
            iteration = integrator_variables['iteration']
            if iteration % self._trajectory_write_interval == 0:
                pass ##do not record if the state work was recorded at the last `_during_integration` pass
            else:
                self.reporter.record([particle], save_to_disk = True)
                self.reporter.reset()
