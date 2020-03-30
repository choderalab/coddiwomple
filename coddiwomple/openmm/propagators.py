"""
OpenMM Propagator Adapter Module
"""

#####Imports#####
from coddiwomple.propagators import Propagator
from openmmtools import cache, utils
from openmmtools import integrators
from openmmtools.utils import Timer
from openmmtools import mcmc
from perses.dispersed.utils import check_platform, configure_platform #TODO: make sure this is functional and supports mixed precision
from simtk import unit
import simtk.openmm as openmm
from coddiwomple.openmm.utils import get_dummy_integrator
import os
import numpy as np
import logging

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("openmm_propagators")
_logger.setLevel(logging.DEBUG)

#define the cache
cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())

#constants
_OPENMM_ENERGY_UNIT = unit.kilojoules_per_mole

#Propagator Adapter
class OpenMMBaseIntegrationPropagator(mcmc.BaseIntegratorMove, Propagator):
    """
    Generalized OpenMMTools Integrator Propagator
    """
    def __init__(self,
                 openmm_pdf_state,
                 integrator,
                 context_cache=None,
                 reassign_velocities=False,
                 n_restart_attempts=4):

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
              return_state_work = False,
              return_proposal_work = False,
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
            return_state_work : bool, default False
                whether to return the state work at the end of the application
            return_proposal_work : bool, default False
                whether to return the proposal work at the end of the application


        returns
            particle_state : OpenMMParticleState
                The state to apply the move to. This is modified.
            returnable_dict : dict
                dictionary of variables asked to return in returnables

        see also
            openmmtools.utils.Timer
        """
        move_name = self.__class__.__name__  # shortcut

        # reset the integrator
        if reset_integrator:
            self.integrator.reset()

        # Perform the integration.
        for attempt_counter in range(self.n_restart_attempts + 1):

            # If we reassign velocities, we can ignore the ones in particle_state.
            particle_state.apply_to_context(self.context, ignore_velocities=self.reassign_velocities)
            if self.reassign_velocities:
                self.context.setVelocitiesToTemperature(openmm_pdf_state.temperature)

            # Subclasses may implement _before_integration().
            self._before_integration(**kwargs)

            try:
                for _ in range(n_steps):
                    self.integrator.step(1)
                    self._during_integration(**kwargs)
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
        self._after_integration(**kwargs)

        # Updated sampler state.
        # This is an optimization around the fact that Collective Variables are not a part of the State,
        # but are a part of the Context. We do this call twice to minimize duplicating information fetched from
        # the State.
        # Update everything but the collective variables from the State object
        particle_state.update_from_context(context_state, ignore_collective_variables=True)
        # Update only the collective variables from the Context
        particle_state.update_from_context(self.context, ignore_positions=True, ignore_velocities=True,
                                          ignore_collective_variables=False)

        returnable_dict = {}
        if return_state_work:
            returnable_dict['state_work'] = self.integrator.get_state_work(dimensionless = True)
        if return_proposal_work:
            returnable_dict['proposal_work'] = self.integrator.get_proposal_work(dimensionless = True)

        return particle_state, returnable_dict

    def _get_integrator(self):
        pass

    def _before_integration(self, **kwargs):
        pass
    def _during_integration(self, **kwargs):
        pass
    def _after_integration(self, **kwargs):
        pass

#####Integrators#####
class OpenMMLangevinIntegrator(integrators.ThermostatedIntegrator):
    """Integrates Langevin dynamics with a prescribed operator splitting.
    One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt
        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass
        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal
    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive but slow-fluctuating forces. Since forces are only
    evaluated in the V step, we represent this by including in our "alphabet" V0, V1, ...)
    When the system contains holonomic constraints, these steps are confined to the constraint
    manifold.

    examples
        - VVVR
            splitting="O V R V O"
        - BAOAB:
            splitting="V R O R V"
        - g-BAOAB, with K_r=3:
            splitting="V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            splitting="V0 V1 R R O R R V1 R R O R R V1 V0"
    arguments
        _kinetic_energy : str
            This is 0.5*m*v*v by default, and is the expression used for the kinetic energy
        measure_proposal_work : bool, default False
            whether to measure the

    references
        [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7
        https://github.com/choderalab/openmmtools/blob/c2b61c410b255c4e08927acf8cfcb1cf46f64b70/openmmtools/integrators.py#L1010-L1552

    This class exposes the ability to toggle proposal work measure, which, in this regime, is just the negative heat.
    Metropolization is disables because computing proposal work with a metropolization step is a little bit more difficult.

    ToDo
        implement metropolization with an associated
    """

    _kinetic_energy = "0.5 * m * v * v"

    def __init__(self,
                 temperature=300.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 splitting="V R O R V",
                 constraint_tolerance=1e-8,
                 measure_proposal_work = False,
                 measure_shadow_work = False
                 ):
        """Create a Langevin integrator with the prescribed operator splitting.

        arguments
            splitting : string, default: "V R O R V"
                Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.
                Forces are only used in V-step. Handle multiple force groups by appending the force group index
                to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            temperature : np.unit.Quantity compatible with kelvin, default: 300.0*unit.kelvin
               Fictitious "bath" temperature
            collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
               Collision rate
            timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
               Integration timestep
            constraint_tolerance : float, default: 1.0e-8
                Tolerance for constraint solver
            measure_shadow_work : boolean, default: False
                Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`
            measure_proposal_work : boolean, default: False
                accumulate the proposal work (-heat)
        """
        self.class_name = self.__class__.__name__
        _logger.debug(f"initializing {self.class_name}...")
        # Compute constants
        gamma = collision_rate
        self._gamma = gamma

        # Check if integrator is metropolized by checking for M step:
        if splitting.find("{") > -1:
            self._metropolized_integrator = True
            # We need to measure shadow work if Metropolization is used
            measure_shadow_work = True
            raise Exception(f"metropolization is not currently supported")
        else:
            self._metropolized_integrator = False
        _logger.debug(f"{self.class_name}: metropolization is {self._metropolized_integrator}")

        # Record whether we are measuring heat and shadow work
        self._measure_proposal_work = measure_proposal_work
        self._measure_shadow_work = measure_shadow_work
        _logger.debug(f"{self.class_name}: measure_proposal_work is {self._measure_proposal_work}")
        _logger.debug(f"{self.class_name}: measure_shadow_work is {self._measure_shadow_work}")

        ORV_counts, mts, force_group_nV = self._parse_splitting_string(splitting)
        _logger.debug(f"{self.class_name}: successfully parsed splitting string")

        # Record splitting.
        self._splitting = splitting
        self._ORV_counts = ORV_counts
        self._mts = mts
        self._force_group_nV = force_group_nV

        # Create a new CustomIntegrator
        super().__init__(temperature, timestep)

        # Initialize
        self.addPerDofVariable("sigma", 0)

        # Velocity mixing parameter: current velocity component
        h = timestep / max(1, ORV_counts['O'])
        self.addGlobalVariable("a", np.exp(-gamma * h))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * gamma * h)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Add global variables
        _logger.debug(f"{self.class_name}: adding global variables...")
        self._add_global_variables()

        # Add integrator steps
        _logger.debug(f"{self.class_name}: adding integrator steps...")
        self._add_integrator_steps()

    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = {
            'O': (self._add_O_step, False),
            'R': (self._add_R_step, False),
            '{': (self._add_metropolize_start, False),
            '}': (self._add_metropolize_finish, False),
            'V': (self._add_V_step, True)
        }
        return dispatch_table

    def _add_global_variables(self):
        """Add global bookkeeping variables."""
        self.addGlobalVariable("proposal_work", 0)
        self.addGlobalVariable("old_ke", 0)
        self.addGlobalVariable("new_ke", 0)
        self.addGlobalVariable("state_work", 0)


    def reset_proposal_work(self):
        """Reset proposal_work."""
        if self._measure_proposal_work:
            self.setGlobalVariableByName('proposal_work', 0.0)

    def reset_shadow_work(self):
        """Reset shadow work."""
        if self._measure_shadow_work:
            self.setGlobalVariableByName('shadow_work', 0.0)

    def reset_ghmc_statistics(self):
        """Reset GHMC acceptance rate statistics."""
        if self._metropolized_integrator:
            self.setGlobalVariableByName('ntrials', 0)
            self.setGlobalVariableByName('naccept', 0)
            self.setGlobalVariableByName('nreject', 0)

    def reset(self):
        """Reset all statistics (proposal_work, shadow work, acceptance rates, step).
        """
        self.reset_proposal_work()
        self.reset_shadow_work()
        self.reset_ghmc_statistics()

    def _get_energy_with_units(self, variable_name, dimensionless=False):
        """Retrive an energy/work quantity and return as unit-bearing or dimensionless quantity.
        Parameters
        ----------
        variable_name : str
           Name of the global context variable to retrieve
        dimensionless : bool, optional, default=False
           If specified, the energy/work is returned in reduced (kT) unit.
        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the work in kT (float).
           Otherwise, the unit-bearing work in units of energy.
        """
        work = self.getGlobalVariableByName(variable_name) * _OPENMM_ENERGY_UNIT
        if dimensionless:
            return work / self.kT
        else:
            return work

    def get_shadow_work(self, dimensionless=False):
        """Get the current accumulated shadow work.
        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.
        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the protocol work in kT (float).
           Otherwise, the unit-bearing protocol work in units of energy.
        """
        if not self._measure_shadow_work:
            raise Exception("This integrator must be constructed with 'measure_shadow_work=True' to measure shadow work.")
        return self._get_energy_with_units("shadow_work", dimensionless=dimensionless)

    @property
    def shadow_work(self):
        return self.get_shadow_work()

    def get_proposal_work(self, dimensionless=False):
        """Get the current accumulated proposal_work.

        arguments
            dimensionless : bool, optional, default=False
               If specified, the work is returned in reduced (kT) unit.

        returns
        work : unit.Quantity or float
           If dimensionless=True, the proposal work in kT (float).
           Otherwise, the unit-bearing proposal in units of energy.
        """
        if not self._measure_proposal_work:
            raise Exception("This integrator must be constructed with 'measure_proposal_work=True' in order to measure proposal work.")
        return self._get_energy_with_units("proposal_work", dimensionless=dimensionless)

    @property
    def proposal_work(self):
        return self.proposal_work()

    def get_acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators.
        Returns
        -------
        acceptance_rate : float
           Acceptance rate.
           An Exception is thrown if the integrator is not Metropolized.
        """
        if not self._metropolized_integrator:
            raise Exception("This integrator must be Metropolized to return an acceptance rate.")
        return self.getGlobalVariableByName("naccept") / self.getGlobalVariableByName("ntrials")

    @property
    def acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators."""
        return self.get_acceptance_rate()

    @property
    def is_metropolized(self):
        """Return True if this integrator is Metropolized, False otherwise."""
        return self._metropolized_integrator

    def _add_integrator_steps(self):
        """Add the steps to the integrator--this can be overridden to place steps around the integration.
        """
        # Integrate
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        _logger.debug(f"{self.class_name}: adding substep functions...")
        for i, step in enumerate(self._splitting.split()):
            self._substep_function(step)

    def _sanity_check(self, splitting):
        """Perform a basic sanity check on the splitting string to ensure that it makes sense.
        Parameters
        ----------
        splitting : str
            The string specifying the integrator splitting
        mts : bool
            Whether the integrator is a multiple timestep integrator
        allowed_characters : str, optional
            The characters allowed to be present in the splitting string.
            Default RVO and the digits 0-9.
        """

        # Space is just a delimiter--remove it
        splitting_no_space = splitting.replace(" ", "")

        allowed_characters = "0123456789"
        step_dispatch_table = self._step_dispatch_table
        for key in step_dispatch_table:
            allowed_characters += key

        # sanity check to make sure only allowed combinations are present in string:
        for step in splitting.split():
            if step[0]=="V":
                if len(step) > 1:
                    try:
                        force_group_number = int(step[1:])
                        if force_group_number > 31:
                            raise ValueError("OpenMM only allows up to 32 force groups")
                    except ValueError:
                        raise ValueError("You must use an integer force group")
            elif step == "{":
                    if "}" not in splitting:
                        raise ValueError("Use of { must be followed by }")
                    if not self._verify_metropolization(splitting):
                        raise ValueError("Shadow work generating steps found outside the Metropolization block")
            elif step in allowed_characters:
                continue
            else:
                raise ValueError("Invalid step name '%s' used; valid step names are %s" % (step, allowed_characters))

        # Make sure we contain at least one of R, V, O steps
        assert ("R" in splitting_no_space)
        assert ("V" in splitting_no_space)
        assert ("O" in splitting_no_space)

    def _verify_metropolization(self, splitting):
        """Verify that the shadow-work generating steps are all inside the metropolis block
        Returns False if they are not.
        Parameters
        ----------
        splitting : str
            The langevin splitting string
        Returns
        -------
        valid_metropolis : bool
            Whether all shadow-work generating steps are in the {} block
        """
        # check that there is exactly one metropolized region
        #this pattern matches the { literally, then any number of any character other than }, followed by another {
        #If there's a match, then we have an attempt at a nested metropolization, which is unsupported
        regex_nested_metropolis = "{[^}]*{"
        pattern = re.compile(regex_nested_metropolis)
        if pattern.match(splitting.replace(" ", "")):
            raise ValueError("There can only be one Metropolized region.")

        # find the metropolization steps:
        M_start_index = splitting.find("{")
        M_end_index = splitting.find("}")

        # accept/reject happens before the beginning of metropolis step
        if M_start_index > M_end_index:
            return False

        #pattern to find whether any shadow work producing steps lie outside the metropolization region
        RV_outside_metropolis = "[RV](?![^{]*})"
        outside_metropolis_check = re.compile(RV_outside_metropolis)
        if outside_metropolis_check.match(splitting.replace(" ","")):
            return False
        else:
            return True

    def _add_R_step(self):
        """Add an R step (position update) given the velocities.
        """
        if self._measure_shadow_work:
            self.addComputeGlobal("old_pe", "energy")
            self.addComputeSum("old_ke", self._kinetic_energy)

        n_R = self._ORV_counts['R']

        # update positions (and velocities, if there are constraints)
        self.addComputePerDof("x", "x + ((dt / {}) * v)".format(n_R))
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeGlobal("new_pe", "energy")
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke + new_pe) - (old_ke + old_pe)")

    def _add_V_step(self, force_group="0"):
        """Deterministic velocity update, using only forces from force-group fg.
        Parameters
        ----------
        force_group : str, optional, default="0"
           Force group to use for this step
        """
        if self._measure_shadow_work:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        if self._mts:
            self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(self._force_group_nV[force_group], force_group))
        else:
            self.addComputePerDof("v", "v + (dt / {}) * f / m".format(self._force_group_nV["0"]))

        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke - old_ke)")

    def _add_O_step(self):
        """Add an O step (stochastic velocity update)
        """
        if self._measure_proposal_work:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        self.addComputePerDof("v", "(a * v) + (b * sigma * gaussian)")
        self.addConstrainVelocities()

        if self._measure_proposal_work:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("proposal_work", "proposal_work + -1*(new_ke - old_ke)")

    def _substep_function(self, step_string):
        """Take step string, and add the appropriate R, V, O step with appropriate parameters.
        The step string input here is a single character (or character + number, for MTS)
        """
        function, can_accept_force_groups = self._step_dispatch_table[step_string[0]]
        if can_accept_force_groups:
            force_group = step_string[1:]
            function(force_group)
        else:
            function()

    def _parse_splitting_string(self, splitting_string):
        """Parse the splitting string to check for simple errors and extract necessary information
        Parameters
        ----------
        splitting_string : str
            The string that specifies how to do the integrator splitting
        Returns
        -------
        ORV_counts : dict
            Number of O, R, and V steps
        mts : bool
            Whether the splitting specifies an MTS integrator
        force_group_n_V : dict
            Specifies the number of V steps per force group. {"0": nV} if not MTS
        """
        # convert the string to all caps
        splitting_string = splitting_string.upper()

        # sanity check the splitting string
        self._sanity_check(splitting_string)

        ORV_counts = dict()

        # count number of R, V, O steps:
        for step_symbol in self._step_dispatch_table:
            ORV_counts[step_symbol] = splitting_string.count(step_symbol)

        # split by delimiter (space)
        step_list = splitting_string.split(" ")

        # populate a list with all the force groups in the system
        force_group_list = []
        for step in step_list:
            # if the length of the step is greater than one, it has a digit after it
            if step[0] == "V" and len(step) > 1:
                force_group_list.append(step[1:])

        # Make a set to count distinct force groups
        force_group_set = set(force_group_list)

        # check if force group list cast to set is longer than one
        # If it is, then multiple force groups are specified
        if len(force_group_set) > 1:
            mts = True
        else:
            mts = False

        # If the integrator is MTS, count how many times the V steps appear for each
        if mts:
            force_group_n_V = {force_group: 0 for force_group in force_group_set}
            for step in step_list:
                if step[0] == "V":
                    # ensure that there are no V-all steps if it's MTS
                    assert len(step) > 1
                    # extract the index of the force group from the step
                    force_group_idx = step[1:]
                    # increment the number of V calls for that force group
                    force_group_n_V[force_group_idx] += 1
        else:
            force_group_n_V = {"0": ORV_counts["V"]}

        return ORV_counts, mts, force_group_n_V

    def _add_metropolize_start(self):
        """Save the current x and v for a metropolization step later"""
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

    def _add_metropolize_finish(self):
        """Add a Metropolization (based on shadow work) step to the integrator.
        When Metropolization occurs, shadow work is reset.
        """
        self.addComputeGlobal("accept", "step(exp(-(shadow_work)/kT) - uniform)")
        self.addComputeGlobal("ntrials", "ntrials + 1")
        self.beginIfBlock("accept != 1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.addComputeGlobal("nreject", "nreject + 1")
        self.endBlock()
        self.addComputeGlobal("naccept", "ntrials - nreject")
        self.addComputeGlobal("shadow_work", "0")

class OpenMMIAAISLangevinIntegrator(OpenMMLangevinIntegrator):
    """
    Internal Action Annealed Importance Sampling (IA) Langevin Integrator will internally handle the pdf update and accumulate work in the AIS regime.

    """
    def __init__(self,
                 openmm_pdf_update_functions,
                 n_steps,
                 temperature=300.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 splitting="P I H N S V R O R V",
                 constraint_tolerance=1e-8):
        """
        AIS Langevin integrator

        arguments
            openmm_pdf_update_functions : dict
                dict that will update the openmm_pdf_state parameters according to the openmm_pdf_update_functions,
                which is a dict of (key: context_parameter, value: lepton-readable string);
                NOTE : the update scheme is parametrized by an argument called 'fractional_iteration'
            n_steps : int
                the number of steps allocated to the neq protocol
            splitting : string, default: "H V R O R V"
                Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.
                Forces are only used in V-step. Handle multiple force groups by appending the force group index
                to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            temperature : np.unit.Quantity compatible with kelvin, default: 300.0*unit.kelvin
               Fictitious "bath" temperature
            collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
               Collision rate
            timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
               Integration timestep
            constraint_tolerance : float, default: 1.0e-8
                Tolerance for constraint solver
            measure_shadow_work : boolean, default: False
                Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`
            measure_proposal_work : boolean, default: False
                accumulate the proposal work (-heat)
        """


        if openmm_pdf_update_functions is None:
            raise Exception(f"the {self.__class__.__name__} class requires update functions")
        if any('fractional_iteration' not in val for val in openmm_pdf_update_functions.values()):
            _logger.warning(f"'fractional_iteration' is not found in all of the openmm_pdf_update_functions")

        if (n_steps < 0) or (n_steps != int(n_steps)):
            raise Exception('n_steps must be an integer >= 0')

        self._openmm_pdf_update_functions = openmm_pdf_update_functions
        self._n_steps = n_steps # number of integrator steps

        # collect the system parameters.
        self._system_parameters = {system_parameter for system_parameter in self._openmm_pdf_update_functions.keys()}


        super().__init__(temperature=temperature,
                         collision_rate=collision_rate,
                         timestep=timestep,
                         splitting=splitting,
                         constraint_tolerance=constraint_tolerance,
                         measure_proposal_work = False,
                         measure_shadow_work = False)

    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = super()._step_dispatch_table
        dispatch_table['H'] = (self._add_parameter_perturbation_step, False) #update the parameters according to the update functions
        dispatch_table['P'] = (self._add_old_energy_calculation_step, False) #previous energy computation
        dispatch_table['N'] = (self._add_new_energy_calculation_step, False) #new energy computation
        dispatch_table['S'] = (self._add_state_work_calculation_step, False) # compute the updated state work as state_work + (Unew - Uold)
        dispatch_table['I'] = (self._add_iteration_update_step, False) # compute the updated state iteration
        return dispatch_table


    def _add_global_variables(self):
        """Add the appropriate global parameters to the CustomIntegrator. nsteps refers to the number of
        total steps in the protocol.
        Parameters
        ----------
        nsteps : int, greater than 0
            The number of steps in the switching protocol.
        """
        super()._add_global_variables()
        self.addGlobalVariable('Uold', 0) #old energy value before perturbation
        self.addGlobalVariable('Unew', 0) #new energy value after perturbation
        self.addGlobalVariable('niterations', float(self._n_steps)) # total number of NCMC steps to perform; this SHOULD NOT BE CHANGED during the protocol
        self.addGlobalVariable('iteration', 0.0) # step counter for handling initialization and terminating integration
        self.addGlobalVariable('fractional_iteration', 0.0)

        # Keep track of number of Hamiltonian updates per nonequilibrium switch
        n_H = self._ORV_counts['H'] # number of H updates per integrator step
        if n_H != 1:
            raise Exception(f"exactly ONE parameter perturbation step ('H') is allowed in an integration step")

    def _add_update_alchemical_parameters_step(self):
        """
        Add step to update Context parameters according to provided functions.
        """
        for context_parameter in self._openmm_pdf_update_functions:
            if context_parameter in self._system_parameters:
                self.addComputeGlobal(context_parameter, self._openmm_pdf_update_functions[context_parameter])

    def _add_parameter_perturbation_step(self):
        """
        Add alchemical perturbation step, accumulating protocol work.
        TODO: Extend this to be able to handle force groups?
        """
        # Update all slaved alchemical parameters
        self._add_update_alchemical_parameters_step()

    def _add_old_energy_calculation_step(self):
        """
        add step to compute 'Uold'
        """
        self.addComputeGlobal('Uold', 'energy')

    def _add_new_energy_calculation_step(self):
        """
        add step to compute 'Unew'
        """
        self.addComputeGlobal('Unew', 'energy')

    def _add_state_work_calculation_step(self):
        """
        add step to compute 'state_work'
        """
        self.addComputeGlobal('state_work', 'state_work + (Unew - Uold)')

    def _add_iteration_update_step(step):
        """
        add a step to update the iteration
        """
        # Update lambda and increment that tracks updates.
        self.addComputeGlobal('iteration', 'iteration + 1.0')

        # Update the fractional iteration
        self.addComputeGlobal('fractional_iteration', 'iteration / niterations')


    def reset(self):
        super().reset()
        self.setGlobalVariableByName('proposal_work', 0.0)
        self.setGlobalVariableByName('iteration', 0.0)
        self.setGlobalVariableByName('fractional_iteration', 0.0)
        self.setGlobalVariableByName('state_work', 0.0)

    def get_proposal_work(self, dimensionless=True):
        """Get the current accumulated proposal_work.

        arguments
            dimensionless : bool, optional, default True
               If specified, the work is returned in reduced (kT) unit.

        returns
            work : unit.Quantity or float
               If dimensionless=True, the proposal work in kT (float).
               Otherwise, the unit-bearing proposal in units of energy.
        """
        return self._get_energy_with_units("proposal_work", dimensionless=dimensionless)

    def get_state_work(self, dimensionless = True):
        """Get the current accumulated state work.

        arguments
            dimensionless : bool, optional, default True
               If specified, the work is returned in reduced (kT) unit.

        returns
            work : unit.Quantity or float
               If dimensionless=True, the proposal work in kT (float).
               Otherwise, the unit-bearing proposal in units of energy.
        """
        return self._get_energy_with_units("state_work", dimensionless=dimensionless)

    def set_state_work(self, work):
        """
        set the current accumulated state work

        arguments
            work : float (in units kT)
        """
        self.setGlobalVariableByName('state_work', (work * self.kT).value_in_unit(_OPENMM_ENERGY_UNIT))

    def set_proposal_work(self, work):
        """
        set the current accumulated proposal work

        arguments
            work : float (in units kT)
        """
        self.setGlobalVariableByName('proposal_work', (work * self.kT).value_in_unit(_OPENMM_ENERGY_UNIT))


class OpenMMIAEulerMaruyamaIntegrator(OpenMMIAAISLangevinIntegrator):
    """
    Internal Action Euler Maruyama Integrator.
    Adds 'E' to the dispatch table, which executes an Euler-Maruyama move.
    Interestingly, the Euler-Maruyama integration scheme is the first scheme I'm aware of that avoids any and all complications
    associated with a running velocity in the forward/reverse kernel.

    The update scheme works as follows:
        x_(k+1) = x_k + (tau) * force(x_k | parameters) * beta + sqrt(2*(tau)) * R(t)
        where:
            force(x_k | parameters) = -grad(openmm_pdf_state),
            tau = D * dt; [tau] = positions_unit ** 2
            D = 1 / (openmm_pdf_state.beta * 2 * mass_vector); [D] = velocity_unit ** 2
    """
    def __init__(openmm_pdf_update_functions,
                 n_steps,
                 temperature=300.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 splitting="P I H N S V R O R V",
                 constraint_tolerance=1e-8):

        super().__init__(openmm_pdf_update_functions,
                         n_steps,
                         temperature=300.0 * unit.kelvin,
                         collision_rate=1.0 / unit.picoseconds,
                         timestep=1.0 * unit.femtoseconds,
                         splitting="P I H N S V R O R V",
                         constraint_tolerance=1e-8)

        self.addPerDofVariable('x_EM_old', 0.)
        self.addPerDofVariable('x_EM_new', 0.)
        self.addPerDofVariable('f_EM_old', 0.)
        self.addPerDofVariable('f_EM_new', 0.)
        self.addPerDofVariable('gaussian_fract', 0)



    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = super()._step_dispatch_table
        dispatch_table['E'] = (self._add_euler_maruyama_update, False)
        return dispatch_table

    def _add_integrator_steps(self):
        """
        Add the steps to the integrator--this can be overridden to place steps around the integration.
        We update this in the Euler-Maruyama regime because we need to add the diffusion constant D and the update constant tau.
        """
        # Integrate
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)",
                                                      "D" : "kT/(2.0 * m)",
                                                      "tau": "D * dt"})

        for i, step in enumerate(self._splitting.split()):
            self._substep_function(step)

    def _add_euler_maruyama_update(self):
        """
                dx_grad = tau * force * self.openmm_pdf_state.beta #this is the incremental x update from the gradient of the distribution

                #now we add the stochastic part
                dx_stochastic = np.random.randn(self.mass_vector.shape[1], 3) * np.sqrt(2 * tau.value_in_unit(self.x_unit**2)) * unit.x_unit

                #combine them...
                dx = dx_grad + dx_stochastic

                proposal_work_numerator = -np.sum((new_positions - old_positions - tau * old_force * self.openmm_pdf_state.beta)**2) / (4.0 * tau)
                proposal_work_denominator = -np.sum((old_positions - new_positions - tau * new_force * self.openmm_pdf_state.beta)**2) / (4.0 * tau)
        """
        # update positions (and velocities, if there are constraints)
        n_R = self._ORV_counts['R']
        self.addComputePerDof("x", "x + (tau * f / (kT * {}) + (gaussian * sqrt(2.0 * tau))".format(n_R)) #full Euler-Maruyama update
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        self.addConstrainVelocities()

    def _add_global_variables(self):
        """Add the appropriate global parameters to the CustomIntegrator. nsteps refers to the number of
        total steps in the protocol.
        Parameters
        ----------
        nsteps : int, greater than 0
            The number of steps in the switching protocol.
        """
        super()._add_global_variables()
        self.addGlobalVariable('Uold', 0) #




class OpenMMEulerMaruyamaIntegrator(Propagator):
    """
    Generalized Euler Maruyama Integrator.

    The Euler Maruyama Integrator for Langevin diffusion uses the following updated procedure

    NOTES :
        1. Euler Maruyama Integrator does not support SHAKE, SETTLE, or any other constraints.
        2. integrator is not autodiff-able since the gradients of density (Forces) are computed with OpenMM
    """
    def __init__(self, mass_vector, timestep, **kwargs):
        """
        Initialize the integrator

        arguments
            mass_vector : np.ndarray * unit.daltons (or mass units)
                mass vector of size (1,N) where N is the number of atoms
            timestep : float * unit.femtosecond (or time units), default 1.0 * unit.femtosecond
                timestep of dynamics

        attributes
            mass_vector : np.ndarray * unit.daltons (or mass units)
                mass vector of size (1,N) where N is the number of atoms
            timestep : float * unit.femtosecond (or time units), default 1.0 * unit.femtosecond
                timestep of dynamics

        """
        assert mass_vector.shape[0] == 1, f"the mass vector is not a column vector"
        self.mass_vector = mass_vector
        self.timestep = timestep

        self.x_unit = unit.nanometers
        self.t_unit = unit.femtoseconds
        self.m_unit = unit.daltons
        self.u_unit = self.m_unit * self.x_unit **2 / (self.t_unit **2)
        self.f_unit = self.u_unit / self.x_unit

    def apply(self,
              openmm_pdf_state,
              particle_state,
              **kwargs):
        """
        Apply the OpenMMEulerMaruyamaIntegrator once.
        This modifies the given particle_state. The temperature of the
        thermodynamic state is used in Langevin dynamics.

        The update scheme works as follows:
            x_(k+1) = x_k + (tau) * force(x_k | parameters) * beta + sqrt(2*(tau)) * R(t)
            where:
                force(x_k | parameters) = -grad(openmm_pdf_state),
                tau = D * dt;
                D = 1 / (openmm_pdf_state.beta * 2 * mass_vector)

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
        D = 1. / ( openmm_pdf_state.beta * 2 * self.mass_vector[0,:][:,None] )
        tau = D * self.timestep**2
        force = self.get_forces(self.particle_state.state())
        dx_grad = tau * force * self.openmm_pdf_state.beta #this is the incremental x update from the gradient of the distribution

        #now we add the stochastic part
        dx_stochastic = np.random.randn(self.mass_vector.shape[1], 3) * np.sqrt(2 * tau.value_in_unit(self.x_unit**2)) * unit.x_unit

        #combine them...
        dx = dx_grad + dx_stochastic
        old_positions = self.particle_state.positions
        new_positions = old_positions + dx

        #compute the proposal_work
        proposal_work = self._compute_proposal_work(tau, old_positions, new_positions)

        return particle_state, proposal_work

    def get_forces(self, positions):
        """
        return a force matrix of shape (N, 3) w.r.t. self.openmm_pdf_state at the positions defined by
        self.particle_state.positions

        returns
            force : np.ndarray * self.f_unit
                the force array (N,3)
        """
        integrator = get_dummy_integrator()
        particle_state = OpenMMParticleState(positions = positions)
        context, integrator = cache.global_context_cache.get_context(self.openmm_pdf_state, integrator)
        particle_state.apply_to_context(context, ignore_velocities=True)
        context_state = context.getState(getForces=True)
        forces = context_state.getForces(asNumpy=True)

        return forces

    def _compute_proposal_work(self, tau, old_positions, new_positions):
        """
        compute the proposal work defined as follows:
        log(q(x_new|x_old, openmm_pdf_state)) - log(q(x_old | x_new, openmm_pdf_state))

        where q(x' | x) \propto np.exp(||x' - x - tau * grad(log(pi(x))) ||**2)

        arguments
            tau : float * unit.x_unit **2,
                position scaling factor
            old_positions : np.ndarray * unit.x_unit
                old positions
            new_positions : np.ndarray * unit.x_unit
                new positions

        returns
            proposal_work : float
                log ratio of the forward to reverse work
        """
        old_force, new_force = self.get_forces(old_positions), self.get_forces(new_positions)
        proposal_work_numerator = -np.sum((new_positions - old_positions - tau * old_force * self.openmm_pdf_state.beta)**2) / (4.0 * tau)
        proposal_work_denominator = -np.sum((old_positions - new_positions - tau * new_force * self.openmm_pdf_state.beta)**2) / (4.0 * tau)
        proposal_work = proposal_work_numerator - proposal_work_denominator
        return proposal_work
