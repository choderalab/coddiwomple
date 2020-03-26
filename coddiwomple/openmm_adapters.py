"""
OpenMM Adapter Module:
This module is a consolidation of standard SMC targets, proposals, etc. specifically for openMM interoperability.
"""

#####Imports#####
from coddiwomple.states import ParticleState, PDFState
from coddiwomple.distribution_factories import TargetFactory, ProposalFactory
from coddiwomple.propagators import Propagator
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState, IComposableState
from openmmtools import cache, utils
from openmmtools import integrators
from perses.dispersed.utils import check_platform, configure_platform #TODO: make sure this is functional and supports mixed precision
from simtk import unit
import simtk.openmm as openmm
from coddiwomple.openmm_utils import get_dummy_integrator
from openmmtools.mcmc import BaseIntegratorMove
from coddiwomple.reporters import Reporter
import mdtraj.utils as mdtrajutils
import logging

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("openmm_adapters")
_logger.setLevel(logging.DEBUG)

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
        super().__init__(positions = positions, velocities = velocities, box_vectors = box_vectors)

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
        from openmmtools.alchemy import AlchemicalState
        openmm_pdf_state = ThermodynamicState(system, temperature, pressure)
        alchemical_state = AlchemicalState.from_system(system, **kwargs)
        assert isinstance(alchemical_state, IComposableState), f"alchemical state is not an instance of IComposableState"
        self.__dict__ = openmm_pdf_state.__dict__
        self._composable_states = [alchemical_state]
        self.set_system(self._standard_system, fix_state=True)
        _logger.debug(f"successfully instantiated OpenMMPDFState equipped with the following parameters: {self._parameters}")

    def set_parameters(self, parameters):
        """
        update the pdf_state parameters in place

        arguments
            parameters : dict
                dictionary of variables
        """
        assert set([param[0] for param in self._parameters.items() if param[1] is not None]) == set(parameters.keys()), f"the parameter keys supplied do not match the internal parameter names"
        for key,val in parameters.items():
            assert hasattr(self, key), f"{self} does not have a parameter named {key}"
            setattr(self, key, val)
        _logger.debug(f"successfully updated OpenMMPDFState parameters as follows: {parameters}")

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
        super().__init__(pdf_state = pdf_state, parameter_sequence = parameter_sequence, termination_parameters = termination_parameters, **kwargs)


#ProposalFactory Adapter
class OpenMMProposalFactory(ProposalFactory):
    """
    Adapter for ProposalFactory
    """
    def __init__(self, openmm_pdf_state, parameter_sequence, propagator, **kwargs):
        """
        Initialize the OpenMMTargetFactory

        arguments
            pdf_state : coddiwomple.states.OpenMMPDFState
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
        super().__init__(pdf_state = openmm_pdf_state, parameter_sequence = parameter_sequence, propagator = propagator, **kwargs)

    def _generate_initial_sample(self, generation_pdf, propagator, particle_state):
        """

        """

#Propagator Adapter
class OpenMMBaseIntegratorMove(Propagator, BaseIntegratorMove):
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
        super().__init__(n_steps = n_steps,
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
    """Langevin dynamics segment as a (pseudo) Monte Carlo move.
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

    attributes
        timestep : simtk.unit.Quantity
            The timestep to use for Langevin integration (time units).
        collision_rate : simtk.unit.Quantity
            The collision rate with fictitious bath particles (1/time units).
        n_steps : int
            The number of integration timesteps to take each time the move
            is applied.
        reassign_velocities : bool
            If True, the velocities will be reassigned from the Maxwell-Boltzmann
            distribution at the beginning of the move.
        context_cache : openmmtools.cache.ContextCache
            The ContextCache to use for Context creation. If None, the global
            cache openmmtools.cache.global_context_cache is used.
    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=10.0/unit.picoseconds,
                 n_steps=1000, reassign_velocities=False, **kwargs):
        super().__init__(n_steps=n_steps,
                         reassign_velocities=reassign_velocities,
                         **kwargs)
        self.timestep = timestep
        self.collision_rate = collision_rate

class OpenMMLangevinDynamicsSplittingMove(OpenMMLangevinDynamicsMove):
    """
    Langevin dynamics segment with custom splitting of the operators and optional Metropolized Monte Carlo validation.
    Besides all the normal properties of the :class:`LangevinDynamicsMove`, this class implements the custom splitting
    sequence of the :class:`openmmtools.integrators.LangevinIntegrator`. Additionally, the steps can be wrapped around
    a proper Generalized Hybrid Monte Carlo step to ensure that the exact distribution is generated.

    NOTE: heat is always measured, but 'shadow work' is not

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
        splitting : string, default: "V R O R V"
            Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.
            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            "{" will cause metropolization, and must be followed later by a "}".
        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

    attributes
        timestep : simtk.unit.Quantity
            The timestep to use for Langevin integration (time units).
        collision_rate : simtk.unit.Quantity
            The collision rate with fictitious bath particles (1/time units).
        n_steps : int
            The number of integration timesteps to take each time the move
            is applied.
        reassign_velocities : bool
            If True, the velocities will be reassigned from the Maxwell-Boltzmann
            distribution at the beginning of the move.
        context_cache : openmmtools.cache.ContextCache
            The ContextCache to use for Context creation. If None, the global
            cache openmmtools.cache.global_context_cache is used.
        splitting : str
            Splitting applied to this integrator represented as a string.
        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver
    """

    def __init__(self,
                 timestep=1.0 * unit.femtosecond,
                 collision_rate=1.0 / unit.picoseconds,
                 n_steps=1000,
                 reassign_velocities=False,
                 splitting="V R O R V",
                 constraint_tolerance=1.0e-8,
                 **kwargs):
        super().__init__(n_steps=n_steps,
                         reassign_velocities=reassign_velocities,
                         timestep=timestep,
                         collision_rate=collision_rate,
                         **kwargs)
        self.splitting = splitting
        self.constraint_tolerance = constraint_tolerance
        _logger.debug(f"successfully initialized an OpenMMLangevinDynamicsSplittingMove")

    def __getstate__(self):
        serialization = super().__getstate__()
        serialization['splitting'] = self.splitting
        serialization['constraint_tolerance'] = self.constraint_tolerance
        return serialization

    def __setstate__(self, serialization):
        super().__setstate__(serialization)
        self.splitting = serialization['splitting']
        self.constraint_tolerance = serialization['constraint_tolerance']

    def _get_integrator(self, openmm_pdf_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        return integrators.LangevinIntegrator(temperature=openmm_pdf_state.temperature,
                                              collision_rate=self.collision_rate,
                                              timestep=self.timestep,
                                              splitting=self.splitting,
                                              constraint_tolerance=self.constraint_tolerance,
                                              measure_shadow_work=False,
                                              measure_heat=True)



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

class OpenMMReporter(Reporter):
    """
    OpenMM specific reporter
    """
    def __init__(self,
                 trajectory_directory,
                 trajectory_prefix,
                 topology,
                 subset_indices = None,
                 **kwargs):
        """
        Initialize the OpenMM particles.

        arguments
            trajectory_directory : str
                name of directory
            trajectory_prefix : str
                prefix of the files to write
            topology : simtk.openmm.app.topology.Topology
                (complete) topology object to which to write
            subset_indices : list(int)
                zero-indexed atom indices to write
        """
        # equip attributes
        self.trajectory_directory, self.trajectory_prefix = trajectory_directory, trajectory_prefix
        self.topology = topology
        self.hex_dict = {}

        #prepare topology object
        if self.trajectory_directory is not None and self.trajectory_prefix is not None:
            _logger.debug(f"creating trajectory storage object...")
            self.write_traj = True
            self.neq_traj_filename = os.path.join(os.getcwd(), self.trajectory_directory,
                                                  f"{self.trajectory_prefix}.neq")
            os.mkdir(os.path.join(os.getcwd(), self.trajectory_directory))
            md_topology = md.Topology().from_openmm(self.topology)
            if self.subset_indices is None:
                self.md_topology = md_topology
                self.subset_indices = len([atom for atom in self.md_topology.atoms()])
            else:
                self.md_topology = complex_md_topology.md_topology.subset(self.subset_indices)
                self.subset_indices = subset_indices
        else:
            self.write_traj = False

    def record(self, particles, save_to_disk = False, **kwargs):
        """
        append the positions, box lengths, and box angles to their respective attributes and save to disk if specified;
        save to disk if specified

        arguments
            particles : list(coddiwomple.particles.Particle)
                list of particle objects
            save_to_disk : bool, default False
                whether to save the trajectory to disk
        """
        if self.write_traj:
            for particle in particles:
                particle_hex = hex(id(particle))
                if particle_hex in self.hex_dict.keys():
                    pass
                else:
                    self.hex_dict[particle_hex] = [tuple(), tuple(), tuple()]

                self.hex_dict[particle_hex][0].append(particle.state().positions[self.subset_indices, :].value_in_unit_system(unit.md_unit_system))
                a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*complex_sampler_state.box_vectors)
                self.hex_dict[particle_hex][1].append([a, b, c])
                self.hex_dict[particle_hex][2].append([alpha, beta, gamma])

                if save_to_disk:
                    filename = f"{self.neq_traj_filename}.{particle_hex}.pdb"
                    self._write_trajectory(self, hex = particle_hex, filename = filename)

    def _write_trajectory(self, hex, filename):
        """
        write a trajectory given a filename.

        arguments
            filename : str
                name of the file to write
            hex : str
                hex memory address
        """

        traj = md.Trajectory(np.array(self.hex_dict[hex][0]),
                             unitcell_lengths = np.array(self.hex_dict[hex][1]),
                             unitcell_angles = np.array(self.hex_dict[hex][2])
                             )
        traj.center_coordinates()
        traj.image_molecules(inplace=True)
        traj.save(filename)
