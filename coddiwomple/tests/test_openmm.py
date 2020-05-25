"""
Unit and regression test for the coddiwomple.openmm package.
"""
# Import package, test suite, and other packages as needed
import numpy as np
import sys
import os
from simtk import openmm, unit
from openmmtools.testsystems import HarmonicOscillator
from coddiwomple.tests.utils import get_harmonic_testsystem
from coddiwomple.tests.utils import HarmonicAlchemicalState

# from coddiwomple.openmm.integrators import *
# from coddiwomple.openmm.propagators import *
# from coddiwomple.openmm.reporters import *
# from coddiwomple.openmm.states import *
# from coddiwomple.openmm.utils import *
from pkg_resources import resource_filename
import tempfile
import shutil
import pickle
import tqdm
import mdtraj as md

def test_OpenMMPDFState():
    """
    conduct a class-wide test on coddiwomple.openmm.states.OpenMMPDFState with the `get_harmonic_testsystem` testsystem
    this will assert successes on __init__, set_parameters, get_parameters, reduced_potential methods
    """
    temperature = 300 * unit.kelvin
    pressure = None
    from coddiwomple.openmm.states import OpenMMPDFState, OpenMMParticleState

    #create the default get_harmonic_testsystem
    testsystem, period, collision_rate, timestep, alchemical_functions = get_harmonic_testsystem(temperature = temperature)

    #test init method
    pdf_state = OpenMMPDFState(system = testsystem.system, alchemical_composability = HarmonicAlchemicalState, temperature = temperature, pressure = pressure)
    assert isinstance(pdf_state._internal_context, openmm.Context)
    print(f"pdf_state parameters: {pdf_state._parameters}")

    #test set_parameters
    new_parameters = {key : 1.0 for key, val in pdf_state._parameters.items() if val is not None}
    pdf_state.set_parameters(new_parameters) #this should set the new parameters, but now we have to make sure that the context actually has those parameters bound
    swig_parameters = pdf_state._internal_context.getParameters()
    context_parameters = {q: swig_parameters[q] for q in swig_parameters}
    assert context_parameters['testsystems_HarmonicOscillator_x0'] == 1.
    assert context_parameters['testsystems_HarmonicOscillator_U0'] == 1.

    #test get_parameters
    returnable_parameters = pdf_state.get_parameters()
    assert len(returnable_parameters) == 2
    assert returnable_parameters['testsystems_HarmonicOscillator_x0'] == 1.
    assert returnable_parameters['testsystems_HarmonicOscillator_U0'] == 1.

    #test reduced_potential
    particle_state = OpenMMParticleState(positions = testsystem.positions) #make a particle state so that we can compute a reduced potential
    reduced_potential = pdf_state.reduced_potential(particle_state)
    externally_computed_reduced_potential = pdf_state._internal_context.getState(getEnergy=True).getPotentialEnergy()*pdf_state.beta
    assert np.isclose(reduced_potential, externally_computed_reduced_potential)

def test_OpenMMParticleState():
    """
    conduct a class-wide test on coddiwomple.openmm.states.OpenMMParticleState with the `get_harmonic_testsystem` testsystem
    this will assert successes on __init__, as well as _all_ methods in the coddiwomple.particles.Particle class
    """
    temperature = 300 * unit.kelvin
    pressure = None
    from coddiwomple.openmm.states import OpenMMPDFState, OpenMMParticleState
    from coddiwomple.particles import Particle

    #create the default get_harmonic_testsystem
    testsystem, period, collision_rate, timestep, alchemical_functions = get_harmonic_testsystem(temperature = temperature)

    #test __init__ method
    particle_state = OpenMMParticleState(positions = testsystem.positions) #make a particle state
    particle = Particle(index = 0, record_state=False, iteration = 0)

    #test update_state
    assert particle.state is None
    assert not particle._record_states
    particle.update_state(particle_state)
    assert particle.state is not None

    #test update_iteration
    assert particle.iteration == 0
    particle.update_iteration()
    assert particle.iteration == 1

    #test update ancestry
    assert particle.ancestry == [0]
    particle.update_ancestry(1)
    assert particle.ancestry == [0,1]

    #the rest of the methods are trivial or would be redundant to test

def test_OpenMMReporter():
    """
    test the OpenMMReporter object for its ability to make appropriate trajectory writes for particles.
    use the harmonic oscillator testsystem

    NOTE : this class will conduct dynamics on 5 particles defined by the harmonic oscillator testsystem in accordance with the coddiwomple.openmm.propagators.OMMBIP
    equipped with the coddiwomple.openmm.integrators.OMMLI integrator, but will NOT explicitly conduct a full test on the propagators or integrators.
    """
    import os
    from coddiwomple.openmm.propagators import OMMBIP
    from coddiwomple.openmm.integrators import OMMLI

    temperature = 300 * unit.kelvin
    pressure = None
    from coddiwomple.openmm.states import OpenMMPDFState, OpenMMParticleState
    from coddiwomple.particles import Particle
    from coddiwomple.openmm.reporters import OpenMMReporter
    import shutil

    #create the default get_harmonic_testsystem
    testsystem, period, collision_rate, timestep, alchemical_functions = get_harmonic_testsystem(temperature = temperature)

    #create a particle state and 5 particles
    particles = []
    for i in range(5):
        particle_state = OpenMMParticleState(positions = testsystem.positions) #make a particle state
        particle = Particle(index = i, record_state=False, iteration = 0)
        particle.update_state(particle_state)
        particles.append(particle)

    #since we are copying over the positions, we need a simple assert statement to make sure that the id(hex(particle_state.positions)) are separate in memory
    position_hexes = [hex(id(particle.state.positions)) for particle in particles]
    assert len(position_hexes) == len(list(set(position_hexes))), f"positions are copied identically; this is a problem"

    #create a pdf_state
    pdf_state = OpenMMPDFState(system = testsystem.system, alchemical_composability = HarmonicAlchemicalState, temperature = temperature, pressure = pressure)

    #create an integrator
    integrator = OMMLI(temperature=temperature, collision_rate=collision_rate, timestep=timestep)

    #create a propagator
    propagator = OMMBIP(openmm_pdf_state = pdf_state, integrator = integrator)

    steps_per_application = 100

    #the only thing we want to do here is to run independent md for each of the particles and save trajectories; at the end, we will delete the directory and the traj files
    temp_traj_dir, temp_traj_prefix = os.path.join(os.getcwd(), 'test_dir'), 'traj_prefix'
    reporter = OpenMMReporter(trajectory_directory = 'test_dir', trajectory_prefix='traj_prefix', md_topology=testsystem.mdtraj_topology)
    assert reporter.write_traj

    num_applications=10
    for application_index in range(num_applications):
        returnables = [propagator.apply(particle.state, n_steps=100, reset_integrator=True, apply_pdf_to_context=True, randomize_velocities=True) for particle in particles]
        _save=True if application_index == num_applications-1 else False
        reporter.record(particles, save_to_disk=_save)
    assert reporter.hex_counter == len(reporter.hex_dict)
    assert os.path.exists(temp_traj_dir)
    assert os.listdir(temp_traj_dir) is not None

    #then we can delete
    shutil.rmtree(temp_traj_dir)




def test_ais_propagator():
    """
    Test the OpenMM Annealed Importance Sampling Propagator and its subclasses
    """
    from perses.annihilation.lambda_protocol import RelativeAlchemicalState
    traj, factory, alchemical_functions = load_prereqs()
    num_frames = traj.n_frames
    pdf_state = OpenMMPDFState(system = factory.hybrid_system, alchemical_composability = RelativeAlchemicalState, pressure=None)
    n_steps = 10

    ais_integrator = OMMLIAIS(
                             alchemical_functions,
                             n_steps,
                             temperature=300.0 * unit.kelvin,
                             collision_rate=1.0 / unit.picoseconds,
                             timestep=1.0 * unit.femtoseconds,
                             splitting="P I H N S V R O R V",
                             constraint_tolerance=1e-6)

    ais_propagator = OMMAISP(openmm_pdf_state = pdf_state, integrator = ais_integrator, record_state_work_interval = 1, reassign_velocities=True)
    frames = np.random.choice(range(num_frames), 10)

    for num_trajectories in tqdm.trange(len(frames)):
        positions = traj.xyz[frames[num_trajectories]] * unit.nanometers
        particle_state = OpenMMParticleState(positions=positions, box_vectors=traj.unitcell_vectors[frames[num_trajectories]]*unit.nanometers)
        _, return_dict = ais_propagator.apply(particle_state, n_steps=n_steps, apply_pdf_to_context = True, reset_integrator = True)
        print(return_dict)

def test_smc_mcmc_resampler():
    """
    Test coddiwomple.openmm.coddiwomple.mcmc_smc_resampler
    """
    from coddiwomple.openmm.coddiwomple import mcmc_smc_resampler
    traj, factory, alchemical_functions = load_prereqs()
    default_functions = {'lambda_sterics_core':
                     lambda x: x,
                     'lambda_electrostatics_core':
                     lambda x: x,
                     'lambda_sterics_insert':
                     lambda x: 2.0 * x if x < 0.5 else 1.0,
                     'lambda_sterics_delete':
                     lambda x: 0.0 if x < 0.5 else 2.0 * (x - 0.5),
                     'lambda_electrostatics_insert':
                     lambda x: 0.0 if x < 0.5 else 2.0 * (x - 0.5),
                     'lambda_electrostatics_delete':
                     lambda x: 2.0 * x if x < 0.5 else 1.0,
                     'lambda_bonds':
                     lambda x: x,
                     'lambda_angles':
                     lambda x: x,
                     'lambda_torsions':
                     lambda x: x
                     }

    lambda_sequence = [{key: default_functions[key](q) for key in default_functions.keys()} for q in np.linspace(0,1,10)]
    _traj_filename = resource_filename('coddiwomple', f"/data/perses_data/fluorobenzene_zero_endstate_cache/vacuum.cache.pdb")

    particles = mcmc_smc_resampler(system = factory.hybrid_system,
                           endstate_cache_filename = _traj_filename,
                           directory_name = None,
                           trajectory_prefix = None,
                           md_topology = factory.hybrid_topology,
                           number_of_particles = 10,
                           parameter_sequence = lambda_sequence)

    assert all(particle.iteration == 9 for particle in particles)
    reduced_free_energy = 1.33
    tolerance = 0.2
    average_free_energy = np.mean([particle.cumulative_work for particle in particles])
    upper_threshold, lower_threshold = (reduced_free_energy + tolerance)/reduced_free_energy, (reduced_free_energy - tolerance)/reduced_free_energy
    assert average_free_energy / reduced_free_energy < upper_threshold and average_free_energy / reduced_free_energy > lower_threshold
    return particles
