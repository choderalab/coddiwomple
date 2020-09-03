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
# from pkg_resources import resource_filename
# import tempfile
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

def test_OMMLI():
    """
    test OMMLI (OpenMMLangevinIntegrator) in the baoab regime on the harmonic test system;
    Specifically, we run MD to convergence and assert that the potential energy of the system and the standard
    deviation thereof is within a specified threshold.
    We also check the accumulation of shadow, proposal works, as well as the ability to reset, initialize, and subsume the integrator into an OMMBIP propagator
    """
    from coddiwomple.openmm.propagators import OMMBIP
    from coddiwomple.openmm.integrators import OMMLI
    import tqdm

    temperature = 300 * unit.kelvin
    pressure = None
    from coddiwomple.openmm.states import OpenMMPDFState, OpenMMParticleState
    from coddiwomple.particles import Particle

    #create the default get_harmonic_testsystem
    testsystem, period, collision_rate, timestep, alchemical_functions = get_harmonic_testsystem(temperature = temperature)

    particle_state = OpenMMParticleState(positions = testsystem.positions) #make a particle state
    particle = Particle(index = 0, record_state=False, iteration = 0)
    particle.update_state(particle_state)

    num_applications = 100

    #create a pdf_state
    pdf_state = OpenMMPDFState(system = testsystem.system, alchemical_composability = HarmonicAlchemicalState, temperature = temperature, pressure = pressure)

    #create an integrator
    integrator = OMMLI(temperature=temperature, collision_rate=collision_rate, timestep=timestep)

    #create a propagator
    propagator = OMMBIP(openmm_pdf_state = pdf_state, integrator = integrator)

    #expected reduced potential
    mean_reduced_potential = testsystem.get_potential_expectation(pdf_state) * pdf_state.beta
    std_dev_reduced_potential = testsystem.get_potential_standard_deviation(pdf_state) * pdf_state.beta

    reduced_pe = []

    #some sanity checks for propagator:
    global_integrator_variables_before_integration = propagator._get_global_integrator_variables()
    print(f"starting integrator variables: {global_integrator_variables_before_integration}")

    #some sanity checks for integrator:
    start_proposal_work = propagator.integrator._get_energy_with_units('proposal_work', dimensionless=True)
    start_shadow_work = propagator.integrator._get_energy_with_units('shadow_work', dimensionless=True)
    assert start_proposal_work == global_integrator_variables_before_integration['proposal_work']
    assert start_shadow_work == global_integrator_variables_before_integration['shadow_work']

    for app_num in tqdm.trange(num_applications):
        particle_state, proposal_work = propagator.apply(particle_state, n_steps=20, reset_integrator=False, apply_pdf_to_context=False, randomize_velocities=True)
        assert proposal_work==0. #this must be the case since we did not pass a 'returnable_key'

        #sanity checks for inter-application methods
        assert propagator.integrator._get_energy_with_units('proposal_work', dimensionless=True) != 0. #this cannot be zero after a step of MD without resets
        assert propagator.integrator._get_energy_with_units('shadow_work', dimensionless=True) != 0. #this cannot be zero after a step of MD without resets
        reduced_pe.append(pdf_state.reduced_potential(particle_state))


    tol=6 * std_dev_reduced_potential
    calc_mean_reduced_pe = np.mean(reduced_pe)
    calc_stddev_reduced_pe = np.std(reduced_pe)
    assert calc_mean_reduced_pe < mean_reduced_potential + tol and calc_mean_reduced_pe > mean_reduced_potential - tol, f"the mean reduced energy and standard deviation ({calc_mean_reduced_pe}, {calc_stddev_reduced_pe}) is outside the tolerance \
        of a theoretical mean potential energy of {mean_reduced_potential} +/- {tol}"
    print(f"the mean reduced energy/standard deviation is {calc_mean_reduced_pe, calc_stddev_reduced_pe} and the theoretical mean reduced energy and stddev are {mean_reduced_potential}")

    #some cleanup of the integrator
    propagator.integrator.reset() #this should reset proposal, shadow, and ghmc staticstics (we omit ghmc stats)
    assert propagator.integrator._get_energy_with_units('proposal_work', dimensionless=True) == 0. #this should be zero after a reset
    assert propagator.integrator._get_energy_with_units('shadow_work', dimensionless=True) == 0. #this should be zero after a reset


def test_OMMBIP():
    """
    test OMMBIP (OpenMMBaseIntegratorPropagator) in the baoab regime on the harmonic test system;
    specifically, we validate the init, apply, _get_global_integrator_variables, _get_context_parameters methods.
    For the sake of testing all of the internal methods, we equip an OMMLI integrator
    """
    from coddiwomple.openmm.propagators import OMMBIP
    from coddiwomple.openmm.integrators import OMMLI

    temperature = 300 * unit.kelvin
    pressure = None
    from coddiwomple.openmm.states import OpenMMPDFState, OpenMMParticleState

    #create the default get_harmonic_testsystem
    testsystem, period, collision_rate, timestep, alchemical_functions = get_harmonic_testsystem(temperature = temperature)

    particle_state = OpenMMParticleState(positions = testsystem.positions) #make a particle state

    num_applications = 100

    #create a pdf_state
    pdf_state = OpenMMPDFState(system = testsystem.system, alchemical_composability = HarmonicAlchemicalState, temperature = temperature, pressure = pressure)

    #create an integrator
    integrator = OMMLI(temperature=temperature, collision_rate=collision_rate, timestep=timestep)

    #create a propagator
    propagator = OMMBIP(openmm_pdf_state = pdf_state, integrator = integrator)

    #check the __init__ method for appropriate equipment
    assert hex(id(propagator.pdf_state)) == hex(id(pdf_state)) #the defined pdf state is tethered to the propagator (this is VERY important for SMC)

    #conduct null application
    prior_reduced_potential = pdf_state.reduced_potential(particle_state)
    return_state, proposal_work = propagator.apply(particle_state, n_steps=0)
    assert proposal_work == 0. #there is no proposal work if returnable_key is None
    assert pdf_state.reduced_potential(particle_state) == prior_reduced_potential
    propagator_state = propagator.context.getState(getEnergy=True)
    assert np.isclose(propagator_state.getPotentialEnergy()*pdf_state.beta,  pdf_state.reduced_potential(particle_state))

    #check context update internals
    prior_reduced_potential = pdf_state.reduced_potential(particle_state)
    parameters = pdf_state.get_parameters() #change an alchemical parameter
    parameters['testsystems_HarmonicOscillator_U0'] = 1. #update parameter dict
    pdf_state.set_parameters(parameters) #set new params

    _ = propagator.apply(particle_state, n_steps=0, apply_pdf_to_context=False) # if we do not apply to context, then the internal_context should not be modified
    assert propagator._get_context_parameters()['testsystems_HarmonicOscillator_U0'] == 0.
    assert np.isclose(propagator.context.getState(getEnergy=True).getPotentialEnergy()*pdf_state.beta, prior_reduced_potential)
    _ = propagator.apply(particle_state, n_steps=0, apply_pdf_to_context=True) # if we do apply to context, then the internal_context should be modified
    assert propagator._get_context_parameters()['testsystems_HarmonicOscillator_U0'] == 1.
    assert np.isclose(prior_reduced_potential + 1.0 * unit.kilojoules_per_mole * pdf_state.beta, propagator.context.getState(getEnergy=True).getPotentialEnergy()*pdf_state.beta)


    #check gettable integrator variables
    integrator_vars = propagator._get_global_integrator_variables()

    #check propagator stability with integrator reset and velocity randomization
    _ = propagator.apply(particle_state, n_steps=1000, reset_integrator=True, apply_pdf_to_context=True, randomize_velocities=True)
