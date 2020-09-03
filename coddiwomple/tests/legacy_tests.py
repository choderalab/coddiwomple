"""
legacy test for the coddiwomple.openmm package.
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
