"""
Unit and regression test for the coddiwomple.openmm package.
"""
# Import package, test suite, and other packages as needed
import numpy as np
import sys
import os
from coddiwomple.openmm.integrators import *
from coddiwomple.openmm.propagators import *
from coddiwomple.openmm.reporters import *
from coddiwomple.openmm.states import *
from coddiwomple.openmm.utils import *
from pkg_resources import resource_filename
import tempfile
import os
import shutil
import pickle
import tqdm
import mdtraj as md

def test_ais_propagator(factory_pickle = f"/data/perses_data/benzene_fluorobenzene.vacuum.factory.pkl",
                        endstate_cache = f"/data/perses_data/fluorobenzene_zero_endstate_cache/vacuum.cache.pdb"):
    """
    Test the OpenMM Annealed Importance Sampling Propagator and its subclasses
    """
    from perses.annihilation.lambda_protocol import RelativeAlchemicalState
    #load the system
    _pickle = resource_filename('coddiwomple', factory_pickle)
    print(f"_pickle: {pickle}")
    if not os.path.exists(_pickle):
        raise ValueError("sorry! %s does not exist. if you just added it, you'll have to re-install" % _pickle)
    with open(_pickle, 'rb') as handle:
        factory = pickle.load(handle)

    #load the endstate trajectory cache
    _traj_filename = resource_filename('coddiwomple', endstate_cache)
    if not os.path.exists(_traj_filename):
        raise ValueError("sorry! %s does not exist. if you just added it, you'll have to re-install" % _traj_filename)
    traj = md.Trajectory.load(_traj_filename)

    num_frames = traj.n_frames
    pdf_state = OpenMMPDFState(system = factory.hybrid_system, alchemical_composability = RelativeAlchemicalState, pressure=None)

    n_steps = 10
    x = 'fractional_iteration'
    alchemical_functions = {
                             'lambda_sterics_core': x,
                             'lambda_electrostatics_core': x,
                             'lambda_sterics_insert': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                             'lambda_sterics_delete': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
                             'lambda_electrostatics_insert': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
                             'lambda_electrostatics_delete': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                             'lambda_bonds': x,
                             'lambda_angles': x,
                             'lambda_torsions': x}

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
