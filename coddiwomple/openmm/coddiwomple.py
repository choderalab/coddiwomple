"""
coddiwomple.py
A pythonic Sequential Monte Carlo (SMC) library for molecular mechanics and Bayesian inference

A compilation of functions (coddiwomple library wrappers) that are commonly used in practice for OpenMM functionality
"""

#####Imports#####
import logging
import os
import sys
from pkg_resources import resource_filename
import random
from coddiwomple.coddiwomple import *
from coddiwomple.openmm.states import OpenMMParticleState, OpenMMPDFState
from coddiwomple.openmm.propagators import *
from coddiwomple.openmm.integrators import *
from coddiwomple.openmm.reporters import OpenMMReporter
from coddiwomple.openmm.utils import *
from coddiwomple.particles import Particle
from perses.annihilation.lambda_protocol import RelativeAlchemicalState
from simtk import unit
import tqdm
import mdtraj as md
import simtk.openmm.app as app


def endstate_equilibration(system,
                           endstate_positions,
                           box_vectors,
                           directory_name,
                           trajectory_prefix,
                           md_topology,
                           number_of_applications,
                           steps_per_application,
                           endstate_parameters = 0.0,
                           alchemical_composability = RelativeAlchemicalState):
    """
    conduct an endstate equilibration with pdf_state parameters defined by `endstate_parameters` with decorrelation
    that will be written to disk

    arguments
        system : openmm.System
            parameterizable system
        endstate_positions : np.ndarray(N,3) * unit.nanometers (or length units)
            starting positions that will be minimized and simulated
        box_vectors : np.ndarray(3,3) * unit.nanometers (or length units)
            starting box vectors
        directory_name : str
            directory that will be written to
        trajectory_prefix : str
            .pdb prefix
        md_topology : mdtraj.Topology
            topology that will write the trajectory
        number_of_applications : int
            number of applications of the propagator
        steps_per_application : int
            number of integration steps per application
        endstate_parameters : float
            endstate parameters
        alchemical_composability : openmmtools.alchemy.AlchemicalState
            composer for alchemical composability creation
    """
    from perses.dispersed.feptasks import minimize
    from perses.dispersed.utils import compute_timeseries
    import mdtraj.utils as mdtrajutils

    #determine pressure
    forces = {type(force).__name__: force for force in system.getForces()}
    if "MonteCarloBarostat" in list(forces.keys()):
        pressure = 1.0 * unit.atmosphere
    else:
        pressure = None
    print(f"pressure: {pressure}")

    particle_state = OpenMMParticleState(positions = endstate_positions,
                                         box_vectors =  np.array(system.getDefaultPeriodicBoxVectors()),
                                        )
    pdf_state = OpenMMPDFState(system = system,
                               alchemical_composability = alchemical_composability,
                               pressure=pressure)

    #set the pdf_state endstate parameters
    pdf_state_parameters = pdf_state.get_parameters()
    reset_pdf_state_parameters = {key: endstate_parameters for key in pdf_state_parameters.keys()}
    pdf_state.set_parameters(reset_pdf_state_parameters)

    langevin_integrator = OMMLI(temperature = pdf_state.temperature,
                                timestep = 2.0 * unit.femtoseconds)

    endstate_propagator = OMMBIP(openmm_pdf_state = pdf_state,
                                 integrator = langevin_integrator)

    reporter = OpenMMReporter(directory_name, trajectory_prefix, md_topology = md_topology)

    particle = Particle(0)
    particle.update_state(particle_state)
    reporter.record([particle])

    minimize(endstate_propagator.pdf_state, particle_state)

    potentials = []
    for i in tqdm.trange(number_of_applications):
        particle_state, _return_dict = endstate_propagator.apply(particle_state, n_steps = steps_per_application)
        reporter.record([particle])
        potentials.append(_return_dict['new_pe'])


    [t0, g, Neff_max, A_t, uncorrelated_indices] = compute_timeseries(np.array(potentials))
    print(f"t0: {t0}, g: {g}, Neff_max: {Neff_max}, A_t: {A_t}, uncorrelated_indices:{uncorrelated_indices}")
    particle_hex = hex(id(particle))
    reporter.hex_dict[particle_hex] = [[reporter.hex_dict[particle_hex][0][q] for q in uncorrelated_indices],
                                       [reporter.hex_dict[particle_hex][1][q] for q in uncorrelated_indices],
                                       [reporter.hex_dict[particle_hex][2][q] for q in uncorrelated_indices]]

    try:
        reporter._write_trajectory(particle_hex, f"{reporter.neq_traj_filename}.{format(reporter.hex_to_index[particle_hex], '04')}.pdb")
    except Exception as e:
        print(e)

def annealed_importance_sampling(system,
                                 endstate_cache_filename,
                                 directory_name,
                                 trajectory_prefix,
                                 md_topology,
                                 alchemical_functions,
                                 number_of_applications,
                                 steps_per_application,
                                 endstate_parameters,
                                 alchemical_composability = RelativeAlchemicalState,
                                 integrator_kwargs = {'temperature': 300.0 * unit.kelvin,
                                                      'collision_rate': 1.0 / unit.picoseconds,
                                                      'timestep': 2.0 * unit.femtoseconds,
                                                      'splitting': "P I H N S V R O R V",
                                                      'constraint_tolerance': 1e-6},
                                record_state_work_interval = 1,
                                trajectory_write_interval = 1,
                                ):
    """
    conduct annealed importance sampling in the openmm regime

    arguments
        system : openmm.System
            parameterizable system
        endstate_cache_filename : str
            path to the endstate cache pdb
        directory_name : str
            directory that will be written to
        trajectory_prefix : str
            .pdb prefix
        md_topology : mdtraj.Topology
            topology that will write the trajectory
        alchemical_functions : dict
            {pdf_parameter <str>: function <str>, lepton-readable}
        number_of_applications : int
            number of applications of the propagator
        steps_per_application : int
            number of integration steps per application
        endstate_parameters : float
            endstate parameters
        alchemical_composability : openmmtools.alchemy.AlchemicalState
            composer for alchemical composability creation
        integrator_kwargs : dict, see default
            kwargs to pass to OMMLIAIS integrator
    """
    #determine pressure
    forces = {type(force).__name__: force for force in system.getForces()}
    if "MonteCarloBarostat" in list(forces.keys()):
        pressure = 1.0 * unit.atmosphere
    else:
        pressure = None

    print(f"pressure: {pressure}")

    traj = md.Trajectory.load(endstate_cache_filename)

    num_frames = traj.n_frames
    pdf_state = OpenMMPDFState(system = system, alchemical_composability = alchemical_composability, pressure=pressure)

    #set the pdf_state endstate parameters
    pdf_state_parameters = pdf_state.get_parameters()
    reset_pdf_state_parameters = {key: endstate_parameters for key in pdf_state_parameters.keys()}
    pdf_state.set_parameters(reset_pdf_state_parameters)

    reporter = OpenMMReporter(directory_name, trajectory_prefix, md_topology = md_topology)

    ais_integrator = OMMLIAIS(
                             alchemical_functions,
                             steps_per_application,
                             **integrator_kwargs)

    ais_propagator = OMMAISPR(openmm_pdf_state = pdf_state,
                              integrator = ais_integrator,
                              record_state_work_interval = record_state_work_interval,
                              reporter = reporter,
                              trajectory_write_interval = trajectory_write_interval,
                              context_cache=None,
                              reassign_velocities=True,
                              n_restart_attempts=0)

    frames = np.random.choice(range(num_frames), number_of_applications)

    particle = Particle(0)

    for i in tqdm.trange(number_of_applications):
        pdf_state.set_parameters(reset_pdf_state_parameters)
        particle_state = OpenMMParticleState(positions = traj.xyz[frames[i]] * unit.nanometers, box_vectors = traj.unitcell_vectors[frames[i]]*unit.nanometers)
        particle.update_state(particle_state)
        try:
            _, _return_dict = ais_propagator.apply(particle_state, n_steps = steps_per_application, reset_integrator=True, apply_pdf_to_context=True)
        except Exception as e:
            print(e)


    return ais_propagator.state_works



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
