"""
Utility functions for unit and regression tests for the coddiwomple.openmm package.
"""
import numpy as np
from simtk import unit
from simtk import openmm
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
from openmmtools.alchemy import AlchemicalState

class HarmonicAlchemicalState(AlchemicalState):
    """
    an AlchemicalState subclass to handle the particular case of harmonic testsystem parameters, which include
    'testsystems_HarmonicOscillator_x0' and 'testsystems_HarmonicOscillator_U0'

    attributes
        testsystems_HarmonicOscillator_x0
        testsystems_HarmonicOscillator_U0
    """
    class _LambdaParameter(AlchemicalState._LambdaParameter):
        pass

    #testsystems_HarmonicOscillator_K = _LambdaParameter('testsystems_HarmonicOscillator_K')
    testsystems_HarmonicOscillator_x0 = _LambdaParameter('testsystems_HarmonicOscillator_x0')
    testsystems_HarmonicOscillator_U0 = _LambdaParameter('testsystems_HarmonicOscillator_U0')


def get_harmonic_testsystem(temperature=300 * unit.kelvin,
                            sigma = 1.0 * unit.angstroms,
                            mass = 39.948*unit.amus):
    """
    simple function to generate a tractable Harmonic Oscillator testsystem

    arguments
        temperature : float * unit.kelvin (or any temperature unit)
            temperature of oscillator
        sigma : float * unit.angstroms (or any length unit)
            the standard deviation of the Harmonic Oscillator
        mass : float * unit.amus (or any mass unit)
            reduced mass of the particles
    returns
        testsystem : openmmtools.testsystems.HarmonicOscillator
            test system to return
        period : float * unit.picoseconds (or any time unit)
            period of oscillator
        collision_rate : float / unit.picoseconds (or any time unit)
            collision rate of oscillator
        timestep : float * unit.picoseconds (or any time unit)
            timestep of the oscillator for MD
        alchemical_functions : dict
            dict of alchemical functions; {<name of alchemical parameter>: lepton-readable function}
    """
    from openmmtools.testsystems import HarmonicOscillator

    #define parameters for the harmonic oscillator
    kT = kB * temperature
    beta = 1. / kT
    K = kT / sigma**2
    period = unit.sqrt(mass/K)
    timestep = period / 20.
    collision_rate = 1. / period

    #define some alchemical parameters
    parameters = dict()
    parameters['testsystems_HarmonicOscillator_x0'] = (0*sigma, 2*sigma)
    parameters['testsystems_HarmonicOscillator_U0'] = (0*kT, 1*kT)
    lambda_name = 'fractional_iteration'
    alchemical_functions = {name : f'(1-{lambda_name})*{value[0].value_in_unit_system(unit.md_unit_system)} + {lambda_name}*{value[1].value_in_unit_system(unit.md_unit_system)}' for (name, value) in parameters.items()}

    testsystem = HarmonicOscillator(K=K, mass=mass)
    system = testsystem.system
    positions = testsystem.positions
    return testsystem, period, collision_rate, timestep, alchemical_functions


def load_prereqs(factory_pickle = f"/data/perses_data/benzene_fluorobenzene.vacuum.factory.pkl",
                 endstate_cache = f"/data/perses_data/fluorobenzene_zero_endstate_cache/vacuum.cache.pdb"):
    """
    load the prerequisites for the tests below

    return
        traj : md.Trajectory
        factory : perses.annihilation.relative.HybridTopologyFactory
        alchemical_functions : dict
            dictionary of alchemical functions
    """
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

    return traj, factory, alchemical_functions
