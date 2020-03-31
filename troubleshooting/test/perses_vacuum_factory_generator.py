#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########################################
# IMPORTS
###########################################
from simtk.openmm import app
from simtk import unit, openmm
import numpy as np
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from perses.annihilation.relative import HybridTopologyFactory
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
from perses.tests import utils
import openeye.oechem as oechem
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
import openmmtools.mcmc as mcmc
import openmmtools.cache as cache
from unittest import skipIf

import pymbar.timeseries as timeseries

import copy
import pymbar

running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

try:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")
except Exception:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")

#############################################
# CONSTANTS
#############################################
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01
ENERGY_THRESHOLD = 1e-1
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CPU")


# In[2]:


def compare_energies(mol_name="naphthalene", ref_mol_name="benzene",atom_expression=['Hybridization'],bond_expression=['Hybridization']):
    """
    Make an atom map where the molecule at either lambda endpoint is identical, and check that the energies are also the same.
    """
    from openmmtools.constants import kB
    from openmmtools import alchemy, states
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
    from perses.annihilation.relative import HybridTopologyFactory
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import simtk.openmm as openmm
    from perses.utils.openeye import iupac_to_oemol, extractPositionsFromOEMol, generate_conformers
    from perses.utils.openeye import generate_expression
    from openmmforcefields.generators import SystemGenerator
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    from perses.tests.utils import validate_endstate_energies
    temperature = 300*unit.kelvin
    # Compute kT and inverse temperature.
    kT = kB * temperature
    beta = 1.0 / kT
    ENERGY_THRESHOLD = 1e-6

    atom_expr, bond_expr = generate_expression(atom_expression), generate_expression(bond_expression)

    mol = iupac_to_oemol(mol_name)
    mol = generate_conformers(mol, max_confs=1)

    refmol = iupac_to_oemol(ref_mol_name)
    refmol = generate_conformers(refmol,max_confs=1)

    from openforcefield.topology import Molecule
    molecules = [Molecule.from_openeye(oemol) for oemol in [refmol, mol]]
    barostat = None
    forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'nonbondedMethod': app.NoCutoff, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus}

    system_generator = SystemGenerator(forcefields = forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs,
                                         small_molecule_forcefield = 'gaff-2.11', molecules=molecules, cache=None)


    topology = generateTopologyFromOEMol(refmol)
    system = system_generator.create_system(topology)
    positions = extractPositionsFromOEMol(refmol)
    
   
    proposal_engine = SmallMoleculeSetProposalEngine([refmol, mol], system_generator)
    proposal = proposal_engine.propose(system, topology, atom_expr = atom_expr, bond_expr = bond_expr)
    geometry_engine = FFAllAngleGeometryEngine()
    new_positions, _ = geometry_engine.propose(proposal, positions, beta = beta, validate_energy_bookkeeping = False)
    _ = geometry_engine.logp_reverse(proposal, new_positions, positions, beta)
    #make a topology proposal with the appropriate data:

    factory = HybridTopologyFactory(proposal, positions, new_positions)
    if not proposal.unique_new_atoms:
        assert geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
        assert geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
        vacuum_added_valence_energy = 0.0
    else:
        added_valence_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential

    if not proposal.unique_old_atoms:
        assert geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
        assert geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
        subtracted_valence_energy = 0.0
    else:
        subtracted_valence_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential

    zero_state_error, one_state_error = validate_endstate_energies(factory._topology_proposal, factory, added_valence_energy, subtracted_valence_energy, beta = 1.0/(kB*temperature), ENERGY_THRESHOLD = ENERGY_THRESHOLD, platform = openmm.Platform.getPlatformByName('Reference'))
    return factory


# In[3]:


import pickle
for derivative in ['fluorobenzene', 'methylbenzene', 'phenol', 'aniline', 'ethylbenzene', 'anisole', '1-methylaniline']:
    factory = compare_energies(mol_name=derivative, ref_mol_name="benzene")
    with open(f"benzene_{derivative}.vacuum.factory.pkl", 'wb') as f:
        pickle.dump(factory, f)


# In[ ]:




