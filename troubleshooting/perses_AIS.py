#!/usr/bin/env python
# coding: utf-8

# anneal with perses...
# 

# In[ ]:


import pickle
import sys
import mdtraj as md
import os
import numpy as np
from simtk import unit


from coddiwomple.openmm.states import OpenMMParticleState, OpenMMPDFState
from coddiwomple.openmm.propagators import *
from perses.annihilation.lambda_protocol import RelativeAlchemicalState
from coddiwomple.openmm.reporters import OpenMMReporter
import os
from coddiwomple.particles import Particle


# In[ ]:


n_steps = int(sys.argv[1])
factory_pkl = sys.argv[2]
traj_folder = sys.argv[3]
save_file = f"{traj_folder}.works.npy"


# In[ ]:


traj = md.Trajectory.load(os.path.join(os.getcwd(), 'test', 'propane_butane', 'solvent.neq.pdb'))
factory_pkl = os.path.join(os.getcwd(), 'test', 'propane_butane.factory.pkl')
with open(factory_pkl, 'rb') as f:
    factory = pickle.load(f)
hybrid_system = factory.hybrid_system
num_frames = traj.n_frames
pdf_state = OpenMMPDFState(system = factory.hybrid_system, alchemical_composability = RelativeAlchemicalState, pressure=1.0 * unit.atmosphere)


# In[ ]:


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
                        'lambda_torsions': x
}

AIS_integrator = OpenMMIAAISLangevinIntegrator(openmm_pdf_update_functions = alchemical_functions,
                                               n_steps = 1000,
                                               temperature=pdf_state.temperature,
                                               collision_rate=1.0 / unit.picoseconds,
                                               timestep=4.0 * unit.femtoseconds,
                                               splitting="P I H N S V R O R V",
                                               constraint_tolerance=1e-6)

propagator = OpenMMBaseIntegrationPropagator(openmm_pdf_state = pdf_state, integrator = AIS_integrator)


# In[ ]:


works = []
for _ in range(100):
    try:
        random_frame = np.random.choice(range(num_frames))
        print(f"random frame chosen: {random_frame}")
        positions = traj.xyz[random_frame] * unit.nanometers

        particle_state = OpenMMParticleState(positions = positions, box_vectors = traj.unitcell_vectors[random_frame] * unit.nanometers)
        _, _return_dict = propagator.apply(particle_state, n_steps = 1000, return_state_work = True, return_proposal_work=True, reset_integrator=True)
        print(_return_dict)
        works.append(_return_dict['state_work'])
    except Exception as e:
        print(e)

works = np.array(works)


# In[ ]:


np.save(save_file, works)

