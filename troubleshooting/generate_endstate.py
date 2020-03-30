#!/usr/bin/env python
# coding: utf-8

# # try to generate some data;
# specifically, i will do the following:
#     
#     1. collect ~100 i.i.d. samples of hybrid propane-butane in vacuum at 300K
#     2. run the forward annealing protocol at several annealing times (figure out what alchemical variables are exposed)
#     3. compute the free energies and variances thereof
#     4. run 2 and 3 again, but with resampling at set intervals (as a function of the effective sample size?)

# ## 1. generate a propane butane alchemical system
# then look at the alchemical variables we can toggle; use perses.tests.test_relative.compare_energies to generate vacuum transform

# In[ ]:


import pickle
with open('propane_butane.factory.pkl', 'rb') as f:
    factory = pickle.load(f)


# In[ ]:


hybrid_system = factory.hybrid_system
hybrid_positions = factory.hybrid_positions


# Now we can load this system into an `OpenMMPDFState` and load the positions into an `OpenMMParticleState`

# In[ ]:


from coddiwomple.openmm.states import OpenMMParticleState, OpenMMPDFState
from coddiwomple.openmm.propagators import *


# In[ ]:


from perses.annihilation.lambda_protocol import RelativeAlchemicalState
particle_state = OpenMMParticleState(positions = hybrid_positions, box_vectors = np.array(hybrid_system.getDefaultPeriodicBoxVectors()))
pdf_state = OpenMMPDFState(system = hybrid_system, alchemical_composability = RelativeAlchemicalState, pressure=1.0 * unit.atmosphere)


# the parameters are non-canonical...let's check how perses typically handles these...

# In[ ]:


x = 'fractional_iteration'
alchemical_functions = {
                        'lambda_sterics_core': x,
                        'lambda_electrostatics_core': x,
                        'lambda_sterics_insert': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                        'lambda_sterics_delete': f"select(step(0.5 - {x}), 0.0, 2.0 * {x})",
                        'lambda_electrostatics_insert': f"select(step(0.5 - {x}), 0.0, 2.0 * {x})",
                        'lambda_electrostatics_delete': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                        'lambda_bonds': x,
                        'lambda_angles': x,
                        'lambda_torsions': x
}


# In[ ]:


langevin_integrator = OpenMMLangevinIntegrator(temperature = pdf_state.temperature, timestep = 2.0 * unit.femtoseconds)


# In[ ]:


endstate_propagator = OpenMMBaseIntegrationPropagator(openmm_pdf_state = pdf_state, integrator = langevin_integrator)


# In[ ]:


from coddiwomple.openmm.reporters import OpenMMReporter
import os
os.system(f"rm -r test2")
reporter = OpenMMReporter('propane_butane', 'solvent', md_topology = factory.hybrid_topology)


# In[ ]:


from coddiwomple.particles import Particle
particle = Particle(0)
particle.update_state(particle_state)
reporter.record([particle])


# In[ ]:


from perses.dispersed.feptasks import minimize


# In[ ]:


minimize(endstate_propagator.pdf_state, particle_state)


# In[ ]:


for i in range(150):
    print(i)
    particle_state, _return_dict = endstate_propagator.apply(particle_state, n_steps = 5000)
    reporter.record([particle], save_to_disk=False)


# In[ ]:


reporter.record([particle], save_to_disk=True)

