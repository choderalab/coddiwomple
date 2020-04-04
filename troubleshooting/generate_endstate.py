#!/usr/bin/env python
# coding: utf-8


import pickle
import sys

pickl = sys.argv[1]

with open(pickl, 'rb') as f:
    factory = pickle.load(f)


# In[ ]:


hybrid_system = factory.hybrid_system
hybrid_positions = factory.hybrid_positions


# Now we can load this system into an `OpenMMPDFState` and load the positions into an `OpenMMParticleState`

# In[ ]:


from coddiwomple.openmm.states import OpenMMParticleState, OpenMMPDFState
from coddiwomple.openmm.propagators import *
from coddiwomple.openmm.integrators import *

# In[ ]:


from perses.annihilation.lambda_protocol import RelativeAlchemicalState
particle_state = OpenMMParticleState(positions = hybrid_positions, box_vectors = np.array(hybrid_system.getDefaultPeriodicBoxVectors()))
pdf_state = OpenMMPDFState(system = hybrid_system, alchemical_composability = RelativeAlchemicalState, pressure=None)


# the parameters are non-canonical...let's check how perses typically handles these...

# In[ ]:


langevin_integrator = OMMLI(temperature = pdf_state.temperature, timestep = 2.0 * unit.femtoseconds)


# In[ ]:

endstate_propagator = OMMBIP(openmm_pdf_state = pdf_state, integrator = langevin_integrator)


# In[ ]:


from coddiwomple.openmm.reporters import OpenMMReporter
import os
folder = sys.argv[2]
reporter = OpenMMReporter(folder, 'vacuum', md_topology = factory.hybrid_topology)


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

import tqdm
for i in tqdm.trange(150):
    print(i)
    particle_state, _return_dict = endstate_propagator.apply(particle_state, n_steps = 1000)
    reporter.record([particle], save_to_disk=False)


# In[ ]:


reporter.record([particle], save_to_disk=True)

