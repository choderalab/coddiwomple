"""
OpenMM State Adapter Module
"""

#####Imports#####
from coddiwomple.states import ParticleState, PDFState
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState, IComposableState
from openmmtools.alchemy import AlchemicalState
from openmmtools import utils
from perses.dispersed.utils import check_platform, configure_platform
import os
import numpy as np
import logging
from simtk import unit
from simtk import openmm
from coddiwomple.openmm.utils import get_dummy_integrator

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("openmm_states")
_logger.setLevel(logging.DEBUG)


#ParticleState Adapter
class OpenMMParticleState(ParticleState, SamplerState):
    """
    ParticleState for openmmtools.states.SamplerState
    """
    def __init__(self, positions, velocities = None, box_vectors = None, **kwargs):
        """
        create a SamplerState

        parameters
            positions : np.array(N,3) * unit.nanometers (or length units)
                positions of state
            velocities : np.array(N,3) * unit.nanometers / unit.picoseconds (or velocity units)
                velocities of state
            box_vectors : np.array(3,3) * unit.nanometers (or velocity units)
                current box vectors
        """
        super().__init__(positions = positions, velocities = velocities, box_vectors = box_vectors)


#PDFState Adapter
class OpenMMPDFState(PDFState, CompoundThermodynamicState):
    """
    PDFState for openmmtools.states.CompoundThermodynamicState
    """

    def __init__(self,
                 system,
                 alchemical_composability = AlchemicalState,
                 temperature = 300 * unit.kelvin,
                 pressure = 1.0 * unit.atmosphere,
                 **kwargs):
        """
        Subclass of coddiwomple.states.PDFState that specifically handles openmmtools.states.CompoundThermodynamicStates
        Init method does the following:
            1. create a openmmtools.states.ThermodynamicState with the given system, temperature, and pressure
            2. create an openmmtools.alchemy.AlchemicalState from the given system
            3. create an openmtools.states.CompoundThermodynamicState 1 and 2

        arguments
            system : openmm.System
                system object to wrap
            alchemical_composability : openmmtools.alchemy.AlchemicalState (super), default openmmtools.alchemy.AlchemicalState
                the class holding the method `from_system` which can compose an alchemical state with exposed parameters from a system
            temperature : float * unit.kelvin (or temperature units), default 300.0 * unit.kelvin
                temperature of the system
            pressure : float * unit.atmosphere (or pressure units), default 1.0 * unit.atmosphere
                pressure of the system

        init method is adapted from https://github.com/choderalab/openmmtools/blob/110524bc5079af77d31f5fab464edd7b668ff5ac/openmmtools/states.py#L2766-L2783
        """
        from openmmtools.alchemy import AlchemicalState
        openmm_pdf_state = ThermodynamicState(system, temperature, pressure)
        alchemical_state = alchemical_composability.from_system(system, **kwargs)
        assert isinstance(alchemical_state, IComposableState), f"alchemical state is not an instance of IComposableState"
        self.__dict__ = openmm_pdf_state.__dict__
        self._composable_states = [alchemical_state]
        self.set_system(self._standard_system, fix_state=True)

        #class create an internal context
        integrator = get_dummy_integrator()
        platform = configure_platform(utils.get_fastest_platform().getName())
        self._internal_context = self.create_context(integrator)
        _logger.debug(f"successfully instantiated OpenMMPDFState equipped with the following parameters: {self._parameters}")

    def set_parameters(self, parameters):
        """
        update the pdf_state parameters in place

        arguments
            parameters : dict
                dictionary of variables
        """
        assert set([param[0] for param in self._parameters.items() if param[1] is not None]) == set(parameters.keys()), f"the parameter keys supplied do not match the internal parameter names"
        for key,val in parameters.items():
            assert hasattr(self, key), f"{self} does not have a parameter named {key}"
            setattr(self, key, val)

        self.apply_to_context(self._internal_context)
        _logger.debug(f"successfully updated OpenMMPDFState parameters as follows: {parameters}")

    def get_parameters(self):
        """
        return the current parameters

        return
            returnable_dict : dict
                dictionary of the current parameter names and values {str: float}
        """
        returnable_dict = {i:j for i, j in self._composable_states[0]._parameters.items() if j is not None}
        return returnable_dict

    def reduced_potential(self, particle_state):
        """
        return the reduced potential energy of a particle state

        arguments
            particle_state : coddiwomple.openmm.states.OpenMMParticleState
                the particle state at which the reduced potential will be calculated
        """
        particle_state.apply_to_context(self._internal_context, ignore_velocities=True)

        state = self._internal_context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()
        volume = state.getPeriodicBoxVolume()

        reduced_potential = potential_energy * self.beta

        if self.pressure is not None:
            reduced_potential += self.pressure * volume * self.beta

        return reduced_potential
