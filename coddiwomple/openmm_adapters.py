"""
OpenMM Adapter Module:
This module is a consolidation of standard SMC targets, proposals, etc. specifically for openMM interoperability.
"""

#####Imports#####
from coddiwomple.states import ParticleState, PDFState
from coddiwomple.targets import TargetFactory
from coddiwomple.proposals import ProposalFactory
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState, IComposableStates
from openmmtools import cache, utils
from perses.dispersed.utils import check_platform, configure_platform #TODO: make sure this is functional and supports mixed precision
from simtk import unit
import simtk.openmm as openmm

#define the cache
cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())


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
        super(OpenMMParticleState, self).__init__(positions = positions, velocities = velocities, box_vectors = box_vectors)

#PDFState Adapter
class OpenMMPDFState(PDFState, CompoundThermodynamicState):
    """
    PDFState for openmmtools.states.CompoundThermodynamicState
    """

    def __init__(self, system, temperature = 300 * unit.kelvin, pressure = 1.0 * unit.atmosphere, **kwargs):
        """
        Subclass of coddiwomple.states.PDFState that specifically handles openmmtools.states.CompoundThermodynamicStates
        Init method does the following:
            1. create a openmmtools.states.ThermodynamicState with the given system, temperature, and pressure
            2. create an openmmtools.alchemy.AlchemicalState from the given system
            3. create an openmtools.states.CompoundThermodynamicState 1 and 2

        arguments
            system : openmm.System
                system object to wrap
            temperature : float * unit.kelvin (or temperature units), default 300.0 * unit.kelvin
                temperature of the system
            pressure : float * unit.atmosphere (or pressure units), default 1.0 * unit.atmosphere
                pressure of the system

        init method is adapted from https://github.com/choderalab/openmmtools/blob/110524bc5079af77d31f5fab464edd7b668ff5ac/openmmtools/states.py#L2766-L2783
        """
        thermodynamic_state = ThermodynamicState(system, temperature, pressure)
        alchemical_state = AlchemicalState.from_system(system, **kwargs)
        assert isinstance(alchemical_state, IComposableStates), f"alchemical state is not an instance of IComposableStates"
        self.__dict__ = thermodynamic_state.__dict__
        self._composable_states = [alchemical_state]
        self.set_system(self._standard_system, fix_state=True)

    def set_parameters(self, parameters):
        """
        update the pdf_state parameters in place

        arguments
            parameters : np.array or float or dict
                generalizable parameterization object
        """


#TargetFactory Adapter
class OpenMMTargetFactory(TargetFactory):
    """
    Adapter for TargetFactory
    """
    def __init__(self, openmm_pdf_state, **kwargs):
        """
        Initialize the OpenMMTargetFactory

        arguments
            openmm_pdf_state : OpenMMPDFState
                thermodynamic state of the targets
        """
        super(TargetFactory, self).__init__(pdf_state = openmm_pdf_state, **kwargs)

#ProposalFactory Adapter
class OpenMMProposalFactory(ProposalFactory):
    """
    Adapter for ProposalFactory
    """
    def __init__(self, openmm_pdf_state, **kwargs):
        """
        Initialize the OpenMMTargetFactory

        arguments
            openmm_pdf_state : OpenMMPDFState
                thermodynamic state of the targets
        """
        super(ProposalFactory, self).__init__(pdf_state = openmm_pdf_state, **kwargs)
