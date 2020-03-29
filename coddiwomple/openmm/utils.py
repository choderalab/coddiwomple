"""
Utility Module
"""

def get_dummy_integrator():
    """
    return a dummy Verlet Integrator
    """
    import simtk.openmm as openmm
    from simtk import unit
    return openmm.VerletIntegrator(1.0)
