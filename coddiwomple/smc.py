"""
SMC Module
"""

#####Imports#####
import numpy as np
import logging
import os

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("SMC")
_logger.setLevel(logging.DEBUG)


def SMC(target_factory,
        proposal_factory,
        num_initial_particles,
        reporter,
        resampler,
        **kwargs):
    """
    arguments
        target_factory : coddiwomple.distribution_factories.TargetFactory
            the generator for target distributions
        proposal_factory : coddiwomple.distribution_factories.ProposalFactory
            the generator for proposal distributions
        num_initial_particles : int
            the number of initial particles
        reporter : coddiwomple.reporters.Reporter
            reporter object for SMC
        resampler : coddiwomple.resamplers.Resampler
            resampler object

    returns
        particles : list of coddiwomple.particles.Particle objects
            particles with histories and ancestries

    Example of a generic SMC algorithm:

    >>> # Inputs:
    >>> target_factory = TargetFactory(**target_kwargs) #generates a sequence of target distributions (can be a prespecified sequence or on-the-fly)
    >>> proposal_factory = ProposalFactory(**proposal_kwargs) #generates a sequence of proposals (treated same way as the target_factory)
    >>> n_initial_particles = num_initial_particles (number of starting particles)
    >>> reporter = Reporter(save_interval = 1, **reporter_kwargs)
    >>> resampler = Resampler(particles, **resampler_kwargs)
    >>>
    >>>#SMC(target_factory = target_factory, proposal_factory = proposal_factory, num_initial_particles = n_initial_particles, reporter = reporter, resampler = resampler)...
    >>> particles = [Particle(**particle_kwargs) for index in range(n_initial_particles)]
    >>> [proposal_factory.generate_initial_sample(particle, **initial_sample_kwargs) for particle in particles] #samples and logps are updated in the particle objects
    >>> incremental_weights = [target_factory.compute_incremental_work(particle) for particle in particles] #not updated in place
    >>>
    >>> while True:
    >>>     resampler.attempt_resample(particles, incremental_weights, **kwargs) #handles the reweighting from the previous incremental work
    >>>     reporter.record(particles) #record the particles...
    >>>     if all([target_factory.terminate(particle) for particle in particles]):
    >>>         break
    >>>     #otherwise...
    >>>     [proposal_factory.propagate(particle) for particle in particles] #update the state, compute the logp_proposal_ratio within the particle objects
    >>>     incremental_weights = [target_factory.compute_incremental_work(particle) for particle in particles]
    >>>
    >>> # Now for analysis expectations
    >>> analyzer = SMCAnalyzer(particles)
    >>> dg = analyzer.compute_free_energy()
    >>> thermo_length = analyzer.compute_observable(thermodynamic_length)
    >>> particle_ancestry = analyzer.compute_ancestry()
    >>> ESS = analyzer.compute_ESS()
    >>> posterior_coordinates = analyzer.get_samples(n_samples = 1000)
    """
