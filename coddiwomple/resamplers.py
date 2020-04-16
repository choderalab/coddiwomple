"""
Resampler Module
"""
#####Imports#####
import copy
import logging
import numpy as np
from scipy.special import logsumexp
from coddiwomple.utils import *

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("resamplers")
_logger.setLevel(logging.DEBUG)

class Resampler():
    """
    Generalized Resampler (super) class for resampling a list of coddiwomple.partiles.Particle objects
    """
    def __init__(self, **kwargs):
        """
        initialize a resampler counter

        arguments
            observable : function, default None
                function to compute an observable from a list of particles;
                if None, resampling will always be conducted; else,
                if the threshold of the observable is surpassed, resampling will be conducted
            threshold : function, default None
                the threshold function of the observable;
                the return of the threshold must be a bool, which determines whether to resample
            observable_kwargs : dict, default None
                observable function kwargs

        attributes
            _resampling_logger : list
                list of iterations wherein the particles are resampled
            observable : function, default None
                function to compute an observable from a list of particles;
                if None, resampling will always be conducted; else,
                if the threshold of the observable is surpassed, resampling will be conducted
            threshold : function, default None
                the threshold function of the observable;
                the return of the threshold must be a bool, which determines whether to resample
            observable_kwargs : dict, default None
                observable function kwargs
            always_resample : bool, default True
                if observable is None, then we always resample, else False

        NOTE : the _resample method in this super class conducts sequential importance sampling (i.e. no resampling is conducted)

        TODO : add support for variable number of resamples
        """
        self._resampling_logger = list()
        _logger.debug(f"instantiating empty resampling logger")

    @property
    def resampling_logger(self):
        return copy.deepcopy(self._resampling_logger)

    def resample(self, particles, incremental_works, observable = None, threshold = None, update_particle_indices = True, **kwargs):
        """
        method to wrap the following:
            1. attempt to resample the particles based on observables
            2. resample particles
            3. increment the _resampling_logger with the current iteration
            4. update particle auxiliary work values
            5. record the state
            6. update the work
            7. update the state index
            8. update the iteration by 1

        arguments
            particles : list(coddiwomple.particles.Partice)
                the particles to query
            incremental_works : np.ndarray
                -log(weight) of the current iteration
            observable : function, default None
                function to compute an observable from a list of particles;
                if None, resampling will always be conducted; else,
                if the threshold of the observable is surpassed, resampling will be conducted
            threshold : function, default None
                the threshold function of the observable;
                the return of the threshold must be a bool, which determines whether to resample

            update_particle_indices : bool, default True
                whether to update the particle indices
        """
        if observable is None or threshold is None:
            resample_bool = True
            _logger.debug(f"observable and/or threshold is None; resampling...")
        else:
            _observable_value = observable(particles, incremental_works, **kwargs)
            _logger.debug(f"found observable value to be {_observable_value}")
            resample_bool = threshold(_observable_value, **kwargs)
            _logger.debug(f"resampling: {resample_bool}")

        if resample_bool:
            _logger.debug(f"resampling...")
            #the we will resample the particles
            previous_cumulative_works = np.array([particle.cumulative_work for particle in particles])
            updated_cumulative_works = previous_cumulative_works + incremental_works
            mean_cumulative_work =  -logsumexp(-updated_cumulative_works) + np.log(len(updated_cumulative_works)) #we always do 'complete' resampling, and we can't do just an empirical average of works
            _logger.debug(f"mean cumulative work: {mean_cumulative_work} (i.e. complete resampling)")
            resampled_indices = self._resample(cumulative_works = updated_cumulative_works, num_resamples = len(updated_cumulative_works), **kwargs) #at present, always resample
            _logger.debug(f"resampled_indices : {resampled_indices}")

            #make necessary pointers for resampling
            copy_particle_auxiliaries = [particle.auxiliary_work for particle in particles]
            copy_particle_states = [particle.state for particle in particles]
            copy_particle_indices = [particle.ancestry[-1] for particle in particles]

            #update the _resampling_logger
            iterations = [particle.iteration for particle in particles]
            assert all(_iter == iterations[0] for _iter in iterations), f"the particles are desynchronized"
            iteration = iterations[0]
            self._resampling_logger.append(iteration)

            for current_particle_index, resampling_particle_index, in enumerate(resampled_indices):
                #conduct particle updates
                self._update_particle(particles[current_particle_index],
                                     from_particle_auxiliary_work = copy_particle_auxiliaries[resampling_particle_index],
                                     from_particle_state = copy.deepcopy(copy_particle_states[resampling_particle_index]),
                                     from_particle_index = copy_particle_indices[resampling_particle_index],
                                     cumulative_work = mean_cumulative_work)
        else:
            _logger.debug(f"omitting resample")
            #then we do not resample the particles
            [self._null_update_particle(particle, incremental_work, **kwargs) for particle, incremental_work in zip(particles, incremental_works)]


    def _null_update_particle(self, particle, incremental_work, **kwargs):
        """
        conduct the following:
            5. update the state (with current state for bookkeeping)
            6. update the work
            7. update the state index (along with ancestry)

        arguments
            particle : coddiwomple.particles.Partice
                the particle to query
            incremental_work : float
                -log(weight) of the current iteration
        """
        #5.
        particle.update_state(particle.state)

        #6.
        particle.update_work(incremental_work)

        #7.
        particle.update_ancestry(particle.ancestry[-1])

    def _update_particle(self,
                         particle_to_update,
                         from_particle_auxiliary_work,
                         from_particle_state,
                         from_particle_index,
                         cumulative_work,
                         amend_state_history = True,
                         **kwargs):
        """
        conduct the following:
            4. update particle auxiliary work values (zero auxiliary work and update from particle_to_update_from)
            5. update the state
            6. update the work
            7. update the state index (along with ancestry)

        arguments
            particle_to_update : coddiwomple.particles.Particle
                particle to update
            from_particle_auxiliary_work : float
                the auxiliary work that will update the particle_to_update
            from_particle_state : coddiwomple.states.ParticleState
                the particle state that will update the particle_to_update
            from_particle_index : float
                the particle index that will update the particle_to_update
            cumulative_work : float
                the -log(weight) that will update the particle_to_update;
                NOTE : the incremental work of the particle_to_update will be computed from the cumulative work
            amend_state_history : bool, default True
                whether to modify the particle._state_history (last iteration)
        """
        #4.
        particle_to_update.zero_auxiliary_work()
        particle_to_update.update_auxiliary_work(from_particle_auxiliary_work)

        #5.
        particle_to_update.update_state(copy.deepcopy(from_particle_state), amend_state_history = amend_state_history)

        #6.
        modified_incremental_work = cumulative_work - particle_to_update.cumulative_work
        particle_to_update.update_work(modified_incremental_work)

        #7.
        particle_to_update.update_ancestry(from_particle_index)

    def _resample(self, cumulative_works, num_resamples, **kwargs):
        """
        Dummy _resample method; dummy method does NOT resample

        arguments
            cumulative_works : np.ndarray (floats)
                ndarray of -log(weights_t) of particles; i.e. cumulative weights of particles at iteration t
            num_resamples : int
                number of resamples to generate

        returns
            resampled_indices : list(int)
                list of indices of particles (argument) that are resampled
        """
        return range(num_resamples)

class MultinomialResampler(Resampler):
    """
    Subclass to conduct a vanilla multinomial resampler
    """
    def __init__(self, **kwargs):
        """
        initialize a resampler counter

        arguments
            observable : function, default None
                function to compute an observable from a list of particles;
                if None, resampling will always be conducted; else,
                if the threshold of the observable is surpassed, resampling will be conducted
            threshold : function, default None
                the threshold function of the observable;
                the return of the threshold must be a bool, which determines whether to resample
            observable_kwargs : dict, default None
                observable function kwargs

        attributes
            _resampling_logger : list
                list of iterations wherein the particles are resampled
            observable : function, default None
                function to compute an observable from a list of particles;
                if None, resampling will always be conducted; else,
                if the threshold of the observable is surpassed, resampling will be conducted
            threshold : function, default None
                the threshold function of the observable;
                the return of the threshold must be a bool, which determines whether to resample
            observable_kwargs : dict, default None
                observable function kwargs
            always_resample : bool, default True
                if observable is None, then we always resample, else False
        """
        super().__init__(**kwargs)

    def _resample(self, cumulative_works, num_resamples, **kwargs):
        """
        conduct a multinomial resample (i.e. categorical )

        arguments
            cumulative_works : np.ndarray (floats)
                ndarray of -log(weights_t) of particles; i.e. cumulative weights of particles at iteration t
            num_resamples : int
                number of resamples to generate

        returns
            resampled_indices : list(int)
                list of indices of particles (argument) that are resampled
        """
        from coddiwomple.utils import normalized_weights

        normalized_weights = normalized_weights(cumulative_works)
        resampled_indices = np.random.choice(len(cumulative_works), num_resamples, p = normalized_weights, replace = True)
        return resampled_indices
