"""
Resampler Module
"""
#####Imports#####
import copy
import logging
import numpy as np
from scipy.special import logsumexp

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("resamplers")
_logger.setLevel(logging.DEBUG)

class Resampler():
    """
    Generalized Resampler (super) class for resampling a list of coddiwomple.partiles.Particle objects
    """
    def __init__(self, observable = None, threshold = None, observable_kwargs = None, **kwargs):
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

        parameters
            resample_log : list
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
        from coddiwomple.utils import add_method
        self.resample_log = []

        #some assertion checks
        if observable is not None:
            assert type(observable) == type(add_method), f"the observable is not a function"
            assert type(threshold) == type(add_method), f"the threshold is not a function"
            self.always_resample = False
            self.observable = observable
            self.threshold = threshold
            self.observable_kwargs = observable_kwargs
        else:
            if any(q is not None for q in [threshold, observable_kwargs]):
                _logger.warning(f"threshold ({threshold}) and/or observable_kwargs ({observable_kwargs}) is not None but the observable was set to None; defaulting threshold and observable_kwargs to None")
            self.observable, self.threshold, self.observable_kwargs = None, None, None
            self.always_resample = True

    @property
    def resample_log(self):
        return copy.deepcopy(self.resample_log)

    def resample(self, particles, incremental_works, update_particle_indices = True, **kwargs):
        """
        method to wrap the following:
            1. attempt to resample the particles based on observables
            2. resample particles
            3. increment the resample_log with the current iteration
            4. update particle auxiliary work values
            5. record the state
            6. update the state index

        arguments
            particles : list(coddiwomple.particles.Partice)
                the particles to query
            incremental_works : np.ndarray
                -log(weight) of the current iteration
            update_particle_indices : bool, default True
                whether to update the particle indices
        """
        observable_value = observable(particles, incremental_works) #compute an observable on the particles
        resample_bool = threshold(observable_value, **kwargs) #ask to resample

        if resample_bool:
            #the we will resample the particles
            previous_cumulative_works = np.array(particle.cumulative_work() for particle in particles)
            updated_cumulative_works = previous_cumulative_works + incremental_works
            mean_cumulative_work =  -logsumexp(-updated_cumulative_works) + np.log(len(updated_cumulative_works)) #we always do 'complete' resampling, and we can't do just an empirical average of works
            resampled_indices = self._resample(cumulative_works = updated_cumulative_works, num_resamples = len(updated_cumulative_works), **kwargs) #at present, always resample

            #make necessary pointers for resampling
            copy_particle_auxiliaries = [particle.auxiliary_work() for particle in particles]
            copy_particle_states = [particle.state() for particle in particles]
            copy_particle_indices = [particle.ancestry()[-1] for particle in particles]

            #update the resample_log
            iteration = len(particles[0].incremental_works())
            self.resample_log.append(iteration)

            for current_particle_index, resampling_particle_index, in enumerate(resampled_indices):
                #conduct particle updates
                self._update_particle(particles[current_particle_index],
                                     from_particle_auxiliary_work = copy_particle_auxiliaries[resampling_particle_index],
                                     from_particle_state = copy.deepcopy(copy_particle_states[resampling_particle_index]),
                                     from_particle_index = copy_particle_indices[resampling_particle_index],
                                     cumulative_work = mean_cumulative_work)
        else:
            #then we do not resample the particles
            [self._null_update_particle(particle, incremental_work, **kwargs) for particle, incremental_work in zip(particles, incremental_works)]


    def _null_update_particle(self, particle, incremental_work, **kwargs):
        """
        conduct the following:
            5. update the state (with current state for bookkeeping)
            6. update the work
            7. update the state index (along with ancestry)

        arguments :
            particle : coddiwomple.particles.Partice
                the particle to query
            incremental_work : float
                -log(weight) of the current iteration
        """
        #5.
        particle.update_state(particle.state())

        #6.
        particle.update_work(incremental_work)

        #7.
        particle.update_ancestry(particle.ancestry()[-1])

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
        modified_incremental_work = cumulative_work - particle_to_update.cumulative_work()
        particle_to_update.update_work(modified_incremental_work)

        #7.
        particle_to_update.update_ancestry(from_particle_index)

    def _resample(cumulative_works, num_resamples, **kwargs):
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
    def __init__(self, observable = None, threshold = None, observable_kwargs = None, **kwargs):
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

        parameters
            resample_log : list
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
        super(Resampler, self).__init__(observable = None, threshold = None, observable_kwargs = None, **kwargs)

    def _resample(cumulative_works, num_resamples, **kwargs):
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
