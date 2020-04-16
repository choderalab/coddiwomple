"""
Utility Module
"""
#####Imports#####
import numpy as np
from scipy.special import logsumexp

def add_method(object, function):
        """
        Bind a function to an object instance

        arguments
            object : class instance
                object to which function will be bound
            function : function
                function which will be bound to object
        """
        from functools import partial
        setattr(object, function.__name__, partial(function, object))

def dummy_function():
    pass

def unnormalized_weights(works):
    """
    simple utility function to compute particle weights from an array of works

    arguments
        works : np.array
            unnormalized -log weights
    returns
        unnormalized_weights : np.array
    """
    unnormalized_weights = np.exp(-1 * works)
    return unnormalized_weights

def normalized_weights(works):
    """
    simple utility function to normalize an array of works

    arguments
        works : np.array
            unnormalized -log weights
    returns
        normalized_weights : np.array
            normalized_weights = np.exp(-1 * works - logsumexp(-1 * works))
    """
    _unnormalized_weights = unnormalized_weights(works)
    normalized_weights = _unnormalized_weights / np.exp(logsumexp(-1 * works))
    return normalized_weights


#####Observables#####
def nESS(particles, incremental_works = None, **kwargs):
    """
    compute a normalized effective sample size

    arguments
        self : object
            self argument is required for tethering to a method
        particles : list(coddiwomple.particles.Partice)
            the particles of interest
        incremental_works : np.ndarray, default None
            incremental works used to augment the current particle cumulative works.
            If None, the nESS of the particles will be computed directly;
            else, the incremental_works will be added to the current particle cumulative weights
            the nESS will be computed.
    returns
        nESS : float
            normalized effective sample size as defined by 1 / sum_1^N_t()
    """
    works_t = np.array([particle.cumulative_work for particle in particles])
    if incremental_works is not None:
        works_t += incremental_works
    _normalized_weights = normalized_weights(works_t)
    _nESS = 1. / (np.sum(_normalized_weights**2) * len(_normalized_weights))
    return _nESS

#####Thresholds#####
def vanilla_floor_threshold(_observable, floor_threshold = 0.5, **kwargs):
    """
    simple floor threshold
    """
    returnable = False if _observable > floor_threshold else True
    return returnable
