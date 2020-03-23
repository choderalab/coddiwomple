"""
Utility Module
"""
#####Imports#####
import numpy as np

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
    unnormalized_weights = unnormalized_weights(works)
    normalized_weights = unnormalized_weights / np.exp(logsumexp(-1 * works))
    return normalized_weight


#####Observables#####
def nESS(particles, incremental_works = None, **kwargs):
    """
    compute a normalized effective sample size

    arguments
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
    works_t = np.array([particle.cumulative_work() for particle in particles])
    if incremental_works is not None:
        works_t += incremental_works
    normalized_works = normalized_weights(works_t)
    nESS = 1. / (np.sum(normalized_weights**2) * len(normalized_weights))
    return nESS
