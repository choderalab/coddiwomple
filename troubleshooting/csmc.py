#!/usr/bin/env python
import numpy as np
import torch

"""
I add logger
"""
#######LOGGING#############################
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("csmc")
_logger.setLevel(logging.DEBUG)
###########################################

"""
Langevin Algorithm utilities
"""
def log_probability(potential):
    """
    generate a log unnormalized probability

    arguments
        potential : np.float
            -log unnormalized probability density
    returns
        log_p : float
            unnormalized log probability
    """
    log_p = np.exp(-potential)
    return log_p

def EL_mu_sigma(x, func, dt, parameters, is_force_function=False,  **kwargs):
    """
    create mean vector and covariance marix for a multivariate gaussian proposal

    argumets
        x : np.array(N)
            positions
        func : function
            either a function that returns a force or a potential that takes x, parameters as args
        dt : float
            timestep
        parameters : arg
            second arg of func
        is_force_function : bool, default False
            whether the func returns a force; if not, it

    returns
        mu : np.array(N)
            mean positions
        Sigma : np.array(N,N)
            covariance matrix

    """
    tau = dt/2.
    if not is_force_function:
        #we have to compute a numerical approximation of the gradient
        force = compute_force(x, func, parameters, **kwargs)
    else: #it is a force function, in which case, we just plug in the x, potential_params
        force = func(x, parameters, **kwargs)

    #now we can compute mu and Sigma
    mu = x + tau * force
    Sigma = 2*tau * np.eye(len(x))

    return mu, Sigma

def compute_force(x, potential_function, potential_parameters, eps=None):
    """
    given a potential function, compute an approximation of the Force (i.e. -grad(potential_function(x, parameters)))

    arguments
        x : np.array(N)
            positions
        potential_function : function
            function that returns a potential (float, -log prob, unnormalized)
        potential_parameters : arg
            second argument of potential_function
        eps : float, default None
            epsilon for computing fprime
    returns
        force : np.array(N)
            force vector

    """
    from scipy.optimize import approx_fprime

    if not eps:
        eps = np.sqrt(np.finfo(float).eps)
    epsilons = [eps]*len(x)
    grad_potential = approx_fprime(x, potential_function, epsilons, potential_parameters)
    return -grad_potential

def uncontrolled_ULA_proposal_utility(x, potential, dt, parameter, x_forward=None):
    """
    make an uncontrolled forward proposal along with its associated log kernel forward and backward probability;
    if x_forward is specified, x_forward will be returned unaltered, and the logp_forward/backward will be computed w.r.t.

    arguments
        x : np.array(N)
            starting position
        potential : function
            potential function that takes x, parameter
        dt : float
            time increment
        parameter : arg
            second argument of `potential`
        x_forward : np.array(N), default None
            forward proposed position

    returns
        x_forward : np.array(N)
            forward position
        logp_forward : float
            log probability of the forward kernel
        logp_backward : float
            log probability of the backward kernel
    """
    from scipy.stats import multivariate_normal
    assert type(x) == np.ndarray
    assert x.ndim == 1

    forward_mu, forward_Sigma = EL_mu_sigma(x, potential, dt, parameter)
    if not x_forward:
        x_forward = multivariate_normal.rvs(mean = forward_mu, cov = forward_Sigma)
    if type(x_forward) != np.ndarray:
        x_forward = np.array([x_forward])
    logp_forward = multivariate_normal.logpdf(x_forward, forward_mu, forward_Sigma)

    backward_mu, backward_Sigma = EL_mu_sigma(x_forward, potential, dt, parameter)
    logp_backward = multivariate_normal.logpdf(x, backward_mu, backward_Sigma)
    return x_forward, logp_forward, logp_backward




def compute_ULA_logw(x_tm1, x_t, forcet_xtm1, forcet_xt, potentialtm1_xtm1, potentialt_xt, dt):
    """
    compute the unadjusted langevin algorithm log_weight

    arguments
        x_tm1 : np.array(N)
            positions at t-1
        forcet_xt : np.array(N)
            force vector at x_t
        forcet_xtm1 : np.array(N)
            force vector at x_{t-1}
        potentialtm1_xtm1 : float
            potential at t-1 (with positions x_tm1)
        potentialt_xt : float
            potential at t (with positions x_t)
        dt : float
            time increment

    returns
        logw : float
            unnormalized ULA log weight

    """
    logw = (
            potentialtm1_xtm1
             + 0.5 * x_tm1.dot(forcet_xtm1)
             + (dt/8.)*np.sum(forcet_xtm1**2)
             - potentialt_xt
             - 0.5 * x_t.dot(forcet_xt)
             - (dt/8.)*np.sum(forcet_xt**2)
             - 0.5 * x_t.dot(forcet_xtm1)
             + 0.5 * x_tm1.dot(forcet_xt)
                )
    return logw

def compute_generalized_logw(log_gamma_old, log_gamma_new, log_forward_kernelt, log_backward_kernel_tm1):
    """
    compute a generalized log incremental (unnormalized) weight

    arguments
        log_gamma_old : float
            log unnormalized probability distribution at t-1 (and position x_{t-1})
        log_gamma_new : float
            log unnormalized probability distribution at t (and positions x_t)
        log_forward_kernelt : float
            log of forward transition kernel probability
        log_backward_kernel_tm1 : float
            log of backward transition kernel probability

    returns
        logw : float
            log incremental weight of path importance
    """
    logw = log_gamma_new + log_backward_kernel_tm1 - log_gamma_old - log_forward_kernelt
    return logw

"""
SIS utilities
"""
def SIS(x, potential, lambda_sequence, dt, metropolize=True):
    """
    single iteration of SIS that allows for metropolization (in which case, it is AIS)

    arguments
        x : np.array(N)
            starting position
        potential : function
            potential function that takes x, parameter
        lambda_sequence : np.array(T+1, A)
            sequence of parameters with T+1 parameters and each parameter is a vector of dimension A
        dt : float
            time increment
        metropolize : bool, default True
            whether to metropolize the ULA moves; if so, this is MALA AIS
    returns
        incremental_log_weights : np.array(T+1)
            incremental log weights of SIS
        trajectory : np.array(T+1, N)
            trajectory of particle
        accept_rate : float or None
            acceptance rate of metropolization. None if metropolize=False

    NOTE : x is assumed to be drawn from potential(parameter = lambda_sequence[0])
    """
    lambda_sequence_length = lambda_sequence.shape[0] #the length of the lambda sequence is the first
    assert type(x) == np.ndarray

    if metropolize:
        accept_tally = 0.
    else:
        accept_tally = None #there is no metropolization

    incremental_log_weights = [0.] #this is true of x is sampled from the appropriate prior (the 0th parameter sequence entry)
    trajectory = [x] #plop in the first position for good measure

    log_gammatm1_tm1 = -potential(x, lambda_sequence[0]) #initialize log unnormalized probability  so that we don't have to recompute every step...

    for idx in range(1, len(lambda_sequence)): #we start from the first iteration since
        #make a forward proposal
        x_forward, logp_forward, logp_backward = uncontrolled_ULA_proposal_utility(x, potential, dt, lambda_sequence[idx])

        #compute log_gamma_t
        log_gammat_t = -potential(x_forward, lambda_sequence[idx])


        if metropolize: #then we need to compute a metropolization step
            log_gammat_xtm1 = -potential(x, lambda_sequence[idx]) #only for AIS
            logp_accept = np.min([0., log_gammat_t + logp_backward - log_gammat_xtm1 - logp_forward])
            if np.log(np.random.rand()) <= logp_accept:
                x = x_forward #accept and rename x
                accept_tally += 1
            else:
                #do nothing and retain x
                pass
            logw = log_gammat_xtm1 - log_gammatm1_tm1 #this is the log_weight for AIS
        else:
            logw = compute_generalized_logw(log_gammatm1_tm1, log_gammat_t, logp_forward, logp_backward)
            logw_check = log_gammat_t + logp_backward - log_gammatm1_tm1 - logp_forward
            assert np.isclose(logw, logw_check)

            x = x_forward #always accepted

        log_gammatm1_tm1 = -potential(x, lambda_sequence[idx]) #reinitialize the old log gamma
        trajectory.append(x)
        incremental_log_weights.append(logw)

    if metropolize:
        accept_rate = accept_tally / (lambda_sequence_length - 1)
    else:
        accept_rate = None

    return incremental_log_weights, trajectory, accept_rate


def gmm_ensemble_SIS(num_particles, potential, lambda_sequence, dt, mix_components, mus, Sigmas, metropolize = True):
    """
    wrap the SIS function that draws initial function from a gaussian mixture model

    arguments
        num_particles : int
            number of particles to anneal
        potential : function
            potential function that takes x, parameter
        lambda_sequence : np.array(T+1, A)
            sequence of parameters with T+1 parameters and each parameter is a vector of dimension A
        dt : float
            time increment
        mix_components : np.array(Q)
            gaussian mixture components; asserts this is normalized
        mus : np.array(Q, N)
            mean vectors for each mixing component
        Sigmas : np.array(Q, N, N)
            mixing covariance matrices
        metropolize : bool, default True
            whether to metropolize the ULA moves; if so, this is MALA AIS

    returns
        trajectories : np.array(T+1, N)
            trajectory of particles
        log_weights : np.array(T+1)
            log incremental weights of particles
        full_acceptance_rate : float or None
            mean acceptance rate of metropolization if metropolization

    """
    from scipy.stats import multivariate_normal
    import tqdm
    assert np.sum(mix_components) == 1., f"the mixture components should be normalized. the current sum of mixture components is {np.sum(mix_components)}"
    #loggers
    trajectories = []
    log_weights = []
    if metropolize:
        full_acceptance_rate = 0.
    else:
        full_acceptance_rate=None

    for iteration in tqdm.trange(num_particles):
        #draw a sample
        mixture_component = np.random.choice(range(len(mix_components)))
        mu, Sigma = mus[mixture_component], Sigmas[mixture_component]
        x = multivariate_normal.rvs(mu, Sigma)
        if type(x) != np.ndarray:
            x = np.array([x])
        logw, traj, acceptance_rate = SIS(x, potential, lambda_sequence, dt, metropolize=metropolize)
        log_weights.append(logw)
        trajectories.append(traj)
        if metropolize:
            full_acceptance_rate += acceptance_rate

    trajectories = np.array(trajectories)
    log_weights = np.array(log_weights)
    if metropolize:
        full_acceptance_rate = full_acceptance_rate / num_particles

    return trajectories, log_weights, full_acceptance_rate

"""
twisted utilities
"""
def twisted_gmm_components(uncontrolled_alphas, uncontrolled_mus, uncontrolled_Sigmas, A0, b0, c0):
    """
    compute a twisted gaussian mixture model mixing components, twisted_mu, and twisted_Sigma

    arguments
        uncontrolled_alphas : np.array(A)
            mixture components
        uncontrolled_mus : np.array(A, N)
            matrix of mean vectors corresponding to mixture components A
        uncontrolled_Sigmas : np.array(A, N, N)
            tensor of covariance matrices corresponding to each mixture component
        A0 : np.array(N,N)
            twisting matrix
        b0 : np.array(N)
            twisting control
        c0 : float
            twisting scalar

    returns
        log_alpha_tildes : np.array(A)
            twisted mixing components
        Sigma_tilde_js : np.array(A, N,N)
            twisted covariances

    """
    assert len(uncontrolled_alphas) == len(uncontrolled_mus)
    components, dimensions = uncontrolled_mus.shape
    assert uncontrolled_Sigmas.shape == (components, dimensions, dimensions)

    #compute mixture components
    Sigma_tilde_js = np.linalg.inv(np.linalg.inv(uncontrolled_Sigmas) + 2.0*A0)
    log_zetas = np.array([
                            gmm_log_zetas(sigma_tilde_j, sigma_j, mu_j, b0, c0)
                            for sigma_tilde_j, sigma_j, mu_j in
                            zip(Sigma_tilde_js, uncontrolled_Sigmas, uncontrolled_mus)
                            ])
    log_alpha_tildes = np.log(uncontrolled_alphas) + log_zetas

    return log_alpha_tildes, Sigma_tilde_js

def twisted_gmm_proposal(log_alpha_tildes, Sigma_tilde_js, uncontrolled_mus, uncontrolled_Sigmas, b0):
    """
    make a twisted gaussian mixture model proposal

    arguments
        log_alpha_tildes : np.array(A)
            twisted unnormalized mixture components
        Sigma_tilde_js : np.array(A, N, N)
            twisted covariance matrices of the twisted mixture components
        uncontrolled_mus : np.array(A, N)
            array of untwisted mixing component mean vectors
        uncontrolled_Sigmas : np.array(A, N, N)
            tensor of covariance matrices corresponding to each mixture component
        b0 : np.array(N)
            twisted control

    returns
        x : np.array(N)
            proposed position
        logpdf : float
            log proposal

    """
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    normalized_alpha_tildes = np.exp(log_alpha_tildes - logsumexp(log_alpha_tildes))
    #choose a component
    component_index = np.random.choice(range(len(normalized_alpha_tildes)), p = normalized_alpha_tildes)

    #then choose a position based on that gaussian
    Sigma_tilde_j = Sigma_tilde_js[component_index]
    mu_j = uncontrolled_mus[component_index]
    Sigma_j = uncontrolled_Sigmas[component_index]
    twisted_mean = np.matmul(Sigma_tilde_j, np.matmul(np.linalg.inv(Sigma_j), mu_j) - b0)
    twisted_Sigma = Sigma_tilde_j
    x = multivariate_normal.rvs(mean=twisted_mean, cov = twisted_Sigma)
    if type(x) != np.ndarray: #if we return a float
        x = np.array([x])
    logpdf = multivariate_normal.logpdf(x, mean=twisted_mean, cov = twisted_Sigma)
    return x, logpdf

def compute_twisted_gmm_lognormalizer(log_alpha_tildes):
    """
    compute the twisted gaussian mixture model log normalization constant with unnormalized log_alpha_tildes

    argumets
        log_alpha_tildes : np.array(A)
            twisted unnormalized mixture components
    returns
        lognormalizer : float
            log normalization constant of twisted gmm kernel
    """
    from scipy.special import logsumexp
    return logsumexp(log_alpha_tildes)


def gmm_log_zetas(Sigma_tilde_j, Sigma_j, mu_j, b0, c0):
    """
    compute the logzeta_js mixture components

    arguments
        Sigma_tilde_j : np.array(N, N)
            twisted covariance matrix of the twisted mixture component
        Sigma_j : np.array(N,N)
            untwisted covariance matrix
        b0 : np.array(N)
            twisted control
        c0 : float
            twisted scalar

    returns
        log zeta : float
            twisted proposal components


    """
    from scipy.spatial.distance import mahalanobis
    comp1 = -0.5 * np.log(np.linalg.det(Sigma_j))
    comp2 = 0.5 * np.log(np.linalg.det(Sigma_tilde_j))
    comp3 = 0.5 * mahalanobis(np.matmul(np.linalg.inv(Sigma_j), mu_j), b0, Sigma_tilde_j)**2
    comp4 = -0.5 * mahalanobis(mu_j, np.zeros(len(mu_j)), np.linalg.inv(Sigma_j))**2
    comp5 = -c0
    return comp1 + comp2 + comp3 + comp4 + comp5

def Theta_t(x_tm1, A_t, dt):
    """
    compute Theta_t = (I_d _ 2*dt*A_t(x_tm1))^-1

    arguments
        x_tm1 : np.array(N)
            position at time t-1
        A_t : np.array(N,N)
            twisting matrix at time t
        dt : float
            time increment

    returns
        theta : np.array(N,N)
            theta matrix

    """
    theta = np.linalg.inv(np.eye(len(x_tm1)) + 2. * dt * A_t)
    return theta

def f_t(x_tm1, potential_function, parameters, dt, **kwargs):
    """
    NOTE : parameters should reflect potential function at time t (not t-1)
    compute f_t(x_tm1) = x_tm1 + 0.5 * dt * forcet(x_tm1) = mu

    arguments
        x_tm1 : np.array(N)
            positions at time t-1
        potential_function : function
            function that return -log unnormalized probability density
        parameters : arg
            argument to the potential function
        dt : float
            time increment

    returns
        mu : np.array(N)
            forward euler discretization
    """
    mu, cov = EL_mu_sigma(x_tm1, potential_function, dt, parameters, is_force_function=False,  **kwargs)
    return mu


def twisted_forward_proposal(theta, f, dt, b, **kwargs):
    """
    make a forward twisted proposal; this is a simple wrapper that takes theta, f, dt, b_t (returnable) to return a twisted forward proposal

    arguments
        theta : np.array(N,N)
            twisted theta matrix
        f : np.array(N)
            f_t untwisted control
        dt : float
            time increment
        b : np.array(N)
            twisted control vector
    return
        x : np.array(N)
            proposed position
        logp_forward : float
            the log probability of the forward proposal
    """
    from scipy.stats import multivariate_normal
    twisted_mean = np.matmul(theta, f - dt*b)
    twisted_covariance = dt * theta
    x = multivariate_normal.rvs(mean=twisted_mean, cov = twisted_covariance)
    logp_forward = multivariate_normal.logpdf(x, twisted_mean, twisted_covariance)
    if type(x) != np.ndarray:
        x = np.array([x])
    return x, logp_forward

def twisted_forward_log_normalizer(theta, f, b, c, d, dt):
    """
    perform one line computation (wrapper) to calculate the log normalization constant of the twisted forward proposal

    arguments
        theta : np.array(N,N)
            twisted theta matrix
        f : np.array(N)
            f_t untwisted control
        b : np.array(N)
            twisted control vector
        c : float
            twisted potential float
        d : float
            twisted potential float
        dt : float
            time increment

    returns
        log_forward_normalizer : float
            log forward normalization constant
    """
    from scipy.spatial.distance import mahalanobis

    #there are 3 components that will be computed separately and added.
    comp1 = 0.5 * np.log(np.linalg.det(theta))
    comp2 = (1./(2.*dt)) * mahalanobis(f, dt*b, np.linalg.inv(theta))**2
    comp3 = -(1./(2.*dt)) * f.dot(f) - c - d

    log_forward_normalizer = comp1 + comp2 + comp3
    return log_forward_normalizer


def twisted_zeroth_log_weight(x0, A0, b0, c0, log_pi0_normalizer):
    """
    compute twisted potential log weight at time 0;
    there is an internal consistency to ensure that if the log_pi0_normalizer is 0., then presumably the twists are also naught;
    therefore, the log weight should be zero

    arguments
        x0 : np.array(N)
            position proposed
        A0 : np.array(N,N)
            twisting matrix
        b0 = np.array(N)
            twisting vector
        c0 : float
            twisting float
        log_pi0_normalizer : float
            normalization constant of gmm
    return
        log_w0
    """
    from scipy.spatial.distance import mahalanobis
    twisted_phi = mahalanobis(x0, np.zeros(len(x0)), A0)**2 + x0.dot(b0) + c0
    log_w0 = log_pi0_normalizer + twisted_phi
    if np.all(A0 == 0.) and np.all(b0 == 0.) and np.all(c0 == 0.):
        #this is an internal consistency check. A0, b0, c0 create the log_pi0_normalizer...
        #print(f"zeroth parameters are 0, the log_w0 should also be zero")
        assert np.isclose(log_w0, 0.)
    return log_w0

def twisted_t_log_weight(x, uncontrolled_logw, log_forward_normalizer, A, b, c, d):
    """
    wrapper to compute the twisted log weight at iteration t

    arguments
        x : np.array(N)
            position at iteration t
        uncontrolled_logw : float
            precomputed uncontrolled log incremental weight
        log_forward_normalizer : float
            logK_t(\psi_t)(xtm1)
        A : np.array(N,N)
            twisted matrix
        b : np.array(N)
            twisted control vector
        c : float
            twisted float
        d : float
            twisted float

    returns
        twisted_logw : float
            twisted log incremental weight
    """
    from scipy.spatial.distance import mahalanobis
    phi_t = mahalanobis(x, np.zeros(len(x)), A)**2 + b.dot(x) + c + d
    logw = uncontrolled_logw + log_forward_normalizer + phi_t
    if np.all(A==0.) and np.all(b==0.) and b==0. and d==0.:
        #print(f"twisted t log weight is zero. the log weight should be equal to the uncontrolled log weight...")
        assert logw == uncontrolled_logw #another internal consistency check
    return logw

def twisted_smc(potential,
                parameter_sequence,
                dt,
                initial_twisting_functions,
                twisting_functions,
                twisting_parameters,
                gmm_uncontrolleds,
                compute_optimal_d = False,
                **kwargs):
    """
    run an iteration of non-resampled controlled sequential monte carlo

    arguments
        potential : function
            function that takes positions and parameter and returns a float
        parameter_sequence : np.array(T+1, R)
            parameter sequence that passes to potential. each potential takes an np.array(R)
        dt : float
            time increment
        initial_twisting_functions : dict
            dictionary with keys ['A', 'b', 'c']
            initial_twisting_functions['A'] : np.array(N,N); twisted modifying covariance matrix
            initial_twisting_functions['b'] : np.array(N); twisted modifying control vector
            initial_twisting_functions['c'] : float ; twisted modifying float
        twisting_functions : dict
            dictionary with keys ['A', 'b', 'c']; each is a function or list of functions of length T
            twisting_function_sequence['A'] : returns np.array(N,N) (takes positions vector x)
            twisting_function_sequence['b'] : returns np.array(N) (takes position vector x)
            twisting_function_sequence['c'] : returns float
        twisting_parameters : dict, default None
            dictionary with keys ['A', 'b', 'c', 'd'] that retain a np.array corresponding to the parameter argument of the twisting_functions sequence
            if each value of the twisting_function_sequence is a single function/class, then shape[0]
        gmm_uncontrolleds : dict
            gaussian mixture model parameters for uncontrolled_gaussians; keys include ['alphas', 'mus', 'Sigmas']
            gmm_uncontrolleds['alphas'] : np.array(M)
                mixture components
            gmm_uncontrolleds['mus'] : np.array(M, N)
                mixture component means
            gmm_uncontrolleds['Sigmas'] : np.array(M,N,N)
                mixture component covariance matrices
        compute_optimal_d : bool, default False
            whether to compute the optimal d twisting potential

    returns
        trajectory : np.array(T+1, N)
            trajectory of positions
        logw : np.array(T+1)
            log incremental weights

    #TODO : write a compact twisting function for A,b,c,d that returns a summation of twisted functions.
    """
    from scipy.stats import multivariate_normal

    #determine the formatting of the twisted functions
    twisty_A_is_list = True if type(twisting_functions['A']) == list else False
    twisty_b_is_list = True if type(twisting_functions['b']) == list else False
    twisty_c_is_list = True if type(twisting_functions['c']) == list else False

    #first thing to do is make a GMM proposal...
    uncontrolled_alphas, uncontrolled_mus, uncontrolled_Sigmas = gmm_uncontrolleds['alphas'], gmm_uncontrolleds['mus'], gmm_uncontrolleds['Sigmas'] #pull uncontrolled parameters
    A0, b0, c0 = initial_twisting_functions['A'], initial_twisting_functions['b'], initial_twisting_functions['c'] # pull twisting functions from dictionary
    unnormalized_twisted_log_alphas, twisted_Sigmas = twisted_gmm_components(uncontrolled_alphas, uncontrolled_mus, uncontrolled_Sigmas, A0, b0, c0) #compute twisted gmm components
    x, logp_x = twisted_gmm_proposal(unnormalized_twisted_log_alphas, twisted_Sigmas, uncontrolled_mus, uncontrolled_Sigmas, b0) #make a gmm proposal and its logp_x
    gmm_normalizer = compute_twisted_gmm_lognormalizer(unnormalized_twisted_log_alphas) #compute the normalization constant of the gmm
    log_w0 = twisted_zeroth_log_weight(x, A0, b0, c0, gmm_normalizer) #compute the log weight at time 0; there is an internal consistency check here

    #create loggers
    trajectory = [x]
    log_incremental_weights = [log_w0]

    #now for the loop that iterates across all potential parameters
    log_gammatm1_xtm1 = -potential(x, parameter_sequence[0]) #compute old log unnormalized probability
    for idx in range(1, len(parameter_sequence)):
        _lambda = parameter_sequence[idx]
        #make a twisted proposal
        if twisty_A_is_list:
            A = twisting_functions['A'][idx-1](x)
        else:
            A = twisting_functions['A'](x, twisting_parameters['A'][idx-1])
        if twisty_b_is_list:
            b = twisting_functions['b'][idx-1](x)
        else:
            b = twisting_functions['b'](x, twisting_parameters['b'][idx-1])
        if twisty_c_is_list:
            c = twisting_functions['c'][idx-1]()
        else:
            c = twisting_functions['c'](twisting_parameters['c'][idx-1])

        #define theta and f for the forward proposal
        theta = Theta_t(x, A, dt)
        f = f_t(x, potential, _lambda, dt)

        #make a forward proposal
        x_forward, logp_forward = twisted_forward_proposal(theta, f, dt, b)

        #compute the forward log normalizer
        if compute_optimal_d:
            raise Exception(f"not currently implemented")
        else:
            d = 0.
        log_forward_normalizer = twisted_forward_log_normalizer(theta, f, b, c, d, dt)

        #compute the untwisted forward/backward proposals
        _, logp_forward, logp_backward = uncontrolled_ULA_proposal_utility(x, potential, dt, _lambda, x_forward = x_forward)

        #compute log_gamma_t
        log_gammat_xt = -potential(x_forward, _lambda)

        #compute uncontrolled ULA incremental log works
        uncontrolled_logw = compute_generalized_logw(log_gammatm1_xtm1, log_gammat_xt, logp_forward, logp_backward)

        #compute the twisted weight; there is an internal check here to ensure that if A,b,c,d is zero, then the controlled_logw should be equal to the uncontrolled logw
        controlled_logw = twisted_t_log_weight(x_forward, uncontrolled_logw, log_forward_normalizer, A, b, c, d)

        #recorder
        trajectory.append(x_forward)
        log_incremental_weights.append(controlled_logw)

        #resetters
        x = x_forward
        log_gammatm1_xtm1 = log_gammat_xt

    return trajectory, log_incremental_weights


######
#CSMC Class
######
class ULAEnsembleCSMC():
    """
    conduct an ensemble controlled SMC
    """
    def __init__(self,
                 potential,
                 parameter_sequence,
                 dt,
                 gmm_mixing_components,
                 gmm_means,
                 gmm_covariance_matrices,
                 A0_twisting_model=None,
                 b0_twisting_model=None,
                 c0_twisting_model=None,
                 A_twisting_model=None,
                 b_twisting_model=None,
                 c_twisting_model=None,
                 loss_function = torch.nn.functional.mse_loss,
                 training_epochs=10,
                 validate_fraction=None,
                 optimizer = torch.optim.SGD,
                 optimizer_kwargs = {'lr': 1e-3},
                 **kwargs):
        """
        create a controlled SMC object that will execute sequential iterations of training and running

        arguments
            potential : function (or any object that can take x and parameter; see below)
                this is some function-like object that can take a latent position (x) and a parameter and return a -log unnormalized probability density
            parameter_sequence : iterable (list or numpy array)
                iterable object that parameterizes the potential from the prior to the posterior
            dt : float
                time increment
            gmm_mixing_components : np.array(R)
                mixing components of a gaussian mixture model; the mixture components should sum to 1
            gmm_means : np.array(R, N)
                2d array of means that correspond with the mixing componnets
            gmm_covariance_matrices : np.array(R, N, N)
                covariance matrices of the mixing components
            A0_twisting_model : nn.Sequential, or subclass thereof
                the input argument is exclusively None
                the output is torch.tensor(N,N)
            b0_twisting_model : nn.Sequential, or subclass thereof
                the input argument is exclusively None
                the output is a torch.tensor(N)
            c0_twisting model : nn.Sequential, or subclass thereof
                the input argument is exclusively None
                the output is a torch.tensor(float)
            A_twisting_model : nn.Sequential, or subclass thereof
                the input is a torch.tensor(N)
                the output is a torch.tensor(N, N)
            b_twisting_model : nn.Sequenial, or subclass thereof
                the input is a torch.tensor(N)
                the output is a torch.tensor(N)
            c_twisting_model : nn.Sequential, or subclass thereof
                the input is exclusively None
                the output s a torch.tensor(float)


        """
        #initialize some instance variables that are unchanged throughout
        self.potential = potential
        self.parameter_sequence = parameter_sequence
        self.dt = dt
        self.latent_dimension = len(gmm_means[0])

        #initialize for the gaussian mixture model
        assert np.isclose(np.sum(gmm_mixing_components), 1.)
        self.gmm_mixing_components = gmm_mixing_components
        self.gmm_means = gmm_means
        self.gmm_covariance_matrices = gmm_covariance_matrices

        #initialize the twisting models
        local_vars = locals()
        initial_models = ['A0_twisting_model', 'b0_twisting_model', 'c0_twisting_model']
        t_models = ['A_twisting_model', 'b_twisting_model', 'c_twisting_model']

        for entry in initial_models: #initial twisting modules
            if local_vars[entry] is not None:
                setattr(self, entry, local_vars[entry])
            else:
                setattr(self, entry, None)
        for entry in t_models: #t twisting modules
            if local_vars[entry] is not None:
                setattr(self, entry, local_vars[entry])
            else:
                setattr(self, entry, None)

        #instantiate tensor parameter data...
        #the parameters have shapes like [twisting_iteration][sequence_iteration][parameter_shape]
        #the twisting_iteration starts at 1 since the 0th iteration is uncontrolled
        self.A_parameters, self.b_parameters, self.c_parameters = [], [], []

        #initialize a counter for the number of twists
        self.twisting_iteration = 0

        #ADP parameters
        self._loss_function = loss_function
        self._training_epochs = training_epochs
        self._validate_fraction = None
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs


    def run_twisted_smc(self, num_particles, compute_optimal_ds = False, save_trajectory = True):
        """

        """
        import tqdm

        #make some loggers
        logws = np.zeros((num_particles, len(self.parameter_sequence)))
        trajectory = np.zeros((num_particles, len(self.parameter_sequence), self.latent_dimension)) if save_trajectory else None

        for particle_idx in tqdm.trange(num_particles):
            x, logw = self.conduct_twisted_gmm_proposal() #make the gmm proposal
            logws[particle_idx, 0] = logw #record logw
            if save_trajectory: trajectory[particle_idx, 0, :] = x #record trajectory

            log_gammatm1_xtm1 = -self.potential(x, self.parameter_sequence[0])

            for t in range(1, len(self.parameter_sequence)):
                _lambda = self.parameter_sequence[t]
                A, b, c = self.aggregate_twists(t, x)
                x_forward, log_forward_normalizer = self.conduct_wrapped_forward_proposal(x = x,
                                                                                     _lambda = _lambda,
                                                                                     A_twist = A,
                                                                                     b_twist = b,
                                                                                     c_twist = c,
                                                                                     propose_x_forward = True,
                                                                                     compute_optimal_ds=compute_optimal_ds)
                log_gammat_xt = -self.potential(x_forward, _lambda)
                logw = self.compute_twisted_forward_logw(x = x,
                                                         _lambda = _lambda,
                                                         x_forward = x_forward,
                                                         log_gammatm1_xtm1 = log_gammatm1_xtm1,
                                                         log_gammat_xt = log_gammat_xt,
                                                         log_forward_normalizer = log_forward_normalizer,
                                                         A = A,
                                                         b = b,
                                                         c = c,
                                                         compute_optimal_ds = compute_optimal_ds)

                #recorder
                logws[particle_idx, t] = logw
                if save_trajectory: trajectory[particle_idx, t, :] = x #record trajectory

                #reinitialize
                x = x_forward
                log_gammatm1_xtm1 = log_gammat_xt

        return logws, trajectory

    def compute_twisted_forward_logw(self,
                                     x,
                                     _lambda,
                                     x_forward,
                                     log_gammatm1_xtm1,
                                     log_gammat_xt,
                                     log_forward_normalizer,
                                     A,
                                     b,
                                     c,
                                     compute_optimal_ds = False):
        """
        simple wrapper to compute the twisted log forward weight
        """
        if compute_optimal_ds:
            raise Exception(f"not currently implemented")
        else:
            d = 0.

        _, logp_forward, logp_backward = uncontrolled_ULA_proposal_utility(x, self.potential, self.dt, _lambda, x_forward = x_forward)
        uncontrolled_logw = compute_generalized_logw(log_gammatm1_xtm1, log_gammat_xt, logp_forward, logp_backward)
        controlled_logw = twisted_t_log_weight(x_forward, uncontrolled_logw, log_forward_normalizer, A, b, c, d)
        return controlled_logw

    def conduct_wrapped_forward_proposal(self,
                                         x,
                                         _lambda,
                                         A_twist,
                                         b_twist,
                                         c_twist,
                                         propose_x_forward = True,
                                         compute_optimal_ds=False):
        """
        wrap all of the necessary functionality of the forward proposal given position at x (@ t-1)
        if x_forward is None: then a forward proposal is returned as the first argument, else it returns None;
            in either case,
        """
        from csmc import Theta_t, f_t, twisted_forward_proposal, twisted_forward_log_normalizer, uncontrolled_ULA_proposal_utility, compute_generalized_logw, twisted_t_log_weight

        theta = Theta_t(x, A_twist, self.dt)
        f = f_t(x, self.potential, _lambda, self.dt)

        if propose_x_forward: #then make a proposal
            x_forward, logp_forward = twisted_forward_proposal(theta, f, self.dt, b_twist)
        else:
            x_forward, logp_forward = None, None

        if compute_optimal_ds:
            raise Exception(f"not currently implemented")
        else:
            d = 0.

        log_forward_normalizer = twisted_forward_log_normalizer(theta, f, b_twist, c_twist, d, self.dt)

        return x_forward, log_forward_normalizer

    def conduct_twisted_gmm_proposal(self):
        """
        conduct a gaussian mixture model proposal that does the following
        1. compute the aggregated A,b,c (@ t=0) twisting potentials
        2. use the twisting potentials to twist the gaussian mixture components
        3. make a proposal and a logp of the proposal from twisted mixture components, twisted mean, and a twisted covariance matrix
        4. compute a twisted log initial weight

        arguments:

        returns:
            x : np.array(self.latent_dimension)
                proposed position
            logw0 : float
                incremental logw of the initial proposal (from a Gaussian Mixture Model)

        """
        from csmc import twisted_gmm_components, twisted_gmm_proposal, compute_twisted_gmm_lognormalizer, twisted_zeroth_log_weight
        A0, b0, c0 = self.aggregate_twists(t=0, x=None) #compute the t=0 twisting components
        unnormalized_twisted_log_alphas, twisted_Sigmas = twisted_gmm_components(self.gmm_mixing_components,
                                                                                 self.gmm_means,
                                                                                 self.gmm_covariance_matrices,
                                                                                 A0,
                                                                                 b0,
                                                                                 c0) #compute twisted gmm components
        x, logp_x = twisted_gmm_proposal(unnormalized_twisted_log_alphas,
                                         twisted_Sigmas,
                                         self.gmm_means,
                                         self.gmm_covariance_matrices,
                                         b0) #make a gmm proposal and its logp_x
        gmm_normalizer = compute_twisted_gmm_lognormalizer(unnormalized_twisted_log_alphas) #compute the normalization constant of the gmm
        log_w0 = twisted_zeroth_log_weight(x, A0, b0, c0, gmm_normalizer) #compute the log weight at time 0; there is an internal consistency check here
        return x, log_w0

    def scramble_twisting_parameters(self, naughts=False):
        """
        internal function to 'reset' twisting parameters for the sake of re-regression
        """
        model_list = [self.A_twisting_model, self.b_twisting_model, self.c_twisting_model] if not naughts else [self.A0_twisting_model, self.b0_twisting_model, self.c0_twisting_model]
        for model in model_list:
            if model is not None:
                for item in model._modules.values():
                    try:
                        item.reset_parameters()
                    except Exception as e:
                        pass
                        #print(f"scramble twisting parameters error: {e}")




    def aggregate_twists(self, t, x=None):
        """
        the twisting functions are defined s.t. \psi(i+1) = \psi(i) * \phi(i+1);
        in other words, i have to compute the summation of twisting functions for each iteration
        """
        if t != 0:
            assert x is not None

        A, b, c = np.zeros((self.latent_dimension, self.latent_dimension)), np.zeros(self.latent_dimension), 0.
        if self.twisting_iteration == 0: #then all of the twists are zero
            return A,b,c

        #if we are not at the zeroth twisting iteration, then we have to call the twisting function separately
        (A_model, b_model, c_model) = (self.A0_twisting_model, self.b0_twisting_model, self.c0_twisting_model) if t==0 else (self.A_twisting_model, self.b_twisting_model, self.c_twisting_model)
        for iteration in range(self.twisting_iteration):

            if A_model: set_model_parameters(A_model, self.A_parameters[iteration][t])
            if b_model: set_model_parameters(b_model, self.b_parameters[iteration][t])
            if c_model: set_model_parameters(c_model, self.c_parameters[iteration][t])

            if A_model is not None:
                A += A_model(None).numpy() if t==0 else A_model(x).numpy()
            if b_model is not None:
                b += b_model(None).numpy() if t==0 else b_model(x).numpy()
            if c_model is not None:
                c += c_model(None).numpy()

        return A,b,c


    def ADP(self,
            twisted_logws,
            xs,
            train_batch_size,
            train_epochs,
            compute_optimal_ds=False,
            validation_fraction = None,
            validation_batch_size = None,
            optimizer = torch.optim.SGD,
            optimizer_kwargs = {'lr': 1e-3}
           ):
        """
        conduct Approximate Dynamic Programming to learn an optimal sequence of value functions
        """
        import tqdm
        from torch_utils import render_dataloaders, check_models, loss_batch, fit

        #loggers
        sequential_training_losses = {}
        sequential_validation_losses = {} if validation_fraction is not None else None
        A_parameter_logger = [] if self.A_twisting_model is not None else None
        b_parameter_logger = [] if self.b_twisting_model is not None else None
        c_parameter_logger = [] if self.c_twisting_model is not None else None

        #check inputs
        if validation_fraction == 0.: validate_fraction = None
        if validation_fraction is not None:
            if validation_batch_size is None: validation_batch_size = 2*train_batch_size
        else:
            assert validation_batch_size is None, f"there is no validation fraction, so the validation batch size should be None"

        num_particles, parameter_sequence = twisted_logws.shape
        log_twisters = np.zeros(num_particles)
        loss_function = self.loss_function

        backward_iterator = range(len(self.parameter_sequence))[::-1]
        for t in tqdm.tqdm(backward_iterator):
            _logger.debug(f"\ttime iteration {t}")

            #set v_bar
            v_bar = torch.from_numpy((-twisted_logws[:,t] - log_twisters).astype('float32'))

            #break if this is the 0th iteration
            if t==0:
                models = [self.A0_twisting_model, self.b0_twisting_model, self.c0_twisting_model]
                input = torch.from_numpy(xs[:,0].astype('float32'))
                model_aggregate_function = ULAEnsembleCSMC.torchy_phi0
            else:
                models = [self.A_twisting_model, self.b_twisting_model, self.c_twisting_model]
                x_tm1s = torch.from_numpy(xs[:,t-1].astype('float32'))
                x_ts = torch.from_numpy(xs[:,t].astype('float32'))
                if compute_optimal_ds:
                    raise Exception(f"not currently implemented")
                else:
                    ds = torch.zeros(num_particles,1)
                input = torch.cat((x_tm1s, x_ts, ds), 1)
                model_aggregate_function = ULAEnsembleCSMC.torchy_phi

            train_dataloader, validation_dataloader = render_dataloaders(x = input,
                                                                         y = v_bar,
                                                                         validation_fraction = validation_fraction,
                                                                         train_batch_size = train_batch_size,
                                                                         validate_batch_size = validation_batch_size)
            self.scramble_twisting_parameters()
            training_losses, validation_losses = fit(epochs = train_epochs,
                                                     models = models,
                                                     loss_function = loss_function,
                                                     model_aggregate_function = model_aggregate_function,
                                                     optimizers = [optimizer(m.parameters(), **optimizer_kwargs) for m in models],
                                                     train_dataloader = train_dataloader,
                                                     validation_dataloader = validation_dataloader,
                                                     model_aggregate_function_kwargs = {'latent_dimension': self.latent_dimension})

            sequential_training_losses[t] = training_losses
            if sequential_validation_losses is not None: sequential_validation_losses[t] = validation_losses

            #now i have to recompute log twisting functions; fuck me
            if t != 0:
                As = self.A_twisting_model(torch.from_numpy(xs[:,t-1].astype('float32'))).detach().numpy() if self.A_twisting_model is not None else np.zeros(num_particles, self.latent_dimension, self.latent_dimension)
                bs = self.b_twisting_model(torch.from_numpy(xs[:,t-1].astype('float32'))).detach().numpy() if self.b_twisting_model is not None else np.zeros(num_particles, self.latent_dimension)
                cs = self.c_twisting_model(None).item() * np.ones(num_particles) if self.c_twisting_model is not None else np.zeros(num_particles)
                if compute_optimal_ds:
                    raise Exception(f"not currently implemented")
                else:
                    ds = np.zeros(num_particles)
            else:
                As = np.broadcast_to(self.A0_twisting_model(None).detach().numpy(), (num_particles, self.latent_dimension, self.latent_dimension)) if self.A0_twisting_model is not None else np.zeros(num_particles, self.latent_dimension, self.latent_dimension)
                bs = np.broadcast_to(self.b0_twisting_model(None).detach().numpy(), (num_particles, self.latent_dimension)) if self.b0_twisting_model is not None else np.zeros(num_particles, self.latent_dimension)
                cs = self.c0_twisting_model(None).item() * np.ones(num_particles) if self.c0_twisting_model is not None else np.zeros(num_particles)

                #for the next iteration
                log_twisters = np.array([self.conduct_wrapped_forward_proposal(x = xs[idx, t-1],
                                             _lambda = self.parameter_sequence[t],
                                             A_twist = As[idx],
                                             b_twist = bs[idx],
                                             c_twist = cs[idx],
                                             propose_x_forward = False,
                                             compute_optimal_ds=False) for idx in range(num_particles)])[:,1]

            #log the parameters...
            if A_parameter_logger is not None: A_parameter_logger.append([param.data for param in self.A_twisting_model.parameters()])
            if b_parameter_logger is not None: b_parameter_logger.append([param.data for param in self.b_twisting_model.parameters()])
            if c_parameter_logger is not None: c_parameter_logger.append([param.data for param in self.c_twisting_model.parameters()])

        if A_parameter_logger is not None: self.A_parameters.append(A_parameter_logger[::-1])
        if b_parameter_logger is not None: self.b_parameters.append(b_parameter_logger[::-1])
        if c_parameter_logger is not None: self.c_parameters.append(c_parameter_logger[::-1])

        self.twisting_iteration += 1



    @property
    def loss_function(self):
        return self._loss_function


    @staticmethod
    def set_model_parameters(model, setting_parameters):
        """
        set the parameters of a model
        """
        with torch.no_grad():
            for idx, param in enumerate(model.parameters()):
                param *=0.
                param += setting_parameters[idx]

    @staticmethod
    def torchy_phi0(models, input, latent_dimension):
        """
        model aggregation function where the first two arguments must be models, x_batch
        """
        from torch_utils import batch_quadratic, batch_dot
        bs, dim = list(input.size())
        assert dim == latent_dimension
        x0 = input
        A_model, b_model, c_model = models
        A, b, c = A_model(None).repeat(bs, 1, 1), b_model(None).repeat(bs, 1), c_model(None).repeat(bs)
        return batch_quadratic(x0, A) + batch_dot(x0, b) + c

    @staticmethod
    def torchy_phi(models, input, latent_dimension):
        """
        model aggregation function where the first two arguments must be models, x_batch
        """
        from torch_utils import batch_quadratic, batch_dot
        bs, es_dim = list(input.size())
        assert es_dim == 2*latent_dimension + 1
        x_tm1s = input[:,:latent_dimension]
        x_ts = input[:,latent_dimension:2*latent_dimension]
        ds = input[:,-1]
        A_model, b_model, c_model = models
        A = A_model(x_tm1s)
        b = b_model(x_tm1s)
        c = c_model(None).repeat(bs)
        out = batch_quadratic(x_ts, A) + batch_dot(x_ts, b) + c + ds
        return out
