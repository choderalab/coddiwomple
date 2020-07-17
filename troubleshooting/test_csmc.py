#!/usr/bin/env python

import numpy as np
"""
csmc tests
"""

"""
define some simple potentials for testing
"""
def default_potential(pos, parameter): #define the potential
    """
    default potential anneals between a 1d gaussian centered at 0 and a variance of 0.5 to a 1d gaussian centered at 0 and a variance of (still) 0.5;
    the free energy is identically 0.
    """
    return np.sum((1. - parameter)*pos**2 + parameter*(pos-1)**2)

def nondefault_potential(pos, parameter): #define a non default potential
    """
    nondefault potential anneals from a gaussian with mean=0, variance=1 to a gaussian with mean=0, variance=2;
    the free energy is identically log(sqrt(2)) \approx -0.34
    """
    return np.sum((1. - parameter)*(0.5*pos**2) + parameter*(0.25*pos**2))

"""
testing numpy utilities
"""

def test_mahalanobis():
    """
    this is a little test function to make sure i am using scipy special distance mahalanobis properly
    """
    from scipy import random, linalg
    matrixSize = 10
    A = random.rand(matrixSize,matrixSize)
    B = np.dot(A,A.transpose())

    v = np.random.randn(matrixSize)
    u = np.random.randn(matrixSize)

    #manual mahalanobis
    manual_result = np.sqrt((u-v).dot(np.matmul(np.linalg.inv(B), u-v)))

    #automatic mahalanobis
    from scipy.spatial.distance import mahalanobis
    auto_result = mahalanobis(u, v, np.linalg.inv(B))

    assert np.isclose(manual_result, auto_result), f"auto_result: {auto_result}; manual_result: {manual_result}"

def test_brownian_motion(potential=None, potential_parameters=None, x=None, number_iterations = 10000, dt = 1e-2, return_energy=False, return_trajectory=False):
    """
    run a test of brownian dynamics with a potential and a given potential parameter and an initial position

    arguments
        potential : function, default None
            potential function that takes x, potential_parameters; if None, uses x**2
        potential_parameters : argument
            potential parameters; if None or potential is None, set to 0.
        x : np.array(N)
            default 0.
        number_iterations : int, default 1000
            number of iterations to run
        dt : float, default 1e-2
            time increment
        return_energy : bool, default False
            whether to return the energy of the trajectory
        return_trajectory : bool, default False
            whether to return the trajectory


    returns
        tup : tuple
            (energy_array, trajectory_array)
    """
    import tqdm

    if not potential:
        variance = 0.5
        validate = True
        def potential(x, parameters):
            return np.sum((1. - parameters)*x**2 + parameters*(x-1.)**2)
        potential_parameters = 0.
    else:
        assert potential_parameters is not None

    if x is None:
        from scipy.stats import multivariate_normal
        x = np.array([multivariate_normal.rvs(mean=0., cov=0.5)])

    trajectory = [x]
    energies = [potential(x, potential_parameters)]

    for _ in tqdm.trange(number_iterations):
        mu, Sigma = EL_mu_sigma(x, potential, dt, potential_parameters, is_force_function=False)
        x = multivariate_normal.rvs(mean=mu, cov = Sigma)
        if type(x) != np.ndarray:
            x = np.array([x])
        trajectory.append(x)
        energies.append(potential(x, potential_parameters))

    if return_energy:
        arg_1 = np.array(energies)
    else:
        arg_1 = None

    if return_trajectory:
        arg_2 = np.array(trajectory)
    else:
        arg_2 = None

    if validate: #if the potential is not specified, then we validate that the trajectory position is within the 3*std_dev of mean (0.)
        upper_bound, lower_bound = 3*np.sqrt(variance), -3*np.sqrt(variance)
        mean_trajs = np.mean(np.array(trajectory))
        assert mean_trajs > lower_bound
        assert mean_trajs < upper_bound

    return (arg_1, arg_2)

def test_SIS_logw_computations():
    """
    there are at present 3 ways to compute the logw of an ULA SIS proposal...
    1. manual calculation
    2. compute generalized logw
    3. compute ULA logw
    i will do a random annealing protocol and assert that the logws are consistent
    """
    from scipy.stats import multivariate_normal
    from test_csmc import default_potential
    import tqdm

    start_mean = 0.
    start_cov = 0.5
    num_samples = 100
    dt = np.random.rand()
    potential = default_potential

    for iteration in tqdm.trange(num_samples):
        param_old, param_new = np.random.rand(), np.random.rand()
        x = np.array([multivariate_normal.rvs(mean = start_mean, cov = start_cov)])



        #make a forward proposal
        x_forward, logp_forward, logp_backward = uncontrolled_ULA_proposal_utility(x, potential, dt, param_new)

        log_gammatm1_xtm1 = -potential(x, param_old)
        log_gammat_xt = -potential(x_forward, param_new)

        logw = log_gammat_xt + logp_backward - log_gammatm1_xtm1 - logp_forward
        logw_internal = compute_generalized_logw(log_gamma_old = log_gammatm1_xtm1,
                                            log_gamma_new = log_gammat_xt,
                                            log_forward_kernelt = logp_forward,
                                            log_backward_kernel_tm1 = logp_backward)
        logw_ULA_compute = compute_ULA_logw(x_tm1 = x,
                                                       x_t = x_forward,
                                                        forcet_xtm1 = compute_force(x, potential, param_new),
                                                        forcet_xt = compute_force(x_forward, potential, param_new),
                                                        potentialtm1_xtm1 = -log_gammatm1_xtm1,
                                                        potentialt_xt = -log_gammat_xt,
                                                        dt = dt)
        assert np.isclose(logw, logw_internal)
        assert np.isclose(logw, logw_ULA_compute)

def test_gmm_ensemble_SIS():
    """
    do a simple 1d sequential importance sampling run, both with and without metropolization;
    conduct a separate validation of twisted smc in the uncontrolled regime (i.e. the twisting potential is 0.)
    as a double check, i implement a gaussian mixture model with two components (with the same gaussian parameters)
    """
    from pymbar.exp import EXP
    import tqdm
    from csmc import gmm_ensemble_SIS, twisted_smc, ULAEnsembleCSMC

    tol = 1e-1
    given_free_energy = -.34
    num_particles=1000
    potential = nondefault_potential
    lambda_sequence = np.linspace(0,1,10)
    dt=1.
    mixture_components = np.array([0.5, 0.5])
    mixture_mus = np.array([[0.], [0.]])
    mixture_cov_matrices = np.array([ [[1.]], [[1.]] ])

    SIS_traj, SIS_logw, _ = gmm_ensemble_SIS(num_particles,
                                         potential = potential,
                                         lambda_sequence = lambda_sequence,
                                         dt=dt,
                                         mix_components = mixture_components,
                                         mus = mixture_mus,
                                         Sigmas = mixture_cov_matrices,
                                         metropolize = False)
    AIS_traj, AIS_logw, _ = gmm_ensemble_SIS(num_particles,
                                         potential = potential,
                                         lambda_sequence = lambda_sequence,
                                         dt=dt,
                                         mix_components = mixture_components,
                                         mus = mixture_mus,
                                         Sigmas = mixture_cov_matrices,
                                         metropolize = True)


    csmc = ULAEnsembleCSMC(potential = nondefault_potential,
                                  parameter_sequence = lambda_sequence,
                                  dt = dt,
                                  gmm_mixing_components = mixture_components,
                                  gmm_means = mixture_mus,
                                  gmm_covariance_matrices = mixture_cov_matrices)
    uncontrolled_logws, uncontrolled_trajectory = csmc.run_twisted_smc(num_particles = num_particles, compute_optimal_ds = False, save_trajectory=True)


    #compute free energies
    AIS_cumulative_weights, SIS_cumulative_weights, uncontrolled_smc_cumulative_weights = (np.cumsum(AIS_logw, 1),
                                                                                           np.cumsum(SIS_logw, 1),
                                                                                           np.cumsum(np.array(uncontrolled_logws), 1))

    AIS_works, SIS_works, uncontrolled_smc_works = -AIS_cumulative_weights[:,-1], -SIS_cumulative_weights[:,-1], -uncontrolled_smc_cumulative_weights[:,-1]
    AIS_free_energy, AIS_uncertainty = EXP(AIS_works)
    SIS_free_energy, SIS_uncertainty = EXP(SIS_works)
    uncontrolled_smc_free_energy, uncontrolled_smc_uncertainty = EXP(uncontrolled_smc_works)
    print(f"AIS : F = {AIS_free_energy}; dF = {AIS_uncertainty}")
    print(f"SIS : F = {SIS_free_energy}; dF = {SIS_uncertainty}")
    print(f"uncontrolled smc: F = {uncontrolled_smc_free_energy}; dF = {uncontrolled_smc_uncertainty}")

    assert abs(AIS_free_energy - given_free_energy) < tol
    assert abs(SIS_free_energy - given_free_energy) < tol
    assert abs(uncontrolled_smc_free_energy - given_free_energy) < tol



######
#Test CSMC Class methods
######
