import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

MAX_ATTEMPTS = 25

MIN_VALUE = np.log(np.power(np.finfo(np.float64).eps, 0.5))
MAX_VALUE = np.log(np.finfo(np.float64).max)


def get_chi2_threshold(num_parameters=1):
    return chi2.ppf(q=0.95, df=num_parameters) / 2


def rescale_log(log_array):
    """
    Rescales the log array if it gets too small/large, by multiplying it by a factor of e.
    :param log_array: numpy array containing the logs to be rescaled
    :return: float, factor of e by which the log array has been multiplied.
    """

    n_values = sum(log_array.shape)
    max_limit = MAX_VALUE - n_values - 1
    min_limit = MIN_VALUE + n_values + 1

    non_zero_loglh_array = log_array[log_array > -np.inf]
    if len(non_zero_loglh_array) == 0:
        return 0
    min_lh_value = np.min(non_zero_loglh_array)
    max_lh_value = np.max(non_zero_loglh_array)

    factors = 0
    if max_lh_value > max_limit:
        factors = max_limit - max_lh_value
    elif min_lh_value < min_limit:
        factors = min(min_limit - min_lh_value, max_limit - max_lh_value)
    log_array += factors
    return factors


def optimize_likelihood_params(forest, T, input_parameters, loglikelihood_function, bounds, start_parameters,
                               threads=1, num_attemps=3, optimise_as_logs=None, formatter=lambda _: _):
    """
    Optimizes the likelihood parameters for a given forest and a given MTBD model.


    :param forest: a list of ete3.Tree trees
    :return: tuple: (the values of optimized parameters, CIs)
    """
    #print("\nDEBUG: optimize_likelihood_params:")
    #print(f"  input_parameters = {input_parameters}")
    #print(f"  optimised_parameter_mask = {input_parameters == None}")

    optimised_parameter_mask = input_parameters == None
    if np.all(optimised_parameter_mask == False):
        return start_parameters, loglikelihood_function(forest, *start_parameters, T=T, threads=threads)

    bounds = bounds[optimised_parameter_mask]
    if optimise_as_logs is None:
        optimise_as_logs = [b[0] >= 0 and b[1] <= 1 for b in bounds]
    else:
        optimise_as_logs = optimise_as_logs[optimised_parameter_mask]
    optimised_bounds = np.array(bounds)
    optimised_bounds[optimise_as_logs] = np.log10(np.maximum(optimised_bounds[optimise_as_logs],
                                                             np.finfo(np.float64).eps))

    def get_real_params_from_optimised(ps):
        ps = np.maximum(np.minimum(ps, optimised_bounds[:, 1]), optimised_bounds[:, 0])
        ps[optimise_as_logs] = np.power(10, ps[optimise_as_logs])
        result = np.array(input_parameters)
        result[optimised_parameter_mask] = ps
        return result

    def get_optimised_params_from_real(ps):
        ps = np.array(ps[optimised_parameter_mask], dtype=np.float64)
        ps[optimise_as_logs] = np.log10(ps[optimise_as_logs])
        return ps

    def get_v(ps):
        if np.any(np.isnan(ps)):
            # print(f"{ps}\t-->\t-inf")
            return -np.inf
        ps_real = get_real_params_from_optimised(ps)
        res = loglikelihood_function(forest, *ps_real, T=T, threads=threads)
        # if np.isnan(res) or res == -np.inf:
        #     print(f"{formatter(ps_real)}\t-->\t{res}")
        return -res

    x0 = get_optimised_params_from_real(start_parameters)
    best_log_lh = -get_v(x0)

    successful_attempts = 0
    for i in range(max(num_attemps, MAX_ATTEMPTS)):
        if i == 0:
            vs = x0
        else:
            vs = np.random.uniform(optimised_bounds[:, 0], optimised_bounds[:, 1])
            if num_attemps > 1:
                print(f'Starting parameters: {formatter(get_real_params_from_optimised(vs))}')

        fres = minimize(get_v, x0=vs, method='L-BFGS-B', bounds=optimised_bounds)
        if fres.success and not np.any(np.isnan(fres.x)):
            successful_attempts += 1
            if -fres.fun >= best_log_lh:
                x0 = np.array(fres.x)
                best_log_lh = -fres.fun
                # break
            if num_attemps > 1:
                print(f'Attempt {i + 1} of trying to optimise the parameters:\t'
                      f'{formatter(get_real_params_from_optimised(fres.x))}\t->\t{-fres.fun}.')
        elif num_attemps > 1:
            print(f'Attempt {i + 1} of trying to optimise the parameters failed, due to {fres.message}.')
        if successful_attempts >= num_attemps:
            break
    if not successful_attempts:
        raise ValueError(f'Could not optimise the parameter values for this data, '
                         f'even after {max(num_attemps, MAX_ATTEMPTS)} attempts!')

    optimised_parameters = get_real_params_from_optimised(x0)

    return optimised_parameters, best_log_lh


def estimate_cis(T, forest, input_parameters, loglikelihood_function, optimised_parameters, bounds, threads=1,
                 optimise_as_logs=None, parameter_transformers=None):
    # Create optimised_cis array same as bounds
    optimised_cis = np.array(bounds)

    # Convert input_parameters to numpy array for consistent handling
    input_parameters = np.array(input_parameters)

    # Create a mask for fixed parameters (not None)
    fixed_parameter_mask = np.array([p is not None for p in input_parameters])

    # Set fixed parameter values in CI bounds
    for i in range(len(input_parameters)):
        if fixed_parameter_mask[i]:
            optimised_cis[i, 0] = input_parameters[i]
            optimised_cis[i, 1] = input_parameters[i]

    # Count number of optimized parameters
    n_optimized_params = np.sum(~fixed_parameter_mask)
    print(f'Estimating CIs for {n_optimized_params} free parameters...')

    # Calculate likelihood threshold for CI
    lk_threshold = (loglikelihood_function(forest, *optimised_parameters, T=T, threads=threads)
                    - get_chi2_threshold(num_parameters=n_optimized_params))

    def binary_search(v_min, v_max, get_lk, lower=True):
        v = v_min + (v_max - v_min) / 2
        if (v_max - v_min) < 1e-3:
            return v_min if lower else v_max

        lk_diff = get_lk(v) - lk_threshold
        if np.round(np.abs(lk_diff), 3) == 0:
            return v

        go_left = (lower and lk_diff > 0) or ((not lower) and lk_diff < 0)

        if go_left:
            return binary_search(v_min, v, get_lk, lower)
        return binary_search(v, v_max, get_lk, lower)

    def get_ci(args):
        (i, optimised_value) = args

        # Skip fixed parameters
        if fixed_parameter_mask[i]:
            return

        # Create copies for modification
        rps = np.array(optimised_parameters)
        ip = np.array(input_parameters)

        def get_lk(v):
            rps[i] = v
            ip[i] = v
            return \
            optimize_likelihood_params(forest, T=T, input_parameters=ip, loglikelihood_function=loglikelihood_function,
                                       bounds=optimised_cis, start_parameters=rps,
                                       threads=threads, num_attemps=1,
                                       optimise_as_logs=optimise_as_logs)[-1]

        # Calculate lower and upper bounds
        optimised_cis[i, 0] = binary_search(optimised_cis[i, 0], optimised_value, get_lk, True)
        optimised_cis[i, 1] = binary_search(optimised_value, optimised_cis[i, 1], get_lk, False)

    # Calculate CI for each parameter to be optimized
    for i, v in enumerate(optimised_parameters):
        if not fixed_parameter_mask[i]:
            get_ci((i, v))

    return optimised_cis


def AIC(k, loglk):
    return 2 * k - 2 * loglk
