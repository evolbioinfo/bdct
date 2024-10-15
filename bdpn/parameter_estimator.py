import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

LIKELIHOOD_DIFF_95_CI = chi2.ppf(q=0.95, df=1) / 2

MIN_VALUE = np.log(np.finfo(np.float64).eps)
MAX_VALUE = np.log(np.finfo(np.float64).max)


def rescale_log(log_array):
    """
    Rescales the log array if it gets too small/large, by multiplying it by a factor of e.
    :param log_array: numpy array containing the logs to be rescaled
    :return: float, factor of e by which the log array has been multiplied.
    """

    max_limit = MAX_VALUE
    min_limit = MIN_VALUE

    non_zero_loglh_array = log_array[log_array > -np.inf]
    if len(non_zero_loglh_array) == 0:
        return 0
    min_lh_value = np.min(non_zero_loglh_array)
    max_lh_value = np.max(non_zero_loglh_array)

    factors = 0
    if max_lh_value > max_limit - 2:
        factors = max_limit - max_lh_value - 2
    elif min_lh_value < min_limit + 2:
        factors = min(min_limit - min_lh_value + 2, max_limit - max_lh_value - 2)
    log_array += factors
    return factors


def optimize_likelihood_params(forest, T, input_parameters, loglikelihood_function, bounds, start_parameters,
                               threads=1, num_attemps=5, optimise_as_logs=None, formatter=lambda _: _):
    """
    Optimizes the likelihood parameters for a given forest and a given MTBD model.


    :param forest: a list of ete3.Tree trees
    :return: tuple: (the values of optimized parameters, CIs)
    """
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
        ps_real = get_real_params_from_optimised(ps)
        res = loglikelihood_function(forest, *ps_real, T=T, threads=threads)
        # if np.any(np.isnan(ps)) or np.isnan(res) or res == -np.inf:
        #     print("{}\t-->\t{:.10f}".format(formatter(ps_real), res))
        return -res

    x0 = get_optimised_params_from_real(start_parameters)
    best_log_lh = -get_v(x0)

    for i in range(num_attemps):
        if i == 0:
            vs = x0
        else:
            vs = np.random.uniform(optimised_bounds[:, 0], optimised_bounds[:, 1])
            print('Starting parameters: {}'.format(formatter(get_real_params_from_optimised(vs))))

        fres = minimize(get_v, x0=vs, method='L-BFGS-B', bounds=optimised_bounds)
        if fres.success and not np.any(np.isnan(fres.x)):
            if -fres.fun >= best_log_lh:
                x0 = np.array(fres.x)
                best_log_lh = -fres.fun
                # break
            if num_attemps > 1:
                print('Attempt {} of trying to optimise the parameters: {} -> {}.'
                      .format(i + 1, formatter(get_real_params_from_optimised(fres.x)), -fres.fun))
        elif num_attemps > 1:
            print('Attempt {} of trying to optimise the parameters failed.'.format(i + 1))
    optimised_parameters = get_real_params_from_optimised(x0)

    return optimised_parameters, best_log_lh


def estimate_cis(T, forest, input_parameters, loglikelihood_function, optimised_parameters, bounds, threads=1):
    print('Estimating CIs...')
    optimised_cis = np.array(bounds)
    fixed_parameter_mask = input_parameters != None
    optimised_cis[fixed_parameter_mask, 0] = input_parameters[fixed_parameter_mask]
    optimised_cis[fixed_parameter_mask, 1] = input_parameters[fixed_parameter_mask]
    lk_threshold = loglikelihood_function(forest, *optimised_parameters, T, threads=threads) - LIKELIHOOD_DIFF_95_CI

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
        # print('---------')
        # print(i, optimised_value, (b_min, b_max))

        rps = np.array(optimised_parameters)
        ip = np.array(input_parameters)

        def get_lk(v):
            rps[i] = v
            ip[i] = v
            return optimize_likelihood_params(forest, T, ip, loglikelihood_function,
                                              bounds=optimised_cis, start_parameters=rps,
                                              threads=threads, num_attemps=1)[-1]

        optimised_cis[i, 0] = binary_search(optimised_cis[i, 0], optimised_value, get_lk, True)
        optimised_cis[i, 1] = binary_search(optimised_value, optimised_cis[i, 1], get_lk, False)
        # print(i, optimised_cis[i, :])
        # print('---------')

    # if threads > 1:
    #     with ThreadPool(processes=min(threads, len(bounds))) as pool:
    #         pool.map(func=get_ci,
    #                  iterable=zip(((i, v) for (i, v) in enumerate(optimised_parameters)
    #                                if input_parameters[i] is not None), bounds), chunksize=1)
    # else:
    for _ in ((i, v) for (i, v) in enumerate(optimised_parameters) if input_parameters[i] is None):
        get_ci(_)
    return optimised_cis


def AIC(k, loglk):
    return 2 * k - 2 * loglk
