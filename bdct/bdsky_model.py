import os

import numpy as np
from bdct import bd_model

from bdct.bd_model import DEFAULT_LOWER_BOUNDS, DEFAULT_UPPER_BOUNDS, PARAMETER_NAMES, EPI_PARAMETER_NAMES, \
    REPRODUCTIVE_NUMBER, INFECTIOUS_TIME, SAMPLING_PROBABILITY, TRANSMISSION_RATE, REMOVAL_RATE, get_start_parameters
from bdct.formulas import get_c1, get_c2, get_E, get_log_p, get_u, log_factorial, EPSILON
from bdct.parameter_estimator import optimize_likelihood_params, estimate_cis
from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time, get_T

GRID_SIZE = 11

INTERVAL_END_TIME = 'time at interval end'
INTERVAL = 'interval'


def loglikelihood(forest, *params, T, threads=1, u=-1):
    n_intervals = len(params) // 3

    la_array = params[0:n_intervals * 3:3]
    psi_array = params[1:n_intervals * 3:3]
    rho_array = params[2:n_intervals * 3:3]

    skyline_times = optimized_values2time_intervals(params[n_intervals * 3:], T)


    c1_array = np.zeros(n_intervals)
    for i in range(n_intervals):
        c1_array[i] = get_c1(la=la_array[i], psi=psi_array[i], rho=rho_array[i])

    ci_array = np.ones(n_intervals)
    c2_array = np.zeros(n_intervals)
    for i in range(n_intervals - 1, -1, -1):
        if i < n_intervals - 1: # otherwise it will be 1 by default
            ci_array[i] = get_u(la_array[i + 1], psi_array[i + 1], c1_array[i + 1],
                                get_E(c1_array[i + 1], c2_array[i + 1], skyline_times[i], skyline_times[i + 1]))
        c2_array[i] = get_c2(la_array[i], psi_array[i], c1_array[i], ci_array[i])


    log_psi_rho_array = np.log(psi_array) + np.log(rho_array)
    log_la_array = np.log(la_array)

    hidden_lk = get_u(la_array[0], psi_array[0], c1_array[0],
                      E_t=get_E(c1=c1_array[0], c2=c2_array[0], t=0, T=skyline_times[0]))
    if hidden_lk:
        u = len(forest) * hidden_lk / (1 - hidden_lk) if u is None or u < 0 else u
        res = u * np.log(hidden_lk)
    else:
        res = 0

    for tree in forest:
        interval = 0
        t_start = getattr(tree, TIME) - tree.dist
        if n_intervals:
            while interval < (n_intervals - 1) and t_start > skyline_times[interval]:
                interval += 1
        todo = [(tree, interval, t_start)]
        while todo:
            n, interval, t = todo.pop()
            ti = getattr(n, TIME)
            Ti = skyline_times[interval]
            c1_i = c1_array[interval]
            c2_i = c2_array[interval]
            if ti > Ti:
                res += get_log_p(c1_i, t, ti=Ti,
                                 E_t=get_E(c1=c1_i, c2=c2_i, t=t, T=Ti),
                                 E_ti=get_E(c1=c1_i, c2=c2_i, t=Ti, T=Ti))
                todo.append((n, interval + 1, Ti))
            else:
                res += get_log_p(c1_i, t, ti=ti,
                                 E_t=get_E(c1=c1_i, c2=c2_i, t=t, T=Ti),
                                 E_ti=get_E(c1=c1_i, c2=c2_i, t=ti, T=Ti))

                if n.is_leaf():
                    res += log_psi_rho_array[interval]
                else:
                    num_children = len(n.children)
                    res += log_factorial(num_children) + (num_children - 1) * log_la_array[interval]
                    for child in n.children:
                        todo.append((child, interval, ti))

    return res


def get_time_interval_start(n_intervals):
    """
    Create equal intervals represented as proportions of time left till the end
    """
    fractions = np.ones(n_intervals - 1)
    total = n_intervals
    for i in range(n_intervals - 1):
        fractions[i] /= total
        total -= 1
    return fractions


def get_time_interval_bounds():
    return 0.01, 0.99

def time_intervals2optimized_values(skyline_times):
    """
    Converts skyline times [t_1, ..., t_{n-1}, T] to fractions f_i used for optimization,
    where f_1 is t_1 as the fraction of total time (between 0 and T): t_1 / T,
    f_2 is t_2 as the fraction of time between t_1 and T: (t2 - t1) / (T - t1), etc.

    :param skyline_times: np.array containing n skyline change times [t_1, ..., t_{n-1}, T],
        the last time being the end of the sampling period T
    :return: np.array containing n - 1 fractions f_1, f_{n-1} described above.
    """

    fs = np.zeros(len(skyline_times) - 1)
    start_t = 0
    for (idx, st) in enumerate(skyline_times[:-1]):
        # this time as proportion of the time left till the tree end
        fs[idx] = (st - start_t) / (skyline_times[-1] - start_t)
        start_t = st
    return fs

def optimized_values2time_intervals(fs, T):
    """
    Converts values f_i used for optimization to skyline times [t_1, ..., t_{n-1}, T],
    where f_1 is t_1 as the fraction of total time (between 0 and T): t_1 / T,
    f_2 is t_2 as the fraction of time between t_1 and T: (t2 - t1) / (T - t1), etc.

    :param fs: np.array containing f_i described above.
    :return: np.array containing n skyline change times [t_1, ..., t_{n-1}, T],
        the last time being the end of the sampling period T
    """
    skyline_times = np.zeros(len(fs) + 1)
    skyline_times[-1] = T
    t_start = 0
    for i, fraction in enumerate(fs):
        if fraction is not None and t_start is not None:
            skyline_times[i] = t_start + (T - t_start) * fraction
        else:
            skyline_times[i] = None
        t_start = skyline_times[i]
    return skyline_times


def infer(forest, T, la=None, psi=None, p=None, skyline_times=None,
          lower_bounds=DEFAULT_LOWER_BOUNDS, upper_bounds=DEFAULT_UPPER_BOUNDS, ci=False, threads=1, num_attemps=3, **kwargs):
    """
    Infers BD model parameters from a given forest.

    :param forest: list of one or more trees
    :param la: transmission rates (one per skyline interval)
    :param psi: removal rates (one per skyline interval)
    :param p: sampling probabilities  (one per skyline interval)
    :param skyline_times: skyline interval change times
    :param lower_bounds: array of lower bounds for parameter values (la, psi, p)
    :param upper_bounds: array of upper bounds for parameter values (la, psi, p)
    :param ci: whether to calculate the CIs or not
    :return: tuple(vs, cis) of estimated parameter values vs=[la1, psi1, p1, la2, psi2, p2, ...]
        and CIs ci=[[la1_min, la1_max], [psi1_min, psi1_max], [p1_min, p1_max], ...].
        In the case when CIs were not set to be calculated,
        their values would correspond exactly to the parameter values.
    """
    n_la, n_psi, n_p = 0, 0, 0
    if isinstance(la, list) or isinstance(la, np.ndarray):
        n_la = len(la)
    elif la is not None:
        la = [la]
        n_la = 1
    if isinstance(psi, list) or isinstance(psi, np.ndarray):
        n_psi = len(psi)
    elif psi is not None:
        psi = [psi]
        n_psi = 1
    if isinstance(p, list) or isinstance(p, np.ndarray):
        n_p = len(p)
    elif p is not None:
        p = [p]
        n_p = 1

    n_intervals = max(n_la, n_psi, n_p)
    if not n_intervals:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')
    if n_la > 0 and n_la != n_intervals or n_psi > 0 and n_psi != n_intervals or n_p > 0 and n_p != n_intervals:
            raise ValueError(f'Either all or no parameter values of each type should be fixed, '
                             f'however the numbers of given values for lambda is {n_la}, for psi is {n_psi} and for p is {n_p}.')

    n_t = 0
    if isinstance(skyline_times, list) or isinstance(skyline_times, np.ndarray):
        n_t = len(skyline_times)
        skyline_times = np.concatenate((skyline_times, [T]))
    elif skyline_times is not None:
        skyline_times = [skyline_times, T]
        n_t = 1
    else:
        skyline_times = [T]

    if n_t and n_t != n_intervals - 1:
        raise ValueError(f'The skyline times should specify times at which the model changes, '
                         f'however {n_intervals} are specified via fixed parameters, but {n_t} times of change '
                         f'(instead of expected {n_intervals - 1}).')

    if n_t and any(skyline_times) <= 0:
        raise ValueError(f'The skyline times should specify times at which the model changes, '
                         f'and hence can not be negative.')

    for (t1, t2) in zip(skyline_times[:-1], skyline_times[1:]):
        if t1 >= t2:
            if t2 == T:
                raise ValueError(
                    f'The specified skyline change time {t1} is outside of the given forest sampling period 0-{T}.')
            raise ValueError(f'The skyline times should specify times at which the model changes and should be sorted, '
                             f'while you specified {t2} after {t1}.')

    n_parameters = 3 * n_intervals + (n_intervals - 1)
    bounds = np.zeros((n_parameters, 2), dtype=np.float64)
    lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
    if not np.all(upper_bounds >= lower_bounds):
        raise ValueError('Lower bounds cannot be greater than upper bounds')
    if np.any(lower_bounds < 0):
        raise ValueError('Bounds must be non-negative')
    if upper_bounds[-1] > 1:
        raise ValueError('Probability bounds must be between 0 and 1')

    for start in range(n_intervals):
        bounds[start * 3: start * 3 + 3, 0] = lower_bounds
        bounds[start * 3: start * 3 + 3:, 1] = upper_bounds
    bounds[n_intervals * 3:, :] = get_time_interval_bounds()

    start_parameters = np.zeros(n_parameters)
    input_params = np.array([None] * n_parameters)
    for start in range(n_intervals):
        la_i = la[start] if n_la else None
        psi_i = psi[start] if n_psi else None
        p_i = p[start] if n_p else None
        input_params[start * 3: start * 3 + 3] = np.array([la_i, psi_i, p_i])
        if n_intervals > 1:
            vs, _ = bd_model.infer(forest, T=T, la=la_i, psi=psi_i, p=p_i,
                                   lower_bounds=bounds[start * 3: start * 3 + 3, 0],
                                   upper_bounds=bounds[start * 3: start * 3 + 3, 1], ci=False,
                                   num_attemps=1)
            start_parameters[start * 3: start * 3 + 3] = vs
        else:
            start_parameters[start * 3: start * 3 + 3] = get_start_parameters(forest, la_i, psi_i, p_i)
    if n_intervals > 1:
        if n_t:
            start_parameters[n_intervals * 3: ] = time_intervals2optimized_values(skyline_times)
            input_params[n_intervals * 3: ] = start_parameters[n_intervals * 3: ]
        else:
            start_parameters[n_intervals * 3:] = get_time_interval_start(n_intervals)

    best_vs, best_lk = np.array(start_parameters), loglikelihood(forest, *start_parameters, T=T, threads=threads)

    print(f'Lower bounds are set to:\t{format_parameters(*lower_bounds, epi=False, T=T)}')
    print(f'Upper bounds are set to:\t{format_parameters(*upper_bounds, epi=False, T=T)}\n')
    print(f'Starting parameters:\t{format_parameters(*start_parameters, fixed=input_params, T=T)}\tloglikelihood={best_lk}')


    vs, lk = optimize_current_setting(bounds, n_intervals, start_parameters, input_params, forest, T,
                                      n_times_to_optimize=n_intervals - 1 - n_t, threads=1)

    print(f'Estimated BDSKY parameters:\t{format_parameters(*vs, T=T)};\tloglikelihood={lk}')

    if lk > best_lk:
        best_lk = lk
        best_vs = vs
    if ci:
        cis = estimate_cis(T, forest, input_parameters=input_params, loglikelihood_function=loglikelihood,
                           optimised_parameters=best_vs, bounds=bounds, threads=threads)
        print(f'Estimated CIs:\n\tlower:\t{format_parameters(*cis[:, 0], epi=False, T=T)}\n'
              f'\tupper:\t{format_parameters(*cis[:, 1], epi=False, T=T)}')
    else:
        cis = None
    return best_vs, cis


def optimize_current_setting(bounds, n_intervals, start_parameters, input_params, forest, T, n_times_to_optimize=0, threads=1):
    """
    The idea is to optimize with skyline times fixed on the grid first,
    then constraint the time bounds around the best grid value and optimize everything one more time.
    """
    if not n_times_to_optimize:
        return optimize_likelihood_params(forest, T=T, input_parameters=input_params,
                                            loglikelihood_function=loglikelihood, bounds=bounds,
                                            start_parameters=start_parameters, threads=threads,
                                            formatter=lambda _: format_parameters(*_, T=T),
                                            num_attemps=1)
    else:
        bs = np.array(bounds)
        i = len(input_params) - n_times_to_optimize
        lb, up = bs[i]
        step = (up - lb) / GRID_SIZE
        grid_values = np.arange(lb + step, up, step=step)
        best_interval_idx = None
        best_lk = loglikelihood(forest, *start_parameters, T=T, threads=threads)
        best_vs = start_parameters
        for idx, fixed_value in enumerate(grid_values):
            sp = np.array(start_parameters)
            sp[i] = fixed_value
            ip = np.array(input_params)
            ip[i] = fixed_value
            vs, lk = optimize_current_setting(bs, n_intervals, sp, ip, forest, T, n_times_to_optimize - 1, threads)
            # ts = optimized_values2time_intervals(ip[n_intervals * 3:], T)[:-1]
            # ts = ', '.join(f'{ts[_]:g}' if not np.isnan(ts[_]) else f'[{bs[n_intervals*3 + _, 0]:g}-{bs[n_intervals*3 + _, 1]:g}]' if ip[n_intervals*3 + _] is None else f'{ip[n_intervals*3 + _] * 100:g} %' for _ in range(len(ts)))
            # print(f'Best loglk for times {ts} is {lk}')
            if lk > best_lk:
                best_lk = lk
                best_vs = vs
                best_interval_idx = idx
        if best_interval_idx is not None:
            bs[i]  = (grid_values[best_interval_idx - 1] if best_interval_idx > 0 else lb,
                      grid_values[best_interval_idx + 1] if best_interval_idx < len(grid_values) - 1 else up)

        sp = np.array(best_vs)
        vs, lk = optimize_current_setting(bs, n_intervals, sp, input_params, forest, T, n_times_to_optimize - 1, threads)

        # ts = optimized_values2time_intervals(input_params[n_intervals * 3:], T)[:-1]
        # ts = ', '.join(f'{ts[i]:g}' if not np.isnan(
        #     ts[i]) else f'[{bs[n_intervals * 3 + i, 0]:g}-{bs[n_intervals * 3 + i, 1]:g}]' for i in range(len(ts)))
        # print(f'Best loglk for times {ts} is {lk}')
        if lk > best_lk:
            best_lk = lk
            best_vs = vs
        return best_vs, best_lk




def save_results(vs, cis, T, log, ci=False):
    n_intervals = len(vs) // 3

    la_array = vs[0:n_intervals * 3:3]
    psi_array = vs[1:n_intervals * 3:3]
    rho_array = vs[2:n_intervals * 3:3]

    if ci:
        la_ci_array = cis[0:n_intervals * 3:3]
        psi_ci_array = cis[1:n_intervals * 3:3]
        rho_ci_array = cis[2:n_intervals * 3:3]

    # the skyline times are specified as:
    # the first one t1 as a fraction of total time (between 0 and T): t1 / T,
    # the second one t2 as a fraction of time between t1 and T: (t2 - t1) / (T - t1), etc.
    skyline_time_fractions = vs[n_intervals * 3:]
    skyline_times = optimized_values2time_intervals(skyline_time_fractions, T)

    if ci:
        skyline_time_fraction_cis = cis[n_intervals * 3:, :]
        skyline_time_cis = np.array([optimized_values2time_intervals(skyline_time_fraction_cis[:, 0], T),
                                     optimized_values2time_intervals(skyline_time_fraction_cis[:, 1], T)]).T


    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, 'w+') as f:
        label_line = \
            ','.join([INTERVAL, 'type', REPRODUCTIVE_NUMBER, INFECTIOUS_TIME, SAMPLING_PROBABILITY, TRANSMISSION_RATE, REMOVAL_RATE, INTERVAL_END_TIME])
        f.write(f"{label_line}\n")

        for i in range(n_intervals):
            la, psi, rho = la_array[i], psi_array[i], rho_array[i]
            R0 = la / psi
            rt = 1 / psi
            t = skyline_times[i]
            value_line = ",".join(f'{_:g}' for _ in [R0, rt, rho, la, psi, t])
            f.write(f"{i},value,{value_line}\n")
            if ci:
                (la_min, la_max), (psi_min, psi_max), (rho_min, rho_max) = la_ci_array[i, :], psi_ci_array[i, :], rho_ci_array[i, :]
                t_min, t_max = skyline_time_cis[i, :]
                R0_min, R0_max = la_min / psi, la_max / psi
                rt_min, rt_max = 1 / psi_max, 1 / psi_min
                ci_min_line = ",".join(f'{_:g}' for _ in [R0_min, rt_min, rho_min, la_min, psi_min, t_min])
                f.write(f"{i},CI_min,{ci_min_line}\n")
                ci_max_line = ",".join(f'{_:g}' for _ in [R0_max, rt_max, rho_max, la_max, psi_max, t_max])
                f.write(f"{i},CI_max,{ci_max_line}\n")


def format_parameters(*params, T, fixed=None, epi=True):
    n_intervals = len(params) // 3

    la_array = params[0:n_intervals * 3:3]
    psi_array = params[1:n_intervals * 3:3]
    rho_array = params[2:n_intervals * 3:3]

    skyline_times = optimized_values2time_intervals(params[n_intervals * 3:], T)

    epi=False

    names = np.concatenate([PARAMETER_NAMES, EPI_PARAMETER_NAMES]) if epi else PARAMETER_NAMES

    res = ''
    for i in range(n_intervals):
        suffix = f'_{i}' if n_intervals > 1 else ''

        params = [la_array[i], psi_array[i], rho_array[i], la_array[i] / psi_array[i], 1 / psi_array[i]] \
            if epi else [la_array[i], psi_array[i], rho_array[i]]
        if fixed is None:
            res += ', '.join('{}={:.6f}'.format(*_) for _ in zip((f'{n}{suffix}' for n in names), params))
        else:
            if epi:
                fixed = np.concatenate([fixed, [fixed[0] and fixed[1], fixed[1], fixed[2]]])
            res += ', '.join('{}={:.6f}{}'.format(_[0], _[1], '' if _[2] is None else ' (fixed)')
                             for _ in zip((f'{n}{suffix}' for n in names), params, fixed))
        res += '; '

    for i in range(n_intervals - 1):
        if i > 0:
            res += ', '
        res += f'T_{i}={skyline_times[i]}'
        if fixed is not None and fixed[n_intervals * 3 + i]:
            res += ' (fixed)'
    return res

def main():
    """
    Entry point for tree parameter estimation with the BDSKY model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimate BDSKY parameters.")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")

    parser.add_argument('--la', nargs='*', default=None, type=float,
                        help="List of transmission rates (one per skyline interval).")
    parser.add_argument('--psi', nargs='*', default=None, type=float,
                        help="List of removal rates (one per skyline interval).")
    parser.add_argument('--p', nargs='*', default=None, type=float,
                        help="List of sampling probabilities (one per skyline interval).")
    parser.add_argument('--skyline_times', nargs='*', default=None, type=float,
                        help="List of time points specifying when to switch from model i to model i+1 in the Skyline."
                             "Must be sorted in ascending order and contain one less elements "
                             "than the number of models in the Skyline."
                             "The first model always starts at time 0.")

    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=3,
                        help="upper bounds for parameters (la, psi, p)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=3,
                        help="lower bounds for parameters (la, psi, p)", default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    forest = read_forest(params.nwk)
    # resolve_forest(forest)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)
    print('Read a forest of {} trees with {} tips in total, evolving over time {}'
          .format(len(forest), sum(len(_) for _ in forest), T))

    vs, cis = infer(forest, T=T, **vars(params))
    save_results(vs, cis, T, params.log, ci=params.ci)


def loglikelihood_main():
    """
    Entry point for tree likelihood estimation with the BD model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Calculate BD likelihood on a given forest for given parameter values.")

    parser.add_argument('--la', nargs='+', type=float,
                        help="List of transmission rates (one per skyline interval).")
    parser.add_argument('--psi', nargs='+', type=float,
                        help="List of removal rates (one per skyline interval).")
    parser.add_argument('--p', nargs='+', type=float,
                        help="List of sampling probabilities (one per skyline interval).")
    parser.add_argument('--skyline_times', nargs='*', type=float,
                        help="List of time points specifying when to switch from model i to model i+1 in the Skyline."
                             "Must be sorted in ascending order and contain one less elements "
                             "than the number of models in the Skyline."
                             "The first model always starts at time 0.")

    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--u', required=False, type=int, default=-1,
                        help="number of hidden trees (estimated by default)")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    # resolve_forest(forest)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)

    n_la, n_psi, n_p = len(params.la), len(params.psi), len(params.p)

    n_intervals = n_la

    if n_la != n_intervals or n_psi != n_intervals or n_p != n_intervals:
        raise ValueError(f'All the parameter values should cover the same number of skyline intervals, '
                         f'however the numbers of given values for lambda is {n_la}, for psi is {n_psi} and for p is {n_p}.')

    n_t = len(params.skyline_times)

    if n_t != n_intervals - 1:
        raise ValueError(f'The skyline times should specify times at which the model changes, '
                         f'however model parameters are specified for {n_intervals}, but {n_t} times of change are given '
                         f'(instead of expected {n_intervals - 1}).')

    if any(params.skyline_times) <= 0:
        raise ValueError(f'The skyline times should specify times at which the model changes, '
                         f'and hence can not be negative.')

    for (t1, t2) in zip(params.skyline_times[:-1], params.skyline_times[1:]):
        if t1 >= t2:
            raise ValueError(f'The skyline times should specify times at which the model changes and should be sorted, '
                             f'while you specified {t2} after {t1}.')

    ps = np.zeros(n_intervals * 3 + n_intervals - 1)
    ps[0:n_intervals * 3:3] = params.la
    ps[1:n_intervals * 3:3] = params.psi
    ps[2:n_intervals * 3:3] = params.p
    if n_t:
        ps[n_intervals * 3:] = time_intervals2optimized_values(np.concatenate((params.skyline_times, [T])))

    lk = loglikelihood(forest, *ps, T=T)
    print(lk)


if '__main__' == __name__:
    main()
