import os

import numpy as np

from bdpn import bd_model
from bdpn.formulas import get_c1, get_E, get_c2
from bdpn.parameter_estimator import optimize_likelihood_params, estimate_cis
from bdpn.tree_manager import TIME, read_forest, annotate_forest_with_time, get_T, rescale_forest

LOG_SCALING_FACTOR_P = 5
SCALING_FACTOR_P = np.exp(LOG_SCALING_FACTOR_P)

DEFAULT_MIN_PROB = 1e-6
DEFAULT_MAX_PROB = 1 - 1e-6
DEFAULT_MIN_RATE = 1e-3
DEFAULT_MAX_RATE = 1e2
DEFAULT_MAX_PARTNERS = 500

DEFAULT_LOWER_BOUNDS = [DEFAULT_MIN_RATE, DEFAULT_MIN_RATE, DEFAULT_MIN_PROB, 1]
DEFAULT_UPPER_BOUNDS = [DEFAULT_MAX_RATE, DEFAULT_MAX_RATE, DEFAULT_MAX_PROB, DEFAULT_MAX_PARTNERS]

PARAMETER_NAMES = np.array(['la', 'psi', 'rho', 'r'])
EPI_PARAMETER_NAMES = np.array(['R0', 'd'])

EPSILON = 1e-6

N_INTERVALS = 10000


def get_tt_fun_log(T):
    logT = np.log(T + 1)
    logdt = logT / N_INTERVALS
    tt = np.maximum((T - (np.exp(np.arange(0, N_INTERVALS + 1) * logdt) - 1))[::-1], 0)

    def get_index_t_log(t):
        if t <= 0:
            return 0
        if t >= T:
            return len(tt) - 1
        return len(tt) - 1 - int(np.log(T + 1 - t) // logdt)

    return tt, get_index_t_log


def get_tt_fun(T):
    dt = T / N_INTERVALS
    tt = np.arange(0, N_INTERVALS + 1) * dt

    def get_index_t(t):
        if t <= 0:
            return 0
        if t >= T:
            return len(tt) - 1
        return int(t // dt)

    return tt, get_index_t


def get_u_function(T, la, psi, rho, r, as_log=True):
    if as_log:
        tt, t2index = get_tt_fun_log(T)
    else:
        tt, t2index = get_tt_fun(T)

    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, c1)
    psi_not_rho = psi * (1 - rho)
    la_plus_psi = la + psi
    extra_recipients = r - 1

    Us = [1]
    prev_t = T

    # limit = (la + psi - c1) / (2 * la)
    for t in reversed(tt):
        if t == T:
            continue
        dt = prev_t - t
        prev_t = t
        # U = Us[-1] - dt * (la_plus_psi * Us[-1] - psi_not_rho - la * np.power(Us[-1], 2))
        dU = la_plus_psi * Us[-1] - psi_not_rho - la * np.power(Us[-1], 2) * np.exp((Us[-1] - 1) * extra_recipients)
        if dU <= 0:
            Us.extend([Us[-1]] * (len(tt) - len(Us)))
            break
        U = Us[-1] - dt * dU
        # Us.append(max(min(U, Us[-1]), 0))
        Us.append(max(min(U, Us[-1], bd_model.get_u(la, psi, c1, get_E(c1, c2, t, T))), 0))

    return np.array(Us)[::-1], tt, t2index


def get_log_p(t, ti, la, psi, r, Us, t2index):
    la_plus_psi = la + psi
    extra_recipients = r - 1
    if ti == t:
        return 1
    factors = 0
    dt = max((ti - t) / 100, EPSILON)
    t_prev = ti
    P = 1
    while t_prev > t:
        # U = get_u(t_prev, tt, logdt, T, Us)
        # TODO: use linear approximation instead for U value
        U = Us[t2index(t_prev)]

        x = la_plus_psi - la * (2 + extra_recipients * U) * U * np.exp(extra_recipients * (U - 1))

        # if we can approximate U with a constant from now, we have a formula
        if U == Us[0]:
            return np.log(P) + x * (t - t_prev) - factors
            # return log_subtraction(np.log(P), log_subtraction(x * t_prev, x * t)) - factors

        if P < 1e-3:
            P *= SCALING_FACTOR_P
            factors += LOG_SCALING_FACTOR_P

        if x == 0:
            break

        dt_cur = min(dt, t_prev - t) if x < 0 else max(min(dt, t_prev - t, 0.99 / x), EPSILON)

        P -= P * (x * dt_cur)
        if P <= 0:
            return -np.inf
        t_prev -= dt_cur

    return np.log(P) - factors


def get_forest_stats(forest):
    n_children = []
    internal_dists, external_dists = [], []
    for tree in forest:
        for n in tree.traverse():
            if not n.is_leaf():
                n_children.append(len(n.children))
                if n.dist:
                    external_dists.append(n.dist)
            elif n.dist:
                internal_dists.append(n.dist)
    return (min(n_children), np.mean(n_children), max(n_children),
            min(internal_dists) if internal_dists else None,
            np.median(internal_dists) if internal_dists else None,
            max(internal_dists) if internal_dists else None,
            min(external_dists), np.median(external_dists), max(external_dists))


def get_start_parameters(forest_stats, la=None, psi=None, rho=None, r=None):
    la_is_fixed = la is not None and la > 0
    psi_is_fixed = psi is not None and psi > 0
    rho_is_fixed = rho is not None and 0 < rho <= 1

    rho_est = rho if rho_is_fixed else 0.5

    if r is None:
        # mean num children
        r = forest_stats[1] - 1

    if la_is_fixed and psi_is_fixed:
        return np.array([la, psi, rho_est, r], dtype=np.float64)

    psi_est = psi if psi_is_fixed else 1 / forest_stats[-2]
    # if it is a corner case when we only have tips, let's use sampling times
    la_est = la if la_is_fixed else ((1 / forest_stats[4]) if forest_stats[4] else 1.1 * psi_est)
    if la_est <= psi_est:
        if la_is_fixed or not psi_is_fixed:
            psi_est = la_est * 0.75
        else:
            la_est = psi_est * 1.5

    return np.array([la_est, psi_est, rho_est, r], dtype=np.float64)


def loglikelihood(forest, la, psi, rho, r, T, threads=1, u=-1, as_log=True):

    log_psi_rho = np.log(psi) + np.log(rho)
    log_la = np.log(la)
    r_minus_one = r - 1
    log2 = np.log(2)
    log_r_min_1 = np.log(r_minus_one)

    Us, tt, t2index = get_u_function(T, la, psi, rho, r, as_log=as_log)

    hidden_lk = Us[0]
    if hidden_lk:
        u = len(forest) * hidden_lk / (1 - hidden_lk) if u is None or u < 0 else u
        res = u * np.log(hidden_lk)
    else:
        res = 0

    for tree in forest:

        root_ti = getattr(tree, TIME)
        root_t = root_ti - tree.dist
        res += get_log_p(root_t, root_ti, la, psi, r, Us, t2index)

        n = len(tree)
        res += n * log_psi_rho

        for n in tree.traverse('preorder'):
            if not n.is_leaf():
                res += log_la
                t = getattr(n, TIME)
                for child in n.children:
                    ti = getattr(child, TIME)
                    log_pi = get_log_p(t, ti, la, psi, r, Us, t2index)
                    res += log_pi
                U = Us[t2index(t)]
                c = len(n.children)

                if c == 2 and r == 1:
                    res += log2
                else:
                    res += -r_minus_one + (c - 2) * log_r_min_1

                    series = [(c - 1) * c]
                    while series[-1] >= 0:
                        k = len(series)
                        next_value = series[-1] * U * (r - 1) / k * ((c + k) / (c + k - 2))
                        if next_value <= 0:
                            break
                        series.append(next_value)

                    res += np.log(sum(series))
    return res


def format_parameters(la, psi, rho, r, scaling_factor=1, fixed=None):
    if fixed is None:
        return ', '.join('{}={:.10f}'.format(*_)
                         for _ in zip(np.concatenate([PARAMETER_NAMES, EPI_PARAMETER_NAMES]),
                                      [la * scaling_factor, psi * scaling_factor, rho, r, la / psi * r,
                                       1 / (psi * scaling_factor)]))
    else:
        return ', '.join('{}={:.10f}{}'.format(_[0], _[1], '' if _[2] is None else ' (fixed)')
                         for _ in zip(np.concatenate([PARAMETER_NAMES, EPI_PARAMETER_NAMES]),
                                      [la * scaling_factor, psi * scaling_factor, rho, r,
                                       la / psi * r, 1 / (psi * scaling_factor)],
                                      np.concatenate([fixed, [fixed[0]/fixed[1] if fixed[0] and fixed[1] else None,
                                                              1 / fixed[1] if fixed[1] else None]])))


def infer(forest, T, la=None, psi=None, p=None, r=None,
          lower_bounds=DEFAULT_LOWER_BOUNDS, upper_bounds=DEFAULT_UPPER_BOUNDS, ci=False,
          start_parameters=None, threads=1, scaling_factor=1, **kwargs):
    """
    Infers BD model parameters from a given forest.

    :param forest: list of one or more trees
    :param la: transmission rate
    :param psi: removal rate
    :param p: sampling probability
    :param lower_bounds: array of lower bounds for parameter values (la, psi, p)
    :param upper_bounds: array of upper bounds for parameter values (la, psi, p)
    :param ci: whether to calculate the CIs or not
    :return: tuple(vs, cis) of estimated parameter values vs=[la, psi, p]
        and CIs ci=[[la_min, la_max], [psi_min, psi_max], [p_min, p_max]].
        In the case when CIs were not set to be calculated,
        their values would correspond exactly to the parameter values.
    """
    if la is None and psi is None and p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')
    if la:
        la /= scaling_factor
    if psi:
        psi /= scaling_factor
    bounds = np.zeros((4, 2), dtype=np.float64)
    lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
    lower_bounds[:2] /= scaling_factor
    upper_bounds[:2] /= scaling_factor
    forest_stats = get_forest_stats(forest)
    if forest_stats[2] > 2:
        lower_bounds[-1] = max(lower_bounds[-1], 1 + 1e-3)
    if not np.all(upper_bounds >= lower_bounds):
        raise ValueError('Lower bounds cannot be greater than upper bounds')
    if np.any(lower_bounds < 0):
        raise ValueError('Bounds must be non-negative')
    if upper_bounds[2] > 1:
        raise ValueError('Probability bounds must be between 0 and 1')
    if lower_bounds[-1] < 1:
        raise ValueError('Avg number of recipients cannot be below 1')

    bounds[:, 0] = lower_bounds
    bounds[:, 1] = upper_bounds

    input_params = np.array([la, psi, p, r])

    if start_parameters is None:
        # start_parameters = get_start_parameters(forest_stats, la, psi, p, r)
        vs, _ = optimize_likelihood_params(forest, T, input_parameters=np.array(input_params[:-1]),
                                           loglikelihood_function=bd_model.loglikelihood,
                                           bounds=np.array(bounds[:-1, :]),
                                           start_parameters=bd_model.get_start_parameters(forest, la, psi, p),
                                           num_attemps=1)
        start_parameters = np.concatenate([vs, [r if r is not None and r >= 1 else (forest_stats[1] - 1)]])
    start_parameters = np.minimum(np.maximum(start_parameters, bounds[:, 0]), bounds[:, 1])

    print('Lower bounds are set to:\t{}'
          .format(format_parameters(*lower_bounds, scaling_factor=scaling_factor)))
    print('Upper bounds are set to:\t{}'
          .format(format_parameters(*upper_bounds, scaling_factor=scaling_factor)))
    print('Starting parameters:\t{}'
          .format(format_parameters(*start_parameters, scaling_factor=scaling_factor, fixed=input_params)))

    # optimise_as_logs = np.array([True, True, True, False])
    vs, lk = optimize_likelihood_params(forest, T, input_parameters=input_params,
                                        loglikelihood_function=
                                        lambda *args, **kwargs: loglikelihood(*args, **kwargs),
                                        bounds=bounds,
                                        start_parameters=start_parameters,
                                        formatter=lambda _: format_parameters(*_, scaling_factor=scaling_factor))
    print('Estimated BD-mult parameters:\t{};\tloglikelihood={}'
          .format(format_parameters(*vs, scaling_factor=scaling_factor), lk))
    if ci:
        cis = estimate_cis(T, forest, input_parameters=input_params, loglikelihood_function=loglikelihood,
                           optimised_parameters=vs, bounds=bounds, threads=threads)
        print('Estimated CIs:\n\tlower:\t{}\n\tupper:\t{}'
              .format(format_parameters(*cis[:,0], scaling_factor=scaling_factor),
                      format_parameters(*cis[:,1], scaling_factor=scaling_factor)))
    else:
        cis = None
    return vs, cis


def save_results(vs, cis, log, ci=False):
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, 'w+') as f:
        f.write(',{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability',
                                         'transmission rate', 'removal rate', 'avg number of recipients'])))
        la, psi, rho, r = vs
        R0 = la / psi * r
        rt = 1 / psi
        f.write('value,{}\n'.format(','.join(str(_) for _ in [R0, rt, rho, la, psi, r])))
        if ci:
            (la_min, la_max), (psi_min, psi_max), (rho_min, rho_max), (r_min, r_max) = cis
            R0_min, R0_max = la_min / psi_max * r_min, la_max / psi_min * r_max
            rt_min, rt_max = 1 / psi_max, 1 / psi_min
            f.write('CI_min,{}\n'.format(
                ','.join(str(_) for _ in [R0_min, rt_min, rho_min, la_min, psi_min, r_min])))
            f.write('CI_max,{}\n'.format(
                ','.join(str(_) for _ in [R0_max, rt_max, rho_max, la_max, psi_max, r_max])))


def rescale_parameters(vs, scaling_factor):
    vs[:2] /= scaling_factor


def main():
    """
    Entry point for tree parameter estimation with the BD model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimated BD parameters.")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--la', required=False, default=None, type=float, help="transmission rate")
    parser.add_argument('--psi', required=False, default=None, type=float, help="removal rate")
    parser.add_argument('--p', required=False, default=None, type=float, help='sampling probability')
    parser.add_argument('--r', required=False, default=None, type=float, help='avg number of recipients (>= 1)')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=4,
                        help="upper bounds for parameters (la, psi, p, r)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=4,
                        help="lower bounds for parameters (la, psi, p, r)", default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    parser.add_argument('--threads', required=False, type=int, default=1, help="number of threads for parallelization")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    forest = read_forest(params.nwk)
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    print(f'Read a forest of {len(forest)} trees with {sum(len(_) for _ in forest)} tips in total, '
          f'evolving over time {T_initial}')
    T = 10000
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)

    vs, cis = infer(forest, T, **vars(params), scaling_factor=scaling_factor)
    vs[:2] *= scaling_factor
    if cis is not None:
        cis[:2, 0] *= scaling_factor
        cis[:2, 1] *= scaling_factor
    save_results(vs, cis, params.log, ci=params.ci)


def loglikelihood_main():
    """
    Entry point for tree likelihood estimation with the BD model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Calculate BD likelihood on a given forest for given parameter values.")
    parser.add_argument('--la', required=True, type=float, help="transmission rate")
    parser.add_argument('--psi', required=True, type=float, help="removal rate")
    parser.add_argument('--p', required=True, type=float, help='sampling probability')
    parser.add_argument('--r', required=True, type=float, help='avg recipient number')
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--u', required=False, type=int, default=-1,
                        help="number of hidden trees (estimated by default)")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    T = 1000
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    lk = loglikelihood(forest,
                       la=params.la / scaling_factor, psi=params.psi / scaling_factor, rho=params.p, r=params.r, T=T)
    print(lk)


if '__main__' == __name__:
    main()
