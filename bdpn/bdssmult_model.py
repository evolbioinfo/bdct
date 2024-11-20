import os

import numpy as np
from scipy.special import hyp1f1

from bdpn import bd_model
from bdpn.formulas import log_sum
from bdpn.parameter_estimator import optimize_likelihood_params, estimate_cis, rescale_log
from bdpn.tree_manager import TIME, read_forest, annotate_forest_with_time, get_T, rescale_forest

LOGLK_BDSSMULT = 'loglk_bdssmult'

LOG_SCALING_FACTOR_P = 5
SCALING_FACTOR_P = np.exp(LOG_SCALING_FACTOR_P)

DEFAULT_MIN_PROB = 1e-6
DEFAULT_MAX_PROB = 1 - 1e-6
DEFAULT_MIN_RATE = 1e-3
DEFAULT_MAX_RATE = 1e2
DEFAULT_MAX_PARTNERS_R_N = 10
DEFAULT_MAX_R_S_R_N = 25

DEFAULT_LOWER_BOUNDS = [DEFAULT_MIN_RATE, DEFAULT_MIN_RATE, DEFAULT_MIN_PROB, 0.5, 1 + 1e-6, 1 + 1e-6]
DEFAULT_UPPER_BOUNDS = [DEFAULT_MAX_RATE, DEFAULT_MAX_RATE, DEFAULT_MAX_PROB, 1 - 1e-6, DEFAULT_MAX_PARTNERS_R_N, DEFAULT_MAX_R_S_R_N]

PARAMETER_NAMES = np.array(['la', 'psi', 'rho', 'pi_N', 'r_N', 'r_S'])
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


def get_u_function(T, la, psi, rho, pi_N, r_N, r_S, as_log=True):
    if as_log:
        tt, t2index = get_tt_fun_log(T)
    else:
        tt, t2index = get_tt_fun(T)

    psi_not_rho = psi * (1 - rho)
    la_plus_psi = la + psi

    Us_N = [1]
    Us_S = [1]
    prev_t = T

    for t in reversed(tt):
        if t == T:
            continue
        dt = prev_t - t
        prev_t = t
        U_N = Us_N[-1]
        U_S = Us_S[-1]
        pi_N_U_N, pi_S_U_S = U_N * pi_N, U_S * (1 - pi_N)
        U = pi_N_U_N + pi_S_U_S
        dU_N = la_plus_psi * U_N - psi_not_rho - la * U_N * U * np.exp((r_N - 1) * (U - 1))
        dU_S = la_plus_psi * U_S - psi_not_rho - la * U_S * U * np.exp((r_S - 1) * (U - 1))
        if dU_N <= 0 and dU_S <= 0:
            Us_N.extend([Us_N[-1]] * (len(tt) - len(Us_N)))
            Us_S.extend([Us_S[-1]] * (len(tt) - len(Us_S)))
            break
        U_N -= dt * dU_N
        U_S -= dt * dU_S
        Us_N.append(max(min(U_N, Us_N[-1]), 0))
        Us_S.append(max(min(U_S, Us_S[-1]), 0))

    return np.array([np.array(Us_N)[::-1], np.array(Us_S)[::-1]]), tt, t2index


def get_log_p(t, ti, la, psi, pi_N, r_N, r_S, Us, t2index):
    """
    Calculates probabilities of evolving as on a tree branch, starting at time t in state N or S,
    and with the initial condition at time t0 >= t.

    :param t:
    :param ti: time of the initial condition
    :param la: transmission rate
    :param psi: removal rate
    :param pi_N: proportion of normal spreaders
    :param r_N: avg number of recipients per normal spreader's transmission
    :param r_S: avg number of recipients per superspreader's transmission
    :param Us: matrix 2xM of unsampled tree probabilities depending on evolution time,
        where the first row contains there probabilities for a tree started in state N, and the second in S
    :param t2index: mapping function converting a time to the corresponding column index in the Us matrix
    :return: matrix, where the first row contains:
        [p_N(t), p_S(t)], where p_N(t0) = 1 and p_S(t0) = 0;
        and the second: [p_N(t), p_S(t)], where p_N(t0) = 0 and p_S(t0) = 1.


    # TODO: vectorize the calculations
    """

    res = np.array([[0, -np.inf], [-np.inf, 0]])

    if ti == t:
        return res

    la_plus_psi = la + psi
    pi_S = 1 - pi_N
    r_N_min_1 = r_N - 1
    r_S_min_1 = r_S - 1

    for idx, (P_N_0, P_S_0) in enumerate(((1, 0), (0, 1))):

        factors = 0
        dt = max((ti - t) / 100, EPSILON)
        t_prev = ti
        P_N, P_S = P_N_0, P_S_0
        while t_prev > t:
            U_N, U_S = Us[:, t2index(t_prev)]
            P_avg = pi_N * P_N + pi_S * P_S
            U_avg = pi_N * U_N + pi_S * U_S
            U_avg_min_1 = U_avg - 1

            if P_N != 0 and P_S != 0:

                x_N = la_plus_psi - la * np.exp(r_N_min_1 * U_avg_min_1) * (U_avg + U_N * (1 + r_N_min_1 * U_avg) * (P_avg / P_N))
                x_S = la_plus_psi - la * np.exp(r_S_min_1 * U_avg_min_1) * (U_avg + U_S * (1 + r_S_min_1 * U_avg) * (P_avg / P_S))

                # if we can approximate U with a constant from now, we have a formula
                if U_N == Us[0, 0] and U_S == Us[1, 0]:
                    res[idx, :] = [np.log(P_N) + x_N * (t - t_prev) - factors,
                                   np.log(P_S) + x_S * (t - t_prev) - factors]
                    break

                if max(P_N, P_S) < 1e-3:
                    P_N *= SCALING_FACTOR_P
                    P_S *= SCALING_FACTOR_P
                    factors += LOG_SCALING_FACTOR_P

                if x_N == 0 and x_S == 0:
                    res[idx, :] = [np.log(P_N) - factors, np.log(P_S) - factors]
                    break

                x = max(x_N, x_S) if P_N > 0 and P_S > 0 else x_N if P_N_0 == 1 else x_S
                dt_cur = min(dt, t_prev - t) if x < 0 else max(min(dt, t_prev - t, 0.99 / x), EPSILON)

                P_N -= P_N * (x_N * dt_cur)
                P_S -= P_S * (x_S * dt_cur)
                if P_N_0 == 1:
                    P_N = max(P_N, 0)
                if P_S_0 == 1:
                    P_S = max(P_S, 0)

                if P_N == 0 == P_S:
                    res[idx, :] = [-np.inf, -np.inf]
                    break

                res[idx, :] = [np.log(P_N) - factors, np.log(P_S) - factors]

            else:
                x = (la_plus_psi - la * np.exp(r_N_min_1 * U_avg_min_1) * (U_avg + U_N * (1 + r_N_min_1 * U_avg) * (P_avg / P_N))) \
                    if P_N > 0 \
                    else (la_plus_psi - la * np.exp(r_S_min_1 * U_avg_min_1) * (U_avg + U_S * (1 + r_S_min_1 * U_avg) * (P_avg / P_S)))
                dt_cur = min(dt, t_prev - t) if x < 0 else max(min(dt, t_prev - t, 0.99 / x), EPSILON)


                P_N -= P_N * (x * dt_cur) if P_N > 1 \
                    else la_plus_psi * P_N * dt_cur - la * np.exp(r_N_min_1 * U_avg_min_1) * (U_avg * P_N + U_N * (1 + r_N_min_1 * U_avg) * P_avg) * dt_cur
                P_S -= P_S * (x * dt_cur) if P_N <= 0 \
                    else la_plus_psi * P_S * dt_cur - la * np.exp(r_S_min_1 * U_avg_min_1) * (U_avg * P_S + U_S * (1 + r_S_min_1 * U_avg) * P_avg) * dt_cur
                if P_N_0 == 1:
                    P_N = max(P_N, 0)
                if P_S_0 == 1:
                    P_S = max(P_S, 0)

                if P_N == 0 == P_S:
                    res[idx, :] = [-np.inf, -np.inf]
                    break
                if max(P_N, P_S) < 1e-3:
                    P_N *= SCALING_FACTOR_P
                    P_S *= SCALING_FACTOR_P
                    factors += LOG_SCALING_FACTOR_P

                res[idx, :] = [np.log(P_N) - factors, np.log(P_S) - factors]

            t_prev -= dt_cur

    return res


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


def loglikelihood(forest, la, psi, rho, pi_N, r_N, r_S_r_N, T, threads=1, u=-1, as_log=True):

    log_psi_rho = np.log(psi) + np.log(rho)
    log_la = np.log(la)
    r_S = r_S_r_N * r_N
    rs = np.array([r_N, r_S])
    r_min_1 = rs - 1
    log_r_min_1 = np.log(r_min_1)

    pis = np.array([pi_N, 1 - pi_N])
    log_pis = np.log(pis)

    Us, tt, t2index = get_u_function(T, la, psi, rho, pi_N, r_N, r_S, as_log=as_log)

    hidden_lk = pis.dot(Us[:, 0])
    if hidden_lk:
        u = len(forest) * hidden_lk / (1 - hidden_lk) if u is None or u < 0 else u
        res = u * np.log(hidden_lk)
    else:
        res = 0

    for tree in forest:

        for n in tree.traverse('postorder'):
            if n.is_leaf():
                n.add_feature(LOGLK_BDSSMULT, np.array([log_psi_rho, log_psi_rho]))
            else:
                t = getattr(n, TIME)
                t_idx = t2index(t)
                U_avg = pis.dot(Us[:, t_idx])
                log_Us = np.log(Us[:, t_idx])

                r_min_1_U_avg = r_min_1 * U_avg

                c = len(n.children)
                indices = np.arange(c)

                # an array to store the loglikelihood of this node's subtree, given its state is N (index 0) or S (1)
                n_loglk = np.array([-np.inf, -np.inf])

                # c x 2 array, where the i-th row contains loglk of i-th child subtree,
                # given child node i is in state N (column 0) or S (column 1)
                log_lk_children = np.array([getattr(child, LOGLK_BDSSMULT) for child in n.children])
                factors_lk = rescale_log(log_lk_children)
                # c x 2 x 2 matrix, where the i-th row contains logp of i-th child branch,
                # as a 2 x 2 matrix: [[logp_NN, logp_SN],
                #                     [logp_NS, logp_SS]]
                log_p_children = np.array([get_log_p(t, getattr(child, TIME), la, psi, pi_N, r_N, r_S, Us, t2index)
                                           for child in n.children])
                if np.any(np.isnan(log_p_children)):
                    for child in n.children:
                        lk = get_log_p(t, getattr(child, TIME), la, psi, pi_N, r_N, r_S, Us, t2index)

                factors_p = rescale_log(log_p_children)

                # c x 2 matrix, where the i-th row contains logp of i-th child branch
                # averaged over the starting state,
                # [logp_N, logp_S], where N or S are correspondingly the states at the branch end
                log_p_avg_children = np.array([np.log(np.exp(_).dot(pis)) for _ in log_p_children])
                problematic_indx = \
                    [i for i in indices
                     if np.any((np.abs(log_p_avg_children[i, :]) == np.inf) | np.isnan(log_p_avg_children[i, :]))]
                for i in problematic_indx:
                    row_by_col_mult = log_p_children[i] + log_pis
                    log_p_avg_children[i] = np.array([log_sum(row_by_col_mult[0]), log_sum(row_by_col_mult[1])])


                # c vector, where the i-th row contains loglk of i-th child branch+subtree
                # averaged over the starting state: log(p_N L_n + p_S L_S)
                log_p_avg_lk_children = np.array([np.log(np.exp(logp_avg).dot(np.exp(loglk)))
                                                  for (logp_avg, loglk) in zip(log_p_avg_children, log_lk_children)])


                problematic_indx = indices[(np.abs(log_p_avg_lk_children) == np.inf) | np.isnan(log_p_avg_lk_children)]
                for i in problematic_indx:
                    log_p_avg_lk_children[i] = log_sum(log_p_avg_children[i] + log_lk_children[i])

                if sum(log_p_avg_lk_children == -np.inf) or np.any(np.isnan(log_p_avg_lk_children)):
                    print(1)


                # c x 2 matrix, where the i-th row contains loglk of i-th child branch+subtree,
                # depending on the starting state, N (column 0) or S (column 1):
                # [log(p_NN L_n + p_NS L_S), log(p_SN L_n + p_SS L_S)]
                log_p_lk_children = np.array([np.log(np.exp(loglk).dot(np.exp(logp)))
                                              for (logp, loglk) in zip(log_p_children, log_lk_children)])

                problematic_indx = \
                    [i for i in indices
                     if np.any((np.abs(log_p_lk_children[i, :]) == np.inf) | np.isnan(log_p_lk_children[i, :]))]
                for i in problematic_indx:
                    row_by_col_mult = log_lk_children[i] + log_p_children[i].T
                    log_p_lk_children[i] = np.array([log_sum(row_by_col_mult[0]), log_sum(row_by_col_mult[1])])

                if np.any(log_p_lk_children == -np.inf) or np.any(np.isnan(log_p_lk_children)):
                    print(1)


                log_p_avg_lk_children_sum = log_p_avg_lk_children.sum()
                log_p_avg_lk_children_sum_but_i = log_p_avg_lk_children_sum - log_p_avg_lk_children

                c_min_1 = c - 1
                log_c_min_1 = np.log(c_min_1)
                c_min_2 = c - 2
                log_c = np.log(c)
                log_hyp = np.log(hyp1f1([c, c + 1, c, c + 1],
                                        [c_min_1, c, c_min_1, c],
                                        [r_min_1_U_avg[0], r_min_1_U_avg[0], r_min_1_U_avg[1], r_min_1_U_avg[1]]))

                factors = factors_lk + factors_p
                if r_N == 1:
                    # if c > 2 then the likelihood is zero (already set as loglk -inf)
                    if c == 2:
                        n_loglk[0] = (log_la + log_sum(log_p_lk_children[:, 0] + log_p_avg_lk_children_sum_but_i)
                                    - c * factors)
                else:
                    n_loglk[0] = log_la \
                               + log_sum(
                        [log_sum(log_p_lk_children[:, 0] + log_p_avg_lk_children_sum_but_i) - c * factors
                         + log_c_min_1 - r_min_1[0] + c_min_2 * log_r_min_1[0] + log_hyp[0],
                         log_Us[0] + log_p_avg_lk_children_sum - c * factors
                         + log_c - r_min_1[0] + c_min_1 * log_r_min_1[0] + log_hyp[1]
                         ]
                    )
                if r_S == 1:
                    # if c > 2 then the likelihood is zero (already set as loglk -inf)
                    if c == 2:
                        n_loglk[1] = log_la + log_sum(log_p_lk_children[:, 1] + log_p_avg_lk_children_sum_but_i) - c * factors
                else:
                    n_loglk[1] = log_la \
                               + log_sum(
                        [log_sum(log_p_lk_children[:, 1] + log_p_avg_lk_children_sum_but_i) - c * factors
                         + log_c_min_1 - r_min_1[1] + c_min_2 * log_r_min_1[1] + log_hyp[2],
                         log_Us[1] + log_p_avg_lk_children_sum - c * factors
                         + log_c - r_min_1[1] + c_min_1 * log_r_min_1[1] + log_hyp[3]
                         ]
                    )
                n.add_feature(LOGLK_BDSSMULT, n_loglk)

        root_ti = getattr(tree, TIME)
        root_t = root_ti - tree.dist
        # 2 x 2 matrix: [[logp_NN, logp_SN],
        #                [logp_NS, logp_SS]]
        log_p_root = get_log_p(root_t, root_ti, la, psi, pi_N, r_N, r_S, Us, t2index)
        factors = rescale_log(log_p_root)

        # 2 vector of logp averaged over the starting state: [logp_N, logp_S],
        # where N or S are correspondingly the states at the root branch end
        log_p_avg_root = np.log(np.exp(log_p_root).dot(pis))

        # 2 vector loglk, given root node is in state N (index 0) or S (index 1)
        log_lk = getattr(tree, LOGLK_BDSSMULT)
        factors += rescale_log(log_lk)

        res += np.log(np.exp(log_lk).dot(np.exp(log_p_avg_root))) - factors

    return res


def format_parameters(la, psi, rho, pi_N, r_N, r_S_r_N, scaling_factor=1, fixed=None):
    r_S = r_S_r_N * r_N
    if fixed is None:
        return ', '.join('{}={:.10f}'.format(*_)
                         for _ in zip(np.concatenate([PARAMETER_NAMES, EPI_PARAMETER_NAMES]),
                                      [la * scaling_factor, psi * scaling_factor, rho, pi_N, r_N, r_S,
                                       la / psi * (r_N * pi_N + r_S * (1 - pi_N)),
                                       1 / (psi * scaling_factor)]))
    else:
        return ', '.join('{}={:.10f}{}'.format(_[0], _[1], '' if _[2] is None else ' (fixed)')
                         for _ in zip(np.concatenate([PARAMETER_NAMES, EPI_PARAMETER_NAMES]),
                                      [la * scaling_factor, psi * scaling_factor, rho, pi_N, r_N, r_S,
                                       la / psi * (r_N * pi_N + r_S * (1 - pi_N)),
                                       1 / (psi * scaling_factor)],
                                      np.concatenate([fixed, [fixed[0]/fixed[1] if fixed[0] and fixed[1] else None,
                                                              1 / fixed[1] if fixed[1] else None]])))


def infer(forest, T, la=None, psi=None, p=None, pi_N=None, r_N=None, r_S_r_N=None,
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
    bounds = np.zeros((6, 2), dtype=np.float64)
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
    if upper_bounds[3] > 1:
        raise ValueError('Fraction of normal spreaders bounds must be between 0 and 1')
    if lower_bounds[-2] < 1:
        raise ValueError('Avg number of N recipients cannot be below 1')
    if lower_bounds[-1] < 1:
        raise ValueError('Avg number of S recipients cannot be below r_N')

    bounds[:, 0] = lower_bounds
    bounds[:, 1] = upper_bounds

    input_params = np.array([la, psi, p, pi_N, r_N, r_S_r_N])

    if start_parameters is None:
        # start_parameters = get_start_parameters(forest_stats, la, psi, p, r)
        vs, _ = optimize_likelihood_params(forest, T, input_parameters=np.array(input_params[:3]),
                                           loglikelihood_function=bd_model.loglikelihood,
                                           bounds=np.array(bounds[:3, :]),
                                           start_parameters=bd_model.get_start_parameters(forest, la, psi, p),
                                           num_attemps=1)
        r_start = forest_stats[1] - 1
        r_N_defined = r_N is not None and r_N >= 1
        r_S_r_N_defined = r_S_r_N is not None and r_S_r_N >= 1
        r_N_start = r_N if r_N_defined else r_start
        start_parameters = np.concatenate([vs, [pi_N if pi_N is not None and 0 <= pi_N <= 1 else 0.5,
                                                r_N_start, r_S_r_N if r_S_r_N_defined else 1.1]])
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
    print('Estimated BDSS-mult parameters:\t{};\tloglikelihood={}'
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
                                         'transmission rate', 'removal rate',
                                         'N fraction', 'S fraction',
                                         'avg number of recipients N', 'avg number of recipients S'])))
        la, psi, rho, pi_N, r_N, r_S_r_N = vs
        r_S = r_N * r_S_r_N
        R0 = la / psi * (r_N * pi_N + r_S * (1 - pi_N))
        rt = 1 / psi
        f.write('value,{}\n'.format(','.join(str(_) for _ in [R0, rt, rho,
                                                              la, psi,
                                                              pi_N, 1 - pi_N,
                                                              r_N, r_S])))
        if ci:
            (la_min, la_max), (psi_min, psi_max), (rho_min, rho_max), (pi_N_min, pi_N_max), (r_N_min, r_N_max), (r_S_r_N_min, r_S_r_N_max) = cis
            R0_min, R0_max = (la_min / psi_max * (r_N_min * pi_N_max + r_S_r_N_min * r_N_min * (1 - pi_N_max)),
                              la_max / psi_min * (r_N_max * pi_N_min + r_S_r_N_max * r_N_max * (1 - pi_N_min)))
            rt_min, rt_max = 1 / psi_max, 1 / psi_min
            f.write('CI_min,{}\n'.format(
                ','.join(str(_) for _ in [R0_min, rt_min, rho_min, la_min, psi_min, pi_N_min, 1 - pi_N_max, r_N_min, r_S_r_N_min * r_N_min])))
            f.write('CI_max,{}\n'.format(
                ','.join(str(_) for _ in [R0_max, rt_max, rho_max, la_max, psi_max, pi_N_max, 1 - pi_N_min, r_N_max, r_S_r_N_max * r_N_max])))


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
    parser.add_argument('--pi_N', required=False, type=float, help='fraction of normal spreaders')
    parser.add_argument('--r_N', required=False, type=float, help='avg recipient number for normal spreaders (>= 1)')
    parser.add_argument('--r_S_r_N', required=False, type=float, help='r_S / r_N >= 1')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=4,
                        help="upper bounds for parameters (la, psi, p, pi_N, r_N, r_S)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=4,
                        help="lower bounds for parameters (la, psi, p, pi_N, r_N, r_S)", default=DEFAULT_LOWER_BOUNDS)
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
    parser.add_argument('--pi_N', required=True, type=float, help='fraction of normal spreaders')
    parser.add_argument('--r_N', required=True, type=float, help='avg recipient number for normal spreaders')
    parser.add_argument('--r_S_r_N', required=False, type=float, help='r_S / r_N >= 1')
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
                       la=params.la / scaling_factor, psi=params.psi / scaling_factor, rho=params.p,
                       pi_N=params.pi_N, r_N=params.r_N, r_S_r_N=params.r_S_r_N, T=T)
    print(lk)


if '__main__' == __name__:
    main()
