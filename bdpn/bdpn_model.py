import os
from multiprocessing.pool import ThreadPool

import numpy as np

from bdpn import bd_model
from bdpn.formulas import get_log_p, get_c1, get_c2, get_E, get_log_ppb, get_log_pn, get_log_ppb_from_p_pn, \
    get_u, get_log_no_event, get_log_ppa, get_log_ppa_from_ppb, get_log_pb, log_subtraction, log_sum
from bdpn.parameter_estimator import optimize_likelihood_params, estimate_cis
from bdpn.tree_manager import TIME, read_forest, annotate_forest_with_time, get_T, preannotate_notifiers, NOTIFIERS

PARAMETER_NAMES = np.array(['la', 'psi', 'phi', 'rho', 'upsilon'])

DEFAULT_LOWER_BOUNDS = [bd_model.DEFAULT_MIN_RATE, bd_model.DEFAULT_MIN_RATE, bd_model.DEFAULT_MIN_RATE,
                        bd_model.DEFAULT_MIN_PROB, 0]
DEFAULT_UPPER_BOUNDS = [bd_model.DEFAULT_MAX_RATE, bd_model.DEFAULT_MAX_RATE, bd_model.DEFAULT_MAX_RATE * 1e3,
                        bd_model.DEFAULT_MAX_PROB, bd_model.DEFAULT_MAX_PROB]


def preprocess_node(params):
    """
    Precalculates different branch .

    :param node: tree node whose children (if any) are already processed
    :return: void, add a node annotation
    """
    node, T, c1, c2, la, psi, phi, \
        log_not_rho, log_psi_rho_ups, log_psi_rho_not_ups, log_phi_not_ups, log_phi_ups, log_la, log_2_la = params

    ti = getattr(node, TIME)
    tj = ti - node.dist

    E_tj = get_E(c1=c1, c2=c2, t=tj, T=T)
    E_ti = get_E(c1, c2, ti, T)

    log_p = get_log_p(c1=c1, t=tj, ti=ti, E_t=E_tj, E_ti=E_ti)
    log_pn = get_log_pn(la=la, psi=psi, t=tj, ti=ti)

    th = tj + (ti - tj) / 2
    E_th = get_E(c1, c2, th, T)
    E_T = get_E(c1, c2, T, T)
    log_no_event_tj_th = get_log_no_event(la + psi, tj, th)
    log_u_th = np.log(get_u(la, psi, c1, E_th))

    if not node.is_leaf():
        standard_br = log_p + log_2_la
        node.add_feature('standard_br', standard_br)
    else:
        log_pn_psi_rho_ups = log_pn + log_psi_rho_ups
        node.add_feature('lxn', log_pn_psi_rho_ups)

        log_ppa_ti_T = get_log_ppa(la, psi, phi, c1, ti, T, E_ti, E_T)
        log_bottom_n = get_log_pn(la, psi, th, ti)
        psi_ti_minus_th = psi * (ti - th)
        log_u_p_th_ti = get_log_ppb(la, psi, c1, th, ti, E_th, E_ti) \
                        + log_sum([np.log(1 - np.exp(-psi_ti_minus_th)) + psi_ti_minus_th + log_not_rho,
                                   log_ppa_ti_T])
        log_top_standard = log_subtraction(get_log_p(c1, tj, th, E_tj, E_th), log_no_event_tj_th)
        log_standard_n = log_top_standard + (log_u_p_th_ti - log_u_th) + log_bottom_n

        log_p_psi_rho_not_ups = log_p + log_psi_rho_not_ups
        log_standard_n_psi_rho_ups = log_standard_n + log_psi_rho_ups
        lxx = log_sum([log_p_psi_rho_not_ups,
                       log_standard_n_psi_rho_ups])
        node.add_feature('lxx', lxx)

    notifiers = getattr(node, NOTIFIERS)
    if notifiers:
        log_ppb = get_log_ppb_from_p_pn(log_p=log_p, log_pn=log_pn)
        log_ppa = get_log_ppa_from_ppb(log_ppb=log_ppb, psi=psi, phi=phi, t=tj, ti=ti)

        log_bottom_standard = get_log_p(c1, th, ti, E_th, E_ti)
        log_top_ppb = log_subtraction(get_log_ppb(la, psi, c1, tj, th, E_tj, E_th), log_no_event_tj_th)
        log_top_ppa = log_subtraction(get_log_ppa(la, psi, phi, c1, tj, th, E_tj, E_th),
                                      get_log_no_event(la + phi, tj, th))

        notifier2precalculated_values = {}
        for notifier in notifiers:
            tr = getattr(notifier, TIME)
            E_tr = get_E(c1, c2, tr, T)
            log_ppa_tr_T = get_log_ppa(la, psi, phi, c1, tr, T, E_tr, E_T)
            log_pp_tj_ti = (get_log_ppb(la, psi, c1, tj, tr, E_tj, E_tr)
                            + get_log_ppa(la, psi, phi, c1, tr, ti, E_tr, E_ti)) if tj < tr < ti else None
            notifier2precalculated_values[notifier] = tr, E_tr, log_pp_tj_ti, log_ppa_tr_T

        notifier2log_top_pp_u_p = {}
        log_u_ppa_th = get_log_ppa(la, psi, phi, c1, th, T, E_th, E_T)
        for notifier in notifiers:
            tr, E_tr, log_pp_tj_ti, log_ppa_tr_T = notifier2precalculated_values[notifier]
            if th <= tr:
                psi_tr_minus_th = psi * (tr - th)
                notifier2log_top_pp_u_p[notifier] = \
                    log_top_ppb, \
                        get_log_ppb(la, psi, c1, th, tr, E_th, E_tr) \
                        + log_sum([np.log(1 - np.exp(-psi_tr_minus_th)) + psi_tr_minus_th + log_not_rho,
                                   log_ppa_tr_T])
            elif tj >= tr:
                notifier2log_top_pp_u_p[notifier] = log_top_ppa, log_u_ppa_th
            else:
                notifier2log_top_pp_u_p[notifier] = \
                    get_log_ppb(la, psi, c1, tj, tr, E_tj, E_tr) + \
                    log_subtraction(get_log_ppa(la, psi, phi, c1, tr, th, E_tr, E_th),
                                    get_log_no_event(la + phi, tr, th)), log_u_ppa_th

        notifier2log_p_standard \
            = {notifier: log_top_pp + (log_u_p_th_tr - log_u_th) + log_bottom_standard
               for (notifier, (log_top_pp, log_u_p_th_tr)) in notifier2log_top_pp_u_p.items()}

        if not node.is_leaf():
            notifier2branches = {}
            for notifier in notifiers:
                tr, E_tr, log_pp_tj_ti, log_ppa_tr_T = notifier2precalculated_values[notifier]
                observed_branch = log_ppa if tr <= tj else (log_ppb if tr >= ti else log_pp_tj_ti)
                notifier2branches[notifier] = tr, observed_branch + log_la, notifier2log_p_standard[notifier] + log_2_la
            node.add_feature('r2br', notifier2branches)
        else:
            th1 = tj + (ti - tj) / 3
            th2 = tj + 2 * (ti - tj) / 3
            E_th1 = get_E(c1, c2, th1, T)
            E_th2 = get_E(c1, c2, th2, T)
            log_top1_ppb = log_subtraction(get_log_ppb(la, psi, c1, tj, th1, E_tj, E_th1),
                                           get_log_no_event(la + psi, tj, th1))
            log_top1_ppa = log_subtraction(get_log_ppa(la, psi, phi, c1, tj, th1, E_tj, E_th1),
                                           get_log_no_event(la + phi, tj, th1))

            log_u_th1 = np.log(get_u(la, psi, c1, E_th1))
            log_u_th2 = np.log(get_u(la, psi, c1, E_th2))
            psi_ti_minus_th2 = psi * (ti - th2)
            log_u_p_th2_ti = get_log_ppb(la, psi, c1, th2, ti, E_th2, E_ti) \
                             + log_sum([np.log(1 - np.exp(-psi_ti_minus_th2)) + psi_ti_minus_th2 + log_not_rho,
                                        log_ppa_ti_T])

            log_top2_standard = log_subtraction(get_log_p(c1, th1, th2, E_th1, E_th2),
                                                get_log_no_event(la + psi, th1, th2))

            notifier2lnx = {}
            lnx_ppa = log_ppa + log_phi_not_ups
            lnx_ppb = log_ppb + log_psi_rho_not_ups
            log_u_ppa_th1 = get_log_ppa(la, psi, phi, c1, th1, T, E_th1, E_T)
            for notifier in notifiers:
                tr, E_tr, log_pp_tj_ti, log_ppa_tr_T = notifier2precalculated_values[notifier]
                log_top_pp, log_u_p_th_tr = notifier2log_top_pp_u_p[notifier]
                log_mixed_p_n = log_top_pp \
                                + ((log_u_p_th_tr if tr < ti else log_u_p_th_ti) - log_u_th) \
                                + log_bottom_n
                if th1 <= tr:
                    log_top1_pp = log_top1_ppb
                    psi_tr_minus_th1 = psi * (tr - th1)
                    log_u_p_th1 = get_log_ppb(la, psi, c1, th1, tr, E_th1, E_tr) \
                                  + log_sum([np.log(1 - np.exp(-psi_tr_minus_th1)) + psi_tr_minus_th1 + log_not_rho,
                                             log_ppa_tr_T])
                else:
                    log_u_p_th1 = log_u_ppa_th1
                    if tj >= tr:
                        log_top1_pp = log_top1_ppa
                    else:
                        log_top1_pp = get_log_ppb(la, psi, c1, tj, tr, E_tj, E_tr) \
                                      + log_subtraction(get_log_ppa(la, psi, phi, c1, tr, th1, E_tr, E_th1),
                                                        get_log_no_event(la + phi, tr, th1))
                log_mixed_p_standard_n = log_top1_pp + (log_u_p_th1 - log_u_th1) \
                                         + log_top2_standard + (log_u_p_th2_ti - log_u_th2) \
                                         + get_log_pn(la, psi, th2, ti)
                notifier2lnx[notifier] = log_sum([lnx_ppa if tr <= tj \
                                                      else (lnx_ppb if tr >= ti else
                                                            (log_pp_tj_ti + log_phi_not_ups)),
                                                  notifier2log_p_standard[notifier] + log_psi_rho_not_ups,
                                                  log_mixed_p_n + log_psi_rho_ups,
                                                  log_mixed_p_standard_n + log_psi_rho_ups])
            node.add_feature('lnx', notifier2lnx)

            notifier2lnn = {}
            lnn_ppa = get_log_pb(la, phi, tj, ti) + log_phi_ups
            lnn_ppb = log_pn_psi_rho_ups
            for notifier in notifiers:
                tr = getattr(notifier, TIME)
                notifier2lnn[notifier] = lnn_ppa if tr <= tj \
                    else (lnn_ppb if tr >= ti
                          else (get_log_pn(la, psi, tj, tr) + get_log_pb(la, phi, tr, ti) + log_phi_ups))
            node.add_feature('lnn', notifier2lnn)
    return True


def preprocess_forest(forest):
    annotate_forest_with_time(forest)
    preannotate_notifiers(forest)


def loglikelihood(forest, la, psi, phi, rho, upsilon, T, threads=1):
    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)
    log_la, log_psi, log_phi, log_rho, log_not_rho, log_ups, log_not_ups, log_2 = \
        np.log([la, psi, phi, rho, 1 - rho, upsilon, 1 - upsilon, 2])
    log_2_la = log_2 + log_la
    log_psi_rho = log_psi + log_rho
    log_psi_rho_ups = log_psi_rho + log_ups
    log_psi_rho_not_ups = log_psi_rho + log_not_ups
    log_phi_ups = log_phi + log_ups
    log_phi_not_ups = log_phi + log_not_ups

    all_nodes = []
    for tree in forest:
        for node in tree.traverse():
            all_nodes.append((node, T, c1, c2, la, psi, phi,
                              log_not_rho, log_psi_rho_ups, log_psi_rho_not_ups, log_phi_not_ups, log_phi_ups,
                              log_la, log_2_la))

    if threads > 1:
        with ThreadPool(processes=threads) as pool:
            pool.map(func=preprocess_node, iterable=all_nodes, chunksize=max(1, len(all_nodes) // threads + 1))
    else:
        for node in all_nodes:
            preprocess_node(node)

    u = get_u(la, psi, c1, E_t=get_E(c1=c1, c2=c2, t=0, T=T))
    log_likelihood = len(forest) * u / (1 - u) * np.log(u)
    for tree in forest:
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                continue

            i0, i1 = node.children
            is_tip0, is_tip1 = i0.is_leaf(), i1.is_leaf()

            standard_br = getattr(node, 'standard_br')
            notifiers = getattr(node, NOTIFIERS)
            notifier2branches = getattr(node, 'r2br', set())

            if not is_tip0 and not is_tip1:
                node.add_feature('lx', standard_br + getattr(i0, 'lx') + getattr(i1, 'lx'))

                notifier2ln = {}
                for notifier in notifiers:
                    tr, observed_br, mixed_br = notifier2branches[notifier]
                    notifier2ln[notifier] = \
                        log_sum([observed_br + log_sum([getattr(i0, 'ln')[notifier] + getattr(i1, 'lx'),
                                                        getattr(i0, 'lx') + getattr(i1, 'ln')[notifier]]),
                                 mixed_br + getattr(i0, 'lx') + getattr(i1, 'lx')])

                node.add_feature('ln', notifier2ln)
                continue

            ti0, ti1 = getattr(i0, TIME), getattr(i1, TIME)
            if is_tip0 and is_tip1:
                lxx_i0 = getattr(i0, 'lxx')
                lxx_i1 = getattr(i1, 'lxx')
                lxn_i1 = getattr(i1, 'lxn')
                lxn_i0 = getattr(i0, 'lxn')
                lnn_i1_by_i0 = getattr(i1, 'lnn')[i0]
                lnn_i0_by_i1 = getattr(i0, 'lnn')[i1]
                lnx_i0_by_i1 = getattr(i0, 'lnx')[i1]
                lnx_i1_by_i0 = getattr(i1, 'lnx')[i0]
                log_unnotified_subtree = log_sum([lxx_i0 + lxx_i1,
                                                  lxn_i0 + lnx_i1_by_i0,
                                                  lnx_i0_by_i1 + lxn_i1,
                                                  lnn_i0_by_i1 + lnn_i1_by_i0
                                                  ])
                node.add_feature('lx', standard_br + log_unnotified_subtree)

                notifier2ln = {}
                for notifier in notifiers:
                    tr, observed_br, mixed_br = notifier2branches[notifier]
                    first_i0_r = i0 if ti0 <= tr else notifier
                    first_i1_r = i1 if ti1 <= tr else notifier
                    notifier2ln[notifier] = \
                        log_sum([observed_br + log_sum([getattr(i0, 'lnx')[first_i1_r] + lxn_i1,
                                                        getattr(i0, 'lnn')[first_i1_r] + lnn_i1_by_i0,
                                                        lxn_i0 + getattr(i1, 'lnx')[first_i0_r],
                                                        lnn_i0_by_i1 + getattr(i1, 'lnn')[first_i0_r],
                                                        getattr(i0, 'lnn')[notifier] + lnx_i1_by_i0,
                                                        lnx_i0_by_i1 + getattr(i1, 'lnn')[notifier],
                                                        getattr(i0, 'lnx')[notifier] + lxx_i1,
                                                        lxx_i0 + getattr(i1, 'lnx')[notifier]]
                                                       ),
                                 mixed_br + log_unnotified_subtree
                                 ])
                node.add_feature('ln', notifier2ln)
                continue

            # i0 is a tip and i1 is internal
            if is_tip1:
                i0, i1 = i1, i0
                ti0, ti1 = ti1, ti0
            lxx_i0 = getattr(i0, 'lxx')
            lx_i1 = getattr(i1, 'lx')
            lxn_i0 = getattr(i0, 'lxn')
            ln_i1_by_i0 = getattr(i1, 'ln')[i0]
            log_unnotified_subtree = log_sum([lxx_i0 + lx_i1, lxn_i0 + ln_i1_by_i0])
            node.add_feature('lx', standard_br + log_unnotified_subtree)

            notifier2ln = {}
            for notifier in notifiers:
                tr, observed_br, mixed_br = notifier2branches[notifier]
                first_i0_r = i0 if ti0 < tr else notifier
                notifier2ln[notifier] = \
                    log_sum([observed_br + log_sum([getattr(i0, 'lnn')[notifier] + ln_i1_by_i0,
                                                    getattr(i0, 'lnx')[notifier] + lx_i1,
                                                    lxn_i0 + getattr(i1, 'ln')[first_i0_r],
                                                    lxx_i0 + getattr(i1, 'ln')[notifier]]
                                                   ),
                             mixed_br + log_unnotified_subtree
                             ])

            node.add_feature('ln', notifier2ln)
        log_likelihood += getattr(tree, 'lx' if not tree.is_leaf() else 'lxx')
    return log_likelihood


def save_results(vs, cis, log, ci=False):
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, 'w+') as f:
        f.write(',{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability', 'notification probability',
                                         'removal time after notification',
                                         'transmission rate', 'removal rate', 'partner removal rate'])))
        la, psi, phi, rho, rho_p = vs
        R0 = la / psi
        rt = 1 / psi
        prt = 1 / phi
        f.write('value,{}\n'.format(','.join(str(_) for _ in [R0, rt, rho, rho_p, prt, la, psi, phi])))
        if ci:
            (la_min, la_max), (psi_min, psi_max), (psi_p_min, psi_p_max), (rho_min, rho_max), (
                rho_p_min, rho_p_max) = cis
            R0_min, R0_max = la_min / psi, la_max / psi
            rt_min, rt_max = 1 / psi_max, 1 / psi_min
            prt_min, prt_max = 1 / psi_p_max, 1 / psi_p_min
            f.write('CI_min,{}\n'.format(
                ','.join(str(_) for _ in [R0_min, rt_min, rho_min, rho_p_min, prt_min, la_min, psi_min, psi_p_min])))
            f.write('CI_max,{}\n'.format(
                ','.join(str(_) for _ in [R0_max, rt_max, rho_max, rho_p_max, prt_max, la_max, psi_max, psi_p_max])))


def main():
    """
    Entry point for tree parameter estimation with the BDPN model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimated BDPN parameters.")
    parser.add_argument('--la', required=False, default=None, type=float, help="transmission rate")
    parser.add_argument('--psi', required=False, default=None, type=float, help="removal rate")
    parser.add_argument('--p', required=False, default=None, type=float, help='sampling probability')
    parser.add_argument('--upsilon', required=False, default=None, type=float, help='notification probability')
    parser.add_argument('--phi', required=False, default=None, type=float, help='partner removal rate')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=5,
                        help="upper bounds for parameters (la, psi, phi, p, upsilon)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=5,
                        help="lower bounds for parameters (la, psi, phi, p, upsilon)", default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    parser.add_argument('--threads', required=False, type=int, default=1, help="number of threads for parallelization")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the BD model parameters (la, psi, p) needs to be specified '
                         'for identifiability')

    forest = read_forest(params.nwk)
    preprocess_forest(forest)
    T = get_T(T=None, forest=forest)
    print('Read a forest of {} trees with {} tips in total, evolving over time {}'
          .format(len(forest), sum(len(_) for _ in forest), T))
    vs, cis = infer(forest, T, **vars(params))

    save_results(vs, cis, params.log, ci=params.ci)


def loglikelihood_main():
    """
    Entry point for tree likelihood estimation with the BDPN model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Calculate BDPN likelihood on a given forest for given parameter values.")
    parser.add_argument('--la', required=True, type=float, help="transmission rate")
    parser.add_argument('--psi', required=True, type=float, help="removal rate")
    parser.add_argument('--p', required=True, type=float, help='sampling probability')
    parser.add_argument('--upsilon', required=True, type=float, help='notification probability')
    parser.add_argument('--phi', required=True, type=float, help='partner removal rate')
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    preprocess_forest(forest)
    T = get_T(T=None, forest=forest)
    lk = loglikelihood(forest, la=params.la, psi=params.psi, rho=params.p, phi=params.phi, upsilon=params.upsilon, T=T)
    print(lk)


def infer(forest, T, la=None, psi=None, phi=None, p=None, upsilon=None,
          lower_bounds=DEFAULT_LOWER_BOUNDS, upper_bounds=DEFAULT_UPPER_BOUNDS, ci=False, threads=1, **kwargs):
    """
    Infers BDPN model parameters from a given forest.

    :param forest: list of one or more trees
    :param la: transmission rate
    :param psi: removal rate
    :param phi: partner removal rate
    :param p: sampling probability
    :param upsilon: partner notification probability
    :param lower_bounds: array of lower bounds for parameter values (la, psi, phi, p, upsilon)
    :param upper_bounds: array of upper bounds for parameter values (la, psi, phi, p, upsilon)
    :param ci: whether to calculate the CIs or not
    :return: tuple(vs, cis) of estimated parameter values vs=[la, psi, phi, p, upsilon]
        and CIs ci=[[la_min, la_max], ..., [upsilon_min, upsilon_max]]. In the case when CIs were not set to be calculated,
        their values would correspond exactly to the parameter values.
    """
    if la is None and psi is None and p is None:
        raise ValueError('At least one of the BD model parameters (la, psi, p) needs to be specified '
                         'for identifiability')

    bounds = np.zeros((5, 2), dtype=np.float64)
    lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
    if not np.all(upper_bounds >= lower_bounds):
        raise ValueError('Lower bounds cannot be greater than upper bounds')
    if np.any(lower_bounds < 0):
        raise ValueError('Bounds must be non-negative')
    if upper_bounds[-2] > 1 or upper_bounds[-1] > 1:
        raise ValueError('Probability bounds must be between 0 and 1')

    bounds[:, 0] = lower_bounds
    bounds[:, 1] = upper_bounds

    input_params = np.array([la, psi, phi, p, upsilon])
    vs, _ = bd_model.infer(forest, T=T, la=la, psi=psi, p=p,
                           lower_bounds=bounds[[0, 1, 3], 0], upper_bounds=bounds[[0, 1, 3], 1], ci=False)
    vs_extended = np.array([vs[0], vs[1],
                            np.random.uniform(bounds[2, 0], bounds[2, 1], 1)[0] if phi is None else phi,
                            vs[-1], 0])
    if upsilon == 0:
        best_vs, best_lk = vs_extended, bd_model.loglikelihood(forest, *vs, T, threads)
    else:
        upsilon_estimated = upsilon is None or upsilon < 0 or upsilon > 1
        if upsilon_estimated and phi is None and lower_bounds[-1] == 0:
            best_vs, best_lk = vs_extended, bd_model.loglikelihood(forest, *vs, T, threads)
        else:
            best_vs, best_lk = None, -np.inf
        start_la, start_psi, start_rho = vs
        start_phi = min(max(start_psi * 1.25, lower_bounds[2]), upper_bounds[2]) \
            if phi is None or phi < 0 else phi
        start_ups = 0.1 if upsilon_estimated else upsilon
        # for ups in (0.1, 0.5) if upsilon_estimated else (upsilon,):
        start_parameters = np.array([start_la, start_psi, start_phi, start_rho, start_ups])
        print('Starting BDPN parameters:\t{}'
              .format(', '.join('{}={:g}{}'.format(_[0], _[1], '' if _[2] is None else ' (fixed)')
                                for _ in zip(PARAMETER_NAMES, start_parameters, input_params))))
        vs, lk = optimize_likelihood_params(forest, T=T, input_parameters=input_params,
                                            loglikelihood_function=loglikelihood, bounds=bounds,
                                            start_parameters=start_parameters, threads=threads)
        print(
            'Estimated BDPN parameters:\t{};\tloglikelihood={}'
            .format((', '.join('{}={:g}'.format(*_) for _ in zip(PARAMETER_NAMES, vs))), lk))

        if lk > best_lk:
            best_lk = lk
            best_vs = vs
    if ci:
        cis = estimate_cis(T, forest, input_parameters=input_params, loglikelihood_function=loglikelihood,
                           optimised_parameters=best_vs, bounds=bounds, threads=threads)
        print('Estimated CIs:\t{}'
              .format(', '.join('{}=[{:g},{:g}]'.format(p, *p_ci) for (p, p_ci) in zip(PARAMETER_NAMES, cis))))
    else:
        cis = None
    return best_vs, cis


if '__main__' == __name__:
    main()
