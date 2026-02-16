import os
import numpy as np

from bdct.formulas import get_c1, get_c2, get_E, get_log_p, get_u, log_factorial
from bdct.parameter_estimator import estimate_cis, optimize_likelihood_params
from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time, get_T
from bdct.bd_model import infer, save_results, get_start_parameters

REMOVAL_RATE = 'removal rate'
TRANSMISSION_RATE = 'transmission rate'
SAMPLING_PROBABILITY = 'sampling probability'
INFECTIOUS_TIME = 'infectious time'
REPRODUCTIVE_NUMBER = 'R0'

RHO = 'rho'
PSI = 'psi'
LA = 'la'

DEFAULT_MIN_PROB = 1e-6
DEFAULT_MAX_PROB = 1
DEFAULT_MIN_RATE = 1e-3
DEFAULT_MAX_RATE = 1e3

DEFAULT_LOWER_BOUNDS = [DEFAULT_MIN_RATE, DEFAULT_MIN_RATE, DEFAULT_MIN_PROB]
DEFAULT_UPPER_BOUNDS = [DEFAULT_MAX_RATE, DEFAULT_MAX_RATE, DEFAULT_MAX_PROB]

PARAMETER_NAMES = np.array([LA, PSI, RHO])
EPI_PARAMETER_NAMES = np.array([REPRODUCTIVE_NUMBER, INFECTIOUS_TIME])

def loglikelihood_per_tree_rho(forest, la, psi, rhos, T, threads=1, u=-1):
    """
    Log-likelihood with different ρ per tree.

    :param rhos: list/array of len(forest), one ρ per tree
    :return: total logL
    """
    if len(rhos) != len(forest):
        raise ValueError(f"rhos must have {len(forest)} values, found {len(rhos)}")
    res = 0.0
    for i, tree in enumerate(forest):
        rho = float(rhos[i])
        # Calcul c1,c2 spécifiques à l'arbre
        c1 = get_c1(la=la, psi=psi, rho=rho)
        c2 = get_c2(la=la, psi=psi, c1=c1)
        log_psi_rho = np.log(psi) + np.log(rho)
        log_la = np.log(la)
        # Partie origines (t_start spécifique à l'arbre)
        t_start = getattr(tree, TIME) - tree.dist
        hidden_lk = get_u(la, psi, c1, E_t=get_E(c1=c1, c2=c2, t=t_start, T=T))
        if hidden_lk and hidden_lk > 0:
            res += hidden_lk / (1 - hidden_lk) * np.log(hidden_lk)
        # Partie feuilles
        n = len(tree)
        res += n * log_psi_rho
        # Partie branches internes (avec c1,c2 spécifiques)
        for n_node in tree.traverse('preorder'):
            if not n_node.is_leaf():
                t = getattr(n_node, TIME)
                E_t = get_E(c1=c1, c2=c2, t=t, T=T)
                num_children = len(n_node.children)
                res += log_factorial(num_children) + (num_children - 1) * log_la
                for child in n_node.children:
                    ti = getattr(child, TIME)
                    res += get_log_p(c1, t, ti=ti, E_t=E_t, E_ti=get_E(c1, c2, ti, T))
        # Partie racine
        root_ti = getattr(tree, TIME)
        root_t = root_ti - tree.dist
        res += get_log_p(c1, root_t, ti=root_ti, E_t=get_E(c1, c2, root_t, T), E_ti=get_E(c1, c2, root_ti, T))
    return res


def infer_per_tree_rho(forest, T,
                       la=None, psi=None, p=None,
                       lower_bounds=DEFAULT_LOWER_BOUNDS,
                       upper_bounds=DEFAULT_UPPER_BOUNDS,
                       ci=False, threads=1, num_attemps=3, **kwargs):
    n_trees = len(forest)
    # CHECKS FOR BOUNDS
    lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
    if not np.all(upper_bounds >= lower_bounds):
        raise ValueError('Lower bounds cannot be greater than upper bounds')
    if np.any(lower_bounds < 0):
        raise ValueError('Bounds must be non-negative')
    if upper_bounds[-1] > 1:
        raise ValueError('Probability bounds must be between 0 and 1')
    # CHECKS FOR INDENTIFIABILITY
    if la is None and psi is None and (p is None or (not np.isscalar(p) and all(v is None for v in p))):
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')
    # CHECK FOR SAMPLING PROBABILITY
    if p is not None:
        if np.isscalar(p):  # ex: p = 0.4
            if p <= 0 or p > 1:
                raise ValueError('Sampling probability must be between 0 and 1...')
        else:  # ex: p = [0.7, 0.3, 0.5]
            if len(p) != n_trees:
                raise ValueError(f"p list must have exactly {n_trees} values (one per tree)")
            if np.any(np.array(p) <= 0) or np.any(np.array(p) > 1):
                raise ValueError('All sampling probabilities must be between 0 and 1...')
    # 1) input_parameters extended : [la, psi, rho_0, ..., rho_{n-1}]
    input_params_ext = [la, psi]
    # rhos
    if p is None:
        # All ρ_i are free
        input_params_ext.extend([None] * n_trees)
    elif np.isscalar(p):
        # p is a float → all ρ_i fixed to p
        input_params_ext.extend([p] * n_trees)
    else:
        # p is a list/array → different ρ
        if len(p) != n_trees:
            raise ValueError(f"p must have {n_trees} values or be a scalar")
        input_params_ext.extend(p)
    input_params_ext = np.array(input_params_ext, dtype=object)
    # 2) bounds extended
    bounds_ext = np.zeros((2 + n_trees, 2), dtype=np.float64)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    # Initial bounds filling
    bounds_ext[:, 0] = lower_bounds[[0, 1] + [2] * n_trees]  # la, psi, ρ, ρ, ρ...
    bounds_ext[:, 1] = upper_bounds[[0, 1] + [2] * n_trees]  # la, psi, ρ, ρ, ρ...
    # Parameters fixing
    if la is not None:
        bounds_ext[0, :] = [la, la]  # line 0 = la fixed
    if psi is not None:
        bounds_ext[1, :] = [psi, psi]  # line 1 = psi fixed
    if p is not None:
        if np.isscalar(p):
            # All ρ_i fixed to p
            for i in range(2, 2 + n_trees):
                bounds_ext[i, :] = [p, p]
        else:
            # p = list → each ρ_i fixed to its value
            for i, rho_val in enumerate(p):
                bounds_ext[2 + i, :] = [rho_val, rho_val]
    # 3) start_parameters extended
    rho_for_start = 0.5
    if p is not None and np.isscalar(p):
        rho_for_start = float(p)
    # elif p is not None and (not np.isscalar(p)):
        # si p est une liste, on peut prendre la moyenne (ou 0.5)
        # rho_for_start = float(np.mean(p))
    start_base = get_start_parameters(forest, la=la, psi=psi, rho=rho_for_start)
    la0, psi0, rho0 = start_base
    start_params_ext = np.array([la0, psi0] + [rho0] * n_trees, dtype=np.float64)
    # If p is a list with different fixed values, these are given as start params
    if p is not None and (not np.isscalar(p)):
        start_params_ext[2:] = np.array(p, dtype=float)
    # 4) Compute the log-likelihood for initial parameters
    best_vs = np.array(start_params_ext)
    best_lk = loglikelihood_per_tree_rho(
        forest,
        la=start_params_ext[0],  #la
        psi=start_params_ext[1],  #psi
        rhos=start_params_ext[2:], #rhos array
        T=T,
        threads=threads)
    print(f"Bounds la:  [{bounds_ext[0, 0]:.3g}, {bounds_ext[0, 1]:.3g}]")
    print(f"Bounds psi: [{bounds_ext[1, 0]:.3g}, {bounds_ext[1, 1]:.3g}]")
    print(f"Bounds rho_i (same for all i unless fixed): [{lower_bounds[2]:.3g}, {upper_bounds[2]:.3g}]")
    if p is not None:
        if np.isscalar(p):
            print(f"All rhos fixed to {float(p):.6g}")
        else:
            print(f"Rhos fixed per tree: {list(map(float, p))}")
    def lk_wrapper(forest, *ps, T, threads=1):
        ps = np.asarray(ps, dtype=float)
        return loglikelihood_per_tree_rho(
            forest,
            la=ps[0],
            psi=ps[1],
            rhos=ps[2:],
            T=T,
            threads=threads
        )
    # 5) Usage of optimize_likelihood_params
    formatter = lambda vs: format_parameters_per_tree(vs, input_params=input_params_ext)
    vs, lk = optimize_likelihood_params(
        forest,
        T=T,
        input_parameters=input_params_ext,
        loglikelihood_function=lk_wrapper,
        bounds=bounds_ext,
        start_parameters=start_params_ext,
        threads=threads,
        num_attemps=num_attemps,
        formatter=formatter #format_parameters_per_tree
    )
    print(
        f'Estimated BD parameters:\t'
        f'{format_parameters_per_tree(vs, input_params=input_params_ext)};'
        f'\tloglikelihood={lk}'
    )
    # Keep the best likelihood
    if lk > best_lk:
        best_lk = lk
        best_vs = vs
    # CI (optional)
    if ci:
        cis = estimate_cis(
            T,
            forest,
            input_parameters=input_params_ext,
            loglikelihood_function=lk_wrapper,
            optimised_parameters=best_vs,
            bounds=bounds_ext,
            threads=threads
        )
        print(
            "Estimated CIs:\n"
            f"\tlower:\t{format_parameters_per_tree(cis[:, 0], input_params=input_params_ext, epi=False)}\n"
            f"\tupper:\t{format_parameters_per_tree(cis[:, 1], input_params=input_params_ext, epi=False)}"
        )
    else:
        cis = None
    return best_vs, cis

def format_parameters_per_tree(vs, input_params=None, epi=True, max_rhos_show=6):
    vs = np.asarray(vs, dtype=float)
    la, psi = vs[0], vs[1]
    rhos = vs[2:]
    rhos_str = ", ".join(f"{x:.4g}" for x in rhos[:max_rhos_show])
    if len(rhos) > max_rhos_show:
        rhos_str += f", ... ({len(rhos)} total)"
    fixed_str = ""
    if input_params is not None:
        mask = np.asarray(input_params, dtype=object) != None
        fixed_str = " | fixed=" + "".join("1" if m else "0" for m in mask)
    if epi:
        R0 = la / psi
        rt = 1 / psi
        return f"la={la:.6g}, psi={psi:.6g}, R0={R0:.6g}, inf_time={rt:.6g}, rhos=[{rhos_str}]{fixed_str}"
    return f"la={la:.6g}, psi={psi:.6g}, rhos=[{rhos_str}]{fixed_str}"


def main():
    """
    Entry point for tree/forest parameter estimation under the BD model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="BD model parameter estimator. "
                                            "The BD model is parameterised with three parameters: "
                                            "transmission rate la, removal rate psi, and sampling probability upon removal p. "
                                            "At least one of these parameters needs to be given as an input for identifiability.")
    parser.add_argument('--nwk', required=True, type=str,
                        help="input file in newick or nexus format, containing one or multiple transmission trees. "
                             "If multiple trees are provided, they are treated as having the same parameters.")
    parser.add_argument('--la', required=False, default=None, type=float,
                        help="transmission rate (if not provided, will be estimated)")
    parser.add_argument('--psi', required=False, default=None, type=float,
                        help="removal rate (if not provided, will be estimated)")
    parser.add_argument('--p', required=False, default=None, type=float,
                        help='sampling probability (if not provided, will be estimated)')
    parser.add_argument('--start_times', nargs='*', type=float,
                        help='If multiple trees are provided in the input file, their start times '
                             '(i.e., times at the beginning of their root branches) are by default considered to be equal. '
                             'If a different behaviour is needed, one should specify as many start times here '
                             'as there are trees in the input file.')
    parser.add_argument('--p_per_tree', required=False, action='store_true', help="If multiple trees are provided in the input file and there is sampling heterogeneity and p is not fixed, then it will estimate 1 sampling probability per tree in forest.")
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=3,
                        help="upper bounds for BD parameters: la psi p (all need to specified, even the fixed ones)",
                        default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=3,
                        help="lower bounds for BD parameters: la psi p (all need to specified, even the fixed ones)",
                        default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    forest = read_forest(params.nwk)
    # resolve_forest(forest)
    annotate_forest_with_time(forest, start_times=params.start_times)
    t_start = min(getattr(tree, TIME) - tree.dist for tree in forest)
    T = get_T(T=None, forest=forest)
    print('Read a forest of {} trees with {} tips in total, evolving between times {} and {}.'
          .format(len(forest), sum(len(_) for _ in forest), t_start, T))

    if params.p_per_tree:
        vs, cis = infer_per_tree_rho(forest, T=T, **vars(params))
    else:
        vs, cis = infer(forest, T, **vars(params))

    if params.p_per_tree:
        save_results_per_tree(vs, cis, params.log, ci=params.ci)
    else:
        save_results(vs, cis, params.log, ci=params.ci)


def save_results_per_tree(vs, cis, log, ci=False):
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, 'w+') as f:
        la, psi = vs[0], vs[1]
        rhos = vs[2:]
        n_rhos = len(rhos)
        rho_labels = [f"{SAMPLING_PROBABILITY} {i}" for i in range(1, n_rhos + 1)]
        labels = [REPRODUCTIVE_NUMBER, INFECTIOUS_TIME] + rho_labels + [TRANSMISSION_RATE, REMOVAL_RATE]
        label_line = ','.join(labels)
        f.write(f",{label_line}\n")
        R0 = la / psi
        rt = 1 / psi
        vals = [R0, rt] + rhos.tolist() + [la, psi]
        value_line = ",".join(f'{_:g}' for _ in vals)
        f.write(f"value,{value_line}\n")
        if ci:
            (la_min, la_max), (psi_min, psi_max) = cis[0], cis[1]
            rho_min = [cis[2:][i][0] for i in range(0,n_rhos)]
            rho_max = [cis[2:][i][1] for i in range(0,n_rhos)]
            R0_min, R0_max = la_min / psi, la_max / psi
            rt_min, rt_max = 1 / psi_max, 1 / psi_min
            mins = [R0_min, rt_min] + rho_min + [la_min, psi_min]
            ci_min_line = ",".join(f'{_:g}' for _ in mins)
            f.write(f"CI_min,{ci_min_line}\n")
            maxs = [R0_max, rt_max] + rho_max + [la_max, psi_max]
            ci_max_line = ",".join(f'{_:g}' for _ in maxs)
            f.write(f"CI_max,{ci_max_line}\n")

if '__main__' == __name__:
    main()
