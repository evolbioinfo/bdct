import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize


def annotate_tree(tree):
    for n in tree.traverse('preorder'):
        if n.is_root():
            p_time = 0
        else:
            p_time = getattr(n.up, 'time')
        n.add_feature('time', p_time + n.dist)


def get_c1(la, psi, rho):
    return np.power(np.power((la - psi), 2) + 4 * la * psi * rho, 0.5)


def get_c2(la, psi, rho, T):
    c1 = get_c1(la, psi, rho)
    return (c1 + la - psi) / (c1 - la + psi) * np.exp(-T * c1)


def get_c3(la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    return c2 * np.exp(c1 * ti) + 1


def get_u(t, la, psi, rho, T):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    exp_t_c1 = np.exp(t * c1)
    return (la + psi + c1 * (c2 * exp_t_c1 - 1) / (c2 * exp_t_c1 + 1)) / (2 * la)


def get_p(t, la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    c3 = get_c3(la, psi, rho, T, ti)
    return (np.power(c3, 2) * np.exp(c1 * (t - ti))) / np.power(c2 * np.exp(c1 * t) + 1, 2)


def get_p_o(t, la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    c3 = get_c3(la, psi, rho, T, ti)
    return (c3 * np.exp(1/2 * (la + psi + c1) * (t - ti))) / (c2 * np.exp(c1 * t) + 1)


def get_p_nh(t, la, psi, ti):
    return np.exp((la + psi) * (t - ti))


def get_lu(i, la, psi, psi_n, rho, rho_n, T, attribute=True):
    ti = getattr(i, 'time')
    tj = ti - i.dist
    pi_tj = get_p(tj, la, psi, rho, T, ti)
    if i.is_leaf():
        return pi_tj * psi * rho * (1 - rho_n)
    i0, i1 = i.children
    lu_0 = getattr(i0, 'LU') if attribute else get_lu(i0, la, psi, psi_n, rho, rho_n, T, attribute)
    lu_1 = getattr(i1, 'LU') if attribute else get_lu(i1, la, psi, psi_n, rho, rho_n, T, attribute)
    branch_probs = lu_0 * lu_1
    if i0.is_leaf():
        branch_probs += get_ln(i0, la, psi, rho, rho_n) * get_lp(i1, i0, la, psi, psi_n, rho, rho_n, T, attribute)
    if i1.is_leaf():
        branch_probs += get_lp(i0, i1, la, psi, psi_n, rho, rho_n, T, attribute) * get_ln(i1, la, psi, rho, rho_n)
    return pi_tj * 2 * la * branch_probs


def get_ln(i, la, psi, rho, rho_n):
    ti = getattr(i, 'time')
    tj = ti - i.dist
    if i.is_leaf():
        return get_p_nh(tj, la, psi, ti) * psi * rho * rho_n
    raise ValueError('A non-leaf cannot be a notifier')


def get_lp(i, r, la, psi, psi_n, rho, rho_n, T, attribute=True):
    ti = getattr(i, 'time')
    tj = ti - i.dist
    tr = getattr(r, 'time')
    if i.is_leaf():
        if tr > ti:
            return 0
        return get_p_o(tj, la, psi, rho, T, tr) * np.exp(-psi_n * (ti - tr)) * psi_n
    if not attribute:
        i0, i1 = i.children
        branch_probs = get_lp(i0, r, la, psi, psi_n, rho, rho_n, T) * get_lu(i1, la, psi, psi_n, rho, rho_n, T, attribute) \
                       + get_lu(i0, la, psi, psi_n, rho, rho_n, T, attribute) * get_lp(i1, r, la, psi, psi_n, rho, rho_n, T)
        return get_p_o(tj, la, psi, rho, T, ti) * la * branch_probs
    a2c = getattr(i, 'C')
    return sum(c * get_lp(a, r, la, psi, psi_n, rho, rho_n, T, attribute) for (a, c) in a2c.items())


def get_c(i, a, la, psi, psi_n, rho, rho_n, T, attribute=True):
    ti = getattr(i, 'time')
    tj = ti - i.dist
    if i == a:
        return 1
    i_a, i_not_a = i.children
    if a not in i_a.iter_leaves():
        i_a, i_not_a = i_not_a, i_a
    c = getattr(i_a, 'C')[a] if attribute else get_c(i_a, a, la, psi, psi_n, rho, rho_n, T)
    return get_p_o(tj, la, psi, rho, T, ti) * la * get_lu(i_not_a, la, psi, psi_n, rho, rho_n, T, attribute) * c


def loglikelihood(tree, la, psi, psi_n, rho, rho_n, T):
    for n in tree.traverse('postorder'):
        if n.is_leaf():
            n.add_feature('LU', get_lu(n, la, psi, psi_n, rho, rho_n, T, False))
            n.add_feature('C', {n: 1})
            continue
        a2c = {a: get_c(n, a, la, psi, psi_n, rho, rho_n, T, True) for a in n}
        n.add_feature('C', a2c)
        n.add_feature('LU', get_lu(n, la, psi, psi_n, rho, rho_n, T, True))
    return np.log(getattr(tree, 'LU'))

    # return get_lu(tree, la, psi, psi_n, rho, rho_n, T)


def optimize_likelihood_params(tree, T, la=None, psi=None, psi_n=None, rho=None, rho_n=None):
    """
    Optimizes the likelihood parameters for a given forest and a given MTBD model.


    :param forest: a list of ete3.Tree trees, annotated with node states and times via features STATE_K and TI.
    :param T: time at end of the sampling period
    :param model: MTBD model containing starting parameter values
    :param optimise: MTBD model whose rates indicate which parameters need to optimized:
        positive rates correspond to optimized parameters
    :param u: number of hidden trees, where no tip got sampled
    :return: the values of optimised parameters and the corresponding loglikelihood: (MU, LA, PSI, RHO, best_log_lh)
    """

    bounds = []

    l = []
    for n in tree.traverse('preorder'):
        if n.dist:
            l.append(n.dist)
    max_rate = 10 / np.mean(l)
    min_rate = 1 / np.max(l)
    print('Considering ', max_rate, ' as max rate and ', min_rate, ' as min rate')
    avg_rate = (min_rate + max_rate) / 2

    bounds.extend([[min_rate, max_rate]] * (int(la is None) + int(psi is None) + int(psi_n is None)))
    bounds.extend([[1e-3, 1 - 1e-3]] * int(rho is None))
    bounds.extend([[1e-3, 1 - 1e-3]] * int(rho_n is None))
    bounds = np.array(bounds, np.float64)

    def get_real_params_from_optimised(ps):
        ps = np.maximum(np.minimum(ps, bounds[:, 1]), bounds[:, 0])
        result = np.zeros(5)
        i = 0
        if la is None:
            result[0] = ps[i]
            i += 1
        else:
            result[0] = la
        if psi is None:
            result[1] = ps[i]
            i += 1
        else:
            result[1] = psi
        if psi_n is None:
            result[2] = ps[i]
            i += 1
        else:
            result[2] = psi_n
        if rho is None:
            result[3] = ps[i]
            i += 1
        else:
            result[3] = rho
        if rho_n is None:
            result[4] = ps[i]
            i += 1
        else:
            result[4] = rho_n
        return result

    def get_optimised_params_from_real(ps):
        result = []
        if la is None:
            result.append(ps[0])
        if psi is None:
            result.append(ps[1])
        if psi_n is None:
            result.append(ps[2])
        if rho is None:
            result.append(ps[3])
        if rho_n is None:
            result.append(ps[4])
        return np.array(result)


    def get_v(ps):
        if np.any(np.isnan(ps)):
            return np.nan
        ps = get_real_params_from_optimised(ps)
        res = loglikelihood(tree, *ps, T)
        print("{}\t-->\t{:g}".format(ps, res))
        return -res

    x0 = get_optimised_params_from_real([avg_rate, avg_rate, avg_rate, 0.5, 0.5])
    best_log_lh = -get_v(x0)

    def R0(vs):
        vs = get_real_params_from_optimised(vs)
        return vs[0] / vs[1]

    cons = (NonlinearConstraint(R0, 0.2, 100), LinearConstraint(np.eye(len(x0)), bounds[:, 0], bounds[:, 1]),)

    for i in range(10):
        if i == 0:
            vs = x0
        else:
            keep_searching = True
            while keep_searching:
                keep_searching = False
                vs = np.random.uniform(bounds[:, 0], bounds[:, 1])
                for c in cons:
                    if not isinstance(c, LinearConstraint):
                        val = c.fun(vs)
                        if c.lb > val or c.ub < val:
                            keep_searching = True
                            break

        fres = minimize(get_v, x0=vs, method='COBYLA', bounds=bounds, constraints=cons)
        if fres.success and not np.any(np.isnan(fres.x)):
            if -fres.fun >= best_log_lh:
                x0 = np.array(fres.x)
                best_log_lh = -fres.fun
                break
        print('Attempt {} of trying to optimise the parameters: {}.'.format(i, -fres.fun))
    return get_real_params_from_optimised(x0)


nwk = '/home/azhukova/projects/bdpn/trees/bdpn/tree.0.nwk'
tree = Tree(nwk)
annotate_tree(tree)
df = pd.read_csv(nwk.replace('.nwk', '.log'))
rho = df.loc[0, 'sampling probability']
rho_n = df.loc[0, 'notification probability']
R0 = df.loc[0, 'R0']
it = df.loc[0, 'infectious time']
rt = df.loc[0, 'removal time after notification']
psi = 1 / it
psi_n = 1 / rt
la = R0 * psi
T = max(getattr(_, 'time') for _ in tree)
vs = optimize_likelihood_params(tree, T, la=None, psi=None, rho=rho, psi_n=None, rho_n=None)
print('Real params: {}'.format([la, psi, psi_n, rho, rho_n]))
print('Found params: {}'.format(vs))