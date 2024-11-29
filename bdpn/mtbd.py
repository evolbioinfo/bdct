import numpy as np


RTOL = 100 * np.finfo(np.float64).eps
from scipy.integrate import odeint

N_U_STEPS = int(1e7)


def find_index_within_bounds(sol, start, stop, upper_bound, lower_bound=0):
    """
    Find an index i in a given array sol (of shape n x m),
    such that all the values of sol[i, :] are withing the given bounds
    and at least one of them is above the lower bound.

    Note that such index might not be unique. The search starts with the middle of the array
    and if needed proceeds with the binary search in the lower half of the array.

    :param sol: an array where to search
    :param start: start position in the array (inclusive)
    :param stop: stop position in the array (exclusive)
    :param upper_bound: upper bound
    :param lower_bound: lower bound
    :return: the index i that satisfies the above conditions
    """
    i = start + ((stop - start) // 2)
    if i == start or stop <= start:
        return start
    value = sol[i, :]
    if np.all(value >= lower_bound) and np.all(value <= upper_bound) and np.any(value > lower_bound):
        return i
    return find_index_within_bounds(sol, start, i, upper_bound)


def find_time_index(v, tt, start, stop):
    """
    Searches for an index i in time array tt, such that tt[i] >= v > tt[i + 1], using a binary search.

    :param v: a time value for which the index is searched for
    :param tt: a time array [t_n, ..., t_0] such that t_{i + 1} > t_i.
    :param start: start index for the search (inclusive)
    :param stop: stop index for the search (exclusive)
    :return: an index i, such that tt[i] >= v > tt[i + 1].
    """
    i = start + ((stop - start) // 2)
    if i == start or i == stop - 1:
        return i
    if tt[i] >= v:
        if tt[i + 1] < v:
            return i
        return find_time_index(v, tt, i + 1, stop)
    if tt[i - 1] >= v:
        return i - 1
    return find_time_index(v, tt, start, i)


def compute_U(T, MU, LA, PSI, RHO, SIGMA, nsteps=N_U_STEPS):
    """
    Calculates a function get_U which for a given time t: 0 <= t <= T, would return
    an array of unobserved probabilities [U_1(t), ..., U_m(t)].

    U_k(t) are calculated by
    (1) solving their ODEs numerically for an array tt of nsteps times equally spaced between t=T and t=0,
    producing an array of solutions sol of length nstep (for corresponding times in tt)s.
    (2) creating a linear approximation which for a given time t (2a) find an index i such that tt[i] >= t > tt[i+1];
    (2b) returns sol[i + 1] + (sol[i] - sol[i + 1]) * (tt[i] - t) / (tt[i] - tt[i + 1]).


    :param T: time at end of the sampling period
    :param MU: an array of state transition rates
    :param LA: an array of transmission rates
    :param PSI: an array of removal rates
    :param RHO: an array of sampling probabilities
    :param SIGMA: an array of rate sums: MU.sum(axis=1) + LA.sum(axis=1) + PSI
    :return: a function that for a given time t returns the array of corresponding unsampled probabilities:
        t ->  [U_1(t), ..., U_m(t)].
    """
    tt = np.linspace(T, 0, nsteps)
    y0 = np.ones(LA.shape[0], np.float64)
    PSI_NOT_RHO = PSI * (1 - RHO)

    def pdf_U(U, t):
        dU = (SIGMA - LA.dot(U)) * U - MU.dot(U) - PSI_NOT_RHO
        return dU

    sol = odeint(pdf_U, y0, tt, rtol=RTOL)
    sol = np.maximum(sol, 0)

    def get_U(t):
        t = max(0, min(t, T))
        tt_len = len(tt)
        i = find_time_index(t, tt, 0, tt_len)
        sol_prev = sol[i, :]
        if i == (tt_len - 1) or t == tt[i]:
            return sol_prev
        sol_next = sol[i + 1, :]
        if t == tt[i + 1]:
            return sol_next
        return sol_next + (sol_prev - sol_next) * (tt[i] - t) / (tt[i] - tt[i + 1])

    return get_U


def get_P(ti, l, t0, get_U, MU, LA, SIGMA):
    """
    Calculates P_{kl}^{(i)}(t0) for k in 1:m, where the initial condition is specified at time ti >= t0 (time of node i):
    P_{kl}^{(i)}(ti) = 0 for all k=l;
    P_{ll}^{(i)}(ti) = 1.

    :param ti: time for the initial condition (at node i)
    :param l: state of node i (the only state for which the initial condition is non-zero)
    :param t0: time to calculate the values at (t0 <= ti)
    :param get_U: a function to calculate an array of unsampled probabilities for a given time: t -> [U_1, .., U_m]
    :param MU: an array of state transition rates
    :param LA: an array of transmission rates
    :param SIGMA:  an array of rate sums: MU.sum(axis=1) + LA.sum(axis=1) + PSI, where PSI is the array of removal rates
    :return: a tuple containing an array of (potentially rescaled) branch evolution probabilities at time t0:
        [CP^{(i)}_{0l}(t0), .., CP^{(i)}_{ml}(t0)] and a log of the scaling factor: logC
    """
    y0 = np.zeros(LA.shape[0], np.float64)
    y0[l] = 1

    if t0 == ti:
        return y0, 0

    def pdf_Pany_l(P, t):
        U = get_U(t)
        return (SIGMA - LA.dot(U)) * P - (MU + U * LA).dot(P)

    nsteps = 10
    tt = np.linspace(ti, t0, nsteps)
    sol = odeint(pdf_Pany_l, y0, tt, rtol=RTOL)

    if np.any(sol[-1, :] < 0) or np.all(sol[-1, :] == 0) or np.any(sol[-1, :] > 1):
        return get_P_Euler(ti, l, t0, get_U, MU, LA, SIGMA)

    # # If there was an underflow during P_{kl}^{(i)}(t) calculations, we find a time tt[i] before the problem happened
    # # and use its values sol[i, :] as new initial values for a rescaled ODE calculation,
    # # which we solve for CP_{kl}^{(i)}(t). The new initial values become:
    # # CP_{kl}^{(i)}(tt[i]) = C sol[i, k],
    # # where C = 1 / min_positive(sol[i, :]).
    # cs = [1]
    # while np.any(sol[-1, :] < 0) or np.all(sol[-1, :] == 0) or np.any(sol[-1, :] > np.prod(cs)):
    #     tzero = t0
    #     i = find_index_within_bounds(sol, 0, len(tt), np.prod(cs))
    #     while i == 0:
    #         if nsteps == 10000:
    #             return np.zeros(len(sol)), 0
    #         nsteps *= 10
    #         if (ti - tzero) <= 4 * np.finfo(np.float64).eps:
    #             tzero = tzero + (ti - tzero) / 2
    #         tt = np.linspace(ti, tzero, nsteps)
    #         sol = odeint(pdf_Pany_l, y0, tt, rtol=RTOL)
    #         i = find_index_within_bounds(sol, 0, len(tt), np.prod(cs), 0)
    #
    #     vs = sol[i, :]
    #
    #     # print(i, sol[i, :], np.all(sol[i, :] >= 0), np.any(sol[i, :] > 0))
    #     c = 1 / min(sol[i, sol[i, :] > 0])
    #     cs.append(c)
    #     y0 = vs * c
    #     ti = tt[i]
    #     nsteps = 100
    #     tt = np.linspace(ti, t0, nsteps)
    #     sol = odeint(pdf_Pany_l, y0, tt, rtol=RTOL)

    return np.maximum(sol[-1, :], 0), 0 #np.log(cs).sum()


def get_P_Euler(ti, l, t0, get_U, MU, LA, SIGMA):
    """
    Calculates P_{kl}^{(i)}(t0) for k in 1:m, where the initial condition is specified at time ti >= t0 (time of node i):
    P_{kl}^{(i)}(ti) = 0 for all k=l;
    P_{ll}^{(i)}(ti) = 1.

    :param ti: time for the initial condition (at node i)
    :param l: state of node i (the only state for which the initial condition is non-zero)
    :param t0: time to calculate the values at (t0 <= ti)
    :param get_U: a function to calculate an array of unsampled probabilities for a given time: t -> [U_1, .., U_m]
    :param MU: an array of state transition rates
    :param LA: an array of transmission rates
    :param SIGMA:  an array of rate sums: MU.sum(axis=1) + LA.sum(axis=1) + PSI, where PSI is the array of removal rates
    :return: a tuple containing an array of (potentially rescaled) branch evolution probabilities at time t0:
        [CP^{(i)}_{0l}(t0), .., CP^{(i)}_{ml}(t0)] and a log of the scaling factor: logC
    """

    print("Going Euler")
    y0 = np.zeros(LA.shape[0], np.float64)
    y0[l] = 1

    if t0 == ti:
        return y0, 0

    def pdf_Pany_l(P, t):
        U = get_U(t)
        return (SIGMA - LA.dot(U)) * P - (MU + U * LA).dot(P)

    dt = (ti - t0) / 10
    tau = np.inf

    cs = [1]
    while tau > 1e-6 and dt > 1e-3:
        dy_dt = pdf_Pany_l(y0, ti)
        yj = y0 - dt * dy_dt

        if np.any(yj < 0) or np.any(yj > 1) or np.all(yj == 0):
            dt /= 2
            # print('div', yj, dy_dt)
            continue

        yjj = y0 - dt / 2 * dy_dt
        yjj = yjj - dt / 2 * pdf_Pany_l(yjj, ti - dt / 2)
        tau = np.min(yjj - yj)
        if abs(tau) > 1e-6:
            dt = 0.9 * dt * min(2, max(0.3, np.power(1e-6 / 2 / abs(tau), 0.5)))
    # print('set dt to ', dt)

    tj = ti
    yj = np.array(y0)
    while tj > t0:
        yj_next = np.minimum(yj - dt * pdf_Pany_l(yj, tj), np.prod(cs))
        if np.any(yj_next < 0) or np.all(yj_next == 0):
            c = max(1 / min(yj[yj > 0]), 2)
            cs.append(c)
            yj *= c
            continue
        yj = yj_next
        tj -= dt
    # if len(cs) > 1:
    #     print(cs)
    #     exit(0)
    return np.maximum(yj, 0), np.log(cs).sum()