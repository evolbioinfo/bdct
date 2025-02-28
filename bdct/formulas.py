import numpy as np

from bdct.parameter_estimator import rescale_log, MIN_VALUE

N2LOG_FACTORIAL = {0: 0, 1: 0, 2: np.log(2)}


EPSILON = 1e-6

N_INTERVALS = 10000


def get_tt_log(T):
    logT = np.log(T + 1)
    logdt = logT / N_INTERVALS
    tt = np.maximum((T - (np.exp(np.arange(0, N_INTERVALS + 1) * logdt) - 1))[::-1], 0)

    def get_value(array, t_targ):
        if t_targ <= 0:
            return array[0] if len(array.shape) == 1 else array[:, 0]
        if t_targ >= T:
            return array[-1] if len(array.shape) == 1 else array[:, -1]
        i = len(tt) - 1 - int(np.log(T + 1 - t_targ) // logdt)
        t, t_prev = tt[i], tt[i - 1]
        value_t, value_t_prev = (array[i], array[i - 1]) if len(array.shape) == 1 else (array[:, i], array[:, i - 1])

        log_prev = np.log(T + 1 - t_prev)
        log_t = np.log(T + 1 - t)
        log_targ = np.log(T + 1 - t_targ)
        return value_t_prev + (value_t - value_t_prev) * (log_targ - log_prev) / (log_t - log_prev)

    return tt, get_value


def get_tt_normal(T):
    dt = T / N_INTERVALS
    tt = np.arange(0, N_INTERVALS + 1) * dt

    def get_value(array, t_targ):
        if t_targ <= 0:
            return array[0] if len(array.shape) == 1 else array[:, 0]
        if t_targ >= T:
            return array[-1] if len(array.shape) == 1 else array[:, -1]
        i = int((t_targ + dt) // dt)
        t, t_prev = tt[i], tt[i - 1]
        value_t, value_t_prev = (array[i], array[i - 1]) if len(array.shape) == 1 else (array[:, i], array[:, i - 1])
        return value_t_prev + (value_t - value_t_prev) * (t_targ - t_prev) / (t - t_prev)

    return tt, get_value


def get_tt(T, as_log=False):
    return get_tt_log(T) if as_log else get_tt_normal(T)


def log_factorial(n):
    if n not in N2LOG_FACTORIAL:
        N2LOG_FACTORIAL[n] = np.log(n) + log_factorial(n - 1)
    return N2LOG_FACTORIAL[n]


def get_c1(la, psi, rho):
    """
    Return c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    """
    return np.power(np.power(la - psi, 2) + 4 * la * psi * rho, 0.5)


def get_c2(la, psi, c1):
    """
    Return c2 = (c1 + la - psi) / (c1 - la + psi)
    """
    la_minus_psi = la - psi
    return (c1 + la_minus_psi) / (c1 - la_minus_psi)


def get_E(c1, c2, t, T):
    """
    Returns E = c2 * exp(c1 * (t - T))
    """
    return c2 * np.exp(c1 * (t - T))


def get_u(la, psi, c1, E_t):
    """
    Returns U(t), where
    dU(t)/dt = (la + psi) * U(t) − la U^2(t) − psi * (1 − rho)
    U(T) = 1

    :param c1: c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    :param E_t: E_t = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    """
    two_la = 2 * la
    return (la + psi) / two_la + c1 / two_la * ((E_t - 1) / (E_t + 1))


def get_log_p(c1, t, ti, E_t, E_ti):
    """
    Returns log(p(t)), where
    dp(t)/dt = (la + psi) * p(t) − 2 * la * p(t) * U(t)
    p(ti) = 1
    dU(t)/dt = (la + psi) * U(t) − la U^2(t) − psi * (1 − rho)
    U(T) = 1.

    :param c1: c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    :param E_t: E_t = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    :param E_ti: E_ti = c2 * exp(c1 * (ti - T))
    """
    return 2 * (np.log(E_ti + 1) - np.log(E_t + 1)) + c1 * (t - ti)


def get_log_ppb(la, psi, c1, t, ti, E_t, E_ti):
    """
    Returns log(po(t)), where
    dpo(t)/dt = (la + psi) * po(t) − la * po(t) * U(t)
    po(ti) = 1
    dU(t)/dt = (la + psi) * U(t) − la U^2(t) − psi * (1 − rho)
    U(T) = 1.

    :param c1: c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    :param E_t: E_t = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    :param E_ti: E_ti = c2 * exp(c1 * (ti - T))
    """
    return np.log(E_ti + 1) - np.log(E_t + 1) + (c1 + la + psi) * (t - ti) / 2


def get_log_ppb_from_p_pn(log_p, log_pn):
    """
    Returns log(po(t)), where
    dpo(t)/dt = (la + psi) * po(t) − la * po(t) * U(t)
    po(ti) = 1
    if log(p(t)) and log(pn(t)) are known, where
    dpn(t)/dt = (la + psi) * pnh(t)
    pn(ti) = 1,
    dp(t)/dt = (la + psi) * p(t) − 2 * la * p(t) * U(t)
    p(ti) = 1
    """
    return (log_p + log_pn) / 2


def get_log_pn(la, psi, t, ti):
    """
    Returns log(pn(t)), where
    dpn(t)/dt = (la + psi) * pn(t)
    pn(ti) = 1
    """
    return get_log_no_event(la + psi, t, ti)


def get_log_pb(la, phi, t, ti):
    """
    Returns log(pb(t)), where
    dpb(t)/dt = (la + phi) * pb(t)
    pb(ti) = 1
    """
    return get_log_no_event(la + phi, t, ti)


def get_log_no_event(rate, t, ti):
    """
    Returns log(pne(t)), where
    dpne(t)/dt = rate * pne(t)
    pne(ti) = 1
    """
    return -rate * (ti - t)


def get_log_ppa(la, psi, phi, c1, t, ti, E_t, E_ti):
    """
    Returns log(ppb(t)e^(-phi (ti - t))/e^(-psi (ti - t)))
    """
    log_ppb = get_log_ppb(la, psi, c1, t, ti, E_t, E_ti)
    return get_log_ppa_from_ppb(log_ppb, psi, phi, t, ti)


def get_log_ppa_from_ppb(log_ppb, psi, phi, t, ti):
    """
    Returns log(po(t)e^(-phi (ti - t))/e^(-psi (ti - t)) )
    """
    return log_ppb + (psi - phi) * (ti - t)


def log_sum(log_summands):
    """
    Takes [logX1, ..., logXk] as input and returns log(X1 + ... + Xk) as output,
    while taking care of potential under/overflow.

    :param log_summands: an array of summands in log form
    :return: log of the sum
    """
    result = np.array(log_summands, dtype=np.float64)
    factors = rescale_log(result)
    return np.log(np.sum(np.exp(result))) - factors


def log_subtraction(log_minuend, log_subtrahend):
    """
    Takes logX1 and logX2 as input and returns log(X1 - X2) as output,
    while taking care of potential under/overflow.

    :param log_minuend: logX1 in the formula above
    :param log_subtrahend: logX2 in the formula above
    :return: log of the difference
    """
    result = np.array([log_minuend, log_subtrahend], dtype=np.float64)
    factors = rescale_log(result)
    diff = np.sum(np.exp(result) * [1, -1])
    if diff == 0 and log_minuend - log_subtrahend > 0:
        if factors < 0:
            factors = 0
        return MIN_VALUE - factors
    return np.log(diff) - factors

def prob_event1_before_other_events(rate1, *other_rates):
    """
    Calculates the probability that an event occurring at a constant rate rate1 happens
    before any of the events occurring at a constant rates other_rates.

    :param rate1: rate of the first event
    :param other_rates: rates of later events
    :return: probability described above
    """
    return rate1 / (rate1 + sum(other_rates))
