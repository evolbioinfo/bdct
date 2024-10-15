import numpy as np

from bdpn.parameter_estimator import rescale_log, MIN_VALUE


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
    if diff <= 0:
        print(log_minuend, log_subtrahend)
        raise ValueError()
    return np.log(diff) - factors