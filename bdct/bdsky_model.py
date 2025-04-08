import os
import numpy as np
import pandas as pd

from formulas import get_c1, get_c2, get_E, get_log_p, get_u, log_factorial
from parameter_estimator import optimize_likelihood_params, estimate_cis
from tree_manager import TIME, read_forest, annotate_forest_with_time, get_T

# Reuse constants from BD model
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


def rates2epi(params, n_intervals=1):
    """
    Transforms [la_1, ..., la_n, psi_1, ..., psi_n, rho_1, ..., rho_n, t_1, ..., t_{n-1}]
    to [Re_1, ..., Re_n, d_inf_1, ..., d_inf_n, rho_1, ..., rho_n, t_1, ..., t_{n-1}]

    :param params: Flattened parameter vector for rates
    :param n_intervals: Number of intervals in the skyline model
    :return: Flattened parameter vector for epidemiological parameters
    """
    if n_intervals == 1 and len(params) == 3:
        # Handle original case for backward compatibility
        la, psi, rho = params
        return np.array([la / psi, 1 / psi, rho])

    # Extract parameters
    la_values = params[:n_intervals]
    psi_values = params[n_intervals:2 * n_intervals]
    rho_values = params[2 * n_intervals:3 * n_intervals]

    # Calculate epidemiological parameters
    Re_values = [la / psi for la, psi in zip(la_values, psi_values)]
    d_inf_values = [1 / psi for psi in psi_values]

    # Time points (if any)
    time_points = [] if len(params) <= 3 * n_intervals else params[3 * n_intervals:]

    # Construct epi parameter vector
    epi_params = []
    epi_params.extend(Re_values)
    epi_params.extend(d_inf_values)
    epi_params.extend(rho_values)
    epi_params.extend(time_points)

    return np.array(epi_params)


def epi2rates(params, n_intervals=1):
    """
    Transforms [Re_1, ..., Re_n, d_inf_1, ..., d_inf_n, rho_1, ..., rho_n, t_1, ..., t_{n-1}]
    to [la_1, ..., la_n, psi_1, ..., psi_n, rho_1, ..., rho_n, t_1, ..., t_{n-1}]

    :param params: Flattened parameter vector for epidemiological parameters
    :param n_intervals: Number of intervals in the skyline model
    :return: Flattened parameter vector for rates
    """
    if n_intervals == 1 and len(params) == 3:
        # Handle original case for backward compatibility
        Re, d_i, rho = params
        return np.array([Re / d_i, 1 / d_i, rho])

    # Extract parameters
    Re_values = params[:n_intervals]
    d_inf_values = params[n_intervals:2 * n_intervals]
    rho_values = params[2 * n_intervals:3 * n_intervals]

    # Calculate rates
    psi_values = [1 / d_inf for d_inf in d_inf_values]
    la_values = [Re * psi for Re, psi in zip(Re_values, psi_values)]

    # Time points (if any)
    time_points = [] if len(params) <= 3 * n_intervals else params[3 * n_intervals:]

    # Construct rate parameter vector
    rate_params = []
    rate_params.extend(la_values)
    rate_params.extend(psi_values)
    rate_params.extend(rho_values)
    rate_params.extend(time_points)

    return np.array(rate_params)


def get_start_parameters(forest, la=None, psi=None, rho=None):
    """Reuse from BD model to estimate starting parameters"""
    la_is_fixed = la is not None and la > 0
    psi_is_fixed = psi is not None and psi > 0
    rho_is_fixed = rho is not None and 0 < rho <= 1

    rho_est = rho if rho_is_fixed else 0.5

    if la_is_fixed and psi_is_fixed:
        return np.array([la, psi, rho_est], dtype=np.float64)

    # Let's estimate transmission time as a median internal branch length
    # and sampling time as a median external branch length
    internal_dists, external_dists = [], []
    for tree in forest:
        for n in tree.traverse():
            if n.is_root() and not n.dist:
                continue
            (internal_dists if not n.is_leaf() else external_dists).append(n.dist)

    psi_est = psi if psi_is_fixed else 1 / np.median(external_dists)
    # if it is a corner case when we only have tips, let's use sampling times
    la_est = la if la_is_fixed else ((1 / np.median(internal_dists)) if internal_dists else 1.1 * psi_est)
    if la_est <= psi_est:
        if la_is_fixed:
            psi_est = la_est * 0.9
        else:
            la_est *= psi_est * 1.1

    return np.array([la_est, psi_est, rho_est], dtype=np.float64)


def loglikelihood(forest, *parameters, T, threads=1, u=-1, n_intervals=1):
    """
    Calculate log-likelihood for Birth-Death Skyline model with different parameters for different time intervals.

    :param forest: list of one or more trees
    :param parameters: flattened parameters vector [la_1, ..., la_n, psi_1, ..., psi_n, rho_1, ..., rho_n, t_1, ... t_{n-1}]
    :param T: the total time span
    :param threads: number of threads to use
    :param u: number of hidden trees (estimated by default)
    :param n_intervals: number of intervals in the model
    :return: log-likelihood value
    """
    # Reconstruct models from parameters vector
    # For n intervals we should have 3*n + (n-1) parameters
    # [la_1, ..., la_n, psi_1, ..., psi_n, rho_1, ..., rho_n, t_1, ... t_{n-1}]

    if len(parameters) != 3 * n_intervals + (n_intervals - 1) and not (n_intervals == 1 and len(parameters) == 3):
        raise ValueError(
            f"Expected {3 * n_intervals + (n_intervals - 1)} parameters for {n_intervals} intervals, got {len(parameters)}")

    # Split parameters into respective arrays
    la_values = parameters[:n_intervals]
    psi_values = parameters[n_intervals:2 * n_intervals]
    rho_values = parameters[2 * n_intervals:3 * n_intervals]

    # Time points (if more than one interval)
    time_points = []
    if n_intervals > 1:
        time_points = parameters[3 * n_intervals:]
        # Ensure time points are sorted
        if not all(time_points[i] < time_points[i + 1] for i in range(len(time_points) - 1)):
            raise ValueError("Time points must be in ascending order")
        # Ensure last time point is less than T
        if time_points[-1] >= T:
            raise ValueError(f"Last time point {time_points[-1]} must be less than T={T}")

    # Construct models list
    models = []

    # Add all intervals except the last one
    for i in range(n_intervals - 1):
        interval_end = time_points[i]
        models.append((interval_end, la_values[i], psi_values[i], rho_values[i]))

    # Add the last interval (ending at T)
    models.append((T, la_values[-1], psi_values[-1], rho_values[-1]))

    # Sort models by interval_end to ensure they're in chronological order
    models = sorted(models, key=lambda x: x[0])

    # Ensure the last interval ends at T
    if models[-1][0] != T:
        raise ValueError(f"The last interval should end at T={T}, but it ends at {models[-1][0]}")

    # Function to determine which model applies at a given time
    def get_model_for_time(t):
        for i, (interval_end, la, psi, rho) in enumerate(models):
            if t <= interval_end:
                return i, la, psi, rho
        return len(models) - 1, *models[-1][1:]  # Default to the last model if somehow t > T

    # Calculate hidden likelihood recursively backwards through time
    # First, prepare arrays for all models
    intervals = len(models)
    hidden_lk = [0] * intervals
    c1_values = [0] * intervals
    c2_values = [0] * intervals

    # Calculate c1 for all intervals
    for i, (interval_end, la_i, psi_i, rho_i) in enumerate(models):
        c1_values[i] = get_c1(la=la_i, psi=psi_i, rho=rho_i)

    # Set the last interval's hidden likelihood to 1
    hidden_lk[intervals - 1] = 1

    # Work backwards through the intervals
    for i in reversed(range(intervals - 1)):
        # Get current interval parameters
        interval_end, la_i, psi_i, rho_i = models[i]

        # Calculate c2 using the next model's hidden likelihood as C
        c2_values[i] = get_c2(C=hidden_lk[i + 1], la=la_i, psi=psi_i, c1=c1_values[i])

        # Calculate hidden likelihood for this interval
        prev_time = models[i - 1][0] if i > 0 else 0
        hidden_lk[i] = get_u(la_i, psi_i, c1_values[i],
                             E_t=get_E(c1=c1_values[i], c2=c2_values[i],
                                       t=prev_time, T=interval_end))

    # Calculate c2 for the last interval using C=1
    c2_values[intervals - 1] = get_c2(C=1, la=models[intervals - 1][1],
                                      psi=models[intervals - 1][2],
                                      c1=c1_values[intervals - 1])

    # Use the first interval's hidden likelihood for the final calculation
    hidden_lk_final = hidden_lk[0]

    # Avoid division by zero when hidden_lk_final is close to 1
    if hidden_lk_final:
        if hidden_lk_final >= 0.9999:  # Use a threshold close to 1 to avoid numerical issues
            # In this case, use a very large but finite value
            u = 1e6 if u is None or u < 0 else u
        else:
            u = len(forest) * hidden_lk_final / (1 - hidden_lk_final) if u is None or u < 0 else u
        res = u * np.log(hidden_lk_final)
    else:
        res = 0

    # Process each tree in the forest
    for tree in forest:
        # Traverse the tree in preorder (from root to leaves)
        for n in tree.traverse('preorder'):
            t = getattr(n, TIME)
            model_idx, la, psi, rho = get_model_for_time(t)

            if n.is_leaf():
                # Add contribution for leaf nodes (sampling)
                res += np.log(psi) + np.log(rho)
            else:
                # Add contribution for internal nodes (transmission)
                num_children = len(n.children)
                res += log_factorial(num_children) + (num_children - 1) * np.log(la)

                # Process each child branch
                for child in n.children:
                    ti = getattr(child, TIME)
                    child_model_idx, _, _, _ = get_model_for_time(ti)

                    # Check if this branch crosses model boundaries
                    if model_idx == child_model_idx:
                        # No boundary crossing, use single model
                        c1 = get_c1(la=la, psi=psi, rho=rho)
                        c2 = get_c2(C=hidden_lk[model_idx], la=la, psi=psi, c1=c1)
                        E_t = get_E(c1=c1, c2=c2, t=t, T=T)
                        E_ti = get_E(c1=c1, c2=c2, t=ti, T=T)
                        res += get_log_p(c1, t, ti=ti, E_t=E_t, E_ti=E_ti)
                    else:
                        # Branch crosses model boundaries
                        # We need to split the branch at each boundary it crosses
                        current_t = t
                        current_model_idx = model_idx

                        # Process each segment of the branch
                        while current_model_idx != child_model_idx:
                            # Get current model parameters
                            interval_end, current_la, current_psi, current_rho = models[current_model_idx]
                            next_model_idx = current_model_idx + 1

                            # Calculate contribution for this segment
                            c1 = get_c1(la=current_la, psi=current_psi, rho=current_rho)

                            # Use hidden_lk of the next model as C
                            next_hidden_lk = hidden_lk[next_model_idx]
                            c2 = get_c2(C=next_hidden_lk, la=current_la, psi=current_psi, c1=c1)

                            E_t = get_E(c1=c1, c2=c2, t=current_t, T=T)
                            E_interval_end = get_E(c1=c1, c2=c2, t=interval_end, T=T)

                            # Add log-likelihood for this segment
                            res += get_log_p(c1, current_t, ti=interval_end, E_t=E_t, E_ti=E_interval_end)

                            # Move to the next segment
                            current_t = interval_end
                            current_model_idx = next_model_idx

                        # Process the final segment (to the child)
                        final_la, final_psi, final_rho = models[current_model_idx][1:]
                        c1 = get_c1(la=final_la, psi=final_psi, rho=final_rho)
                        c2 = get_c2(C=hidden_lk[current_model_idx], la=final_la, psi=final_psi, c1=c1)
                        E_t = get_E(c1=c1, c2=c2, t=current_t, T=T)
                        E_ti = get_E(c1=c1, c2=c2, t=ti, T=T)
                        res += get_log_p(c1, current_t, ti=ti, E_t=E_t, E_ti=E_ti)

        # Process the root branch
        root_ti = getattr(tree, TIME)
        root_t = root_ti - tree.dist

        # Figure out which models apply to the root branch
        root_start_model_idx, root_start_la, root_start_psi, root_start_rho = get_model_for_time(root_t)
        root_end_model_idx, _, _, _ = get_model_for_time(root_ti)

        if root_start_model_idx == root_end_model_idx:
            # Root branch is within a single model interval
            c1 = get_c1(la=root_start_la, psi=root_start_psi, rho=root_start_rho)
            c2 = get_c2(C=hidden_lk[root_start_model_idx], la=root_start_la, psi=root_start_psi, c1=c1)
            E_t = get_E(c1=c1, c2=c2, t=root_t, T=T)
            E_ti = get_E(c1=c1, c2=c2, t=root_ti, T=T)
            res += get_log_p(c1, root_t, ti=root_ti, E_t=E_t, E_ti=E_ti)
        else:
            # Root branch crosses model boundaries
            current_t = root_t
            current_model_idx = root_start_model_idx

            while current_model_idx != root_end_model_idx:
                interval_end, current_la, current_psi, current_rho = models[current_model_idx]
                next_model_idx = current_model_idx + 1

                c1 = get_c1(la=current_la, psi=current_psi, rho=current_rho)

                # Use hidden_lk of the next model as C
                next_hidden_lk = hidden_lk[next_model_idx]
                c2 = get_c2(C=next_hidden_lk, la=current_la, psi=current_psi, c1=c1)

                E_t = get_E(c1=c1, c2=c2, t=current_t, T=T)
                E_interval_end = get_E(c1=c1, c2=c2, t=interval_end, T=T)

                res += get_log_p(c1, current_t, ti=interval_end, E_t=E_t, E_ti=E_interval_end)

                current_t = interval_end
                current_model_idx = next_model_idx

            # Process the final segment of the root branch
            final_la, final_psi, final_rho = models[current_model_idx][1:]
            c1 = get_c1(la=final_la, psi=final_psi, rho=final_rho)
            c2 = get_c2(C=hidden_lk[current_model_idx], la=final_la, psi=final_psi, c1=c1)
            E_t = get_E(c1=c1, c2=c2, t=current_t, T=T)
            E_ti = get_E(c1=c1, c2=c2, t=root_ti, T=T)
            res += get_log_p(c1, current_t, ti=root_ti, E_t=E_t, E_ti=E_ti)

    return res


def infer_skyline(forest, T, n_intervals=2, la=None, psi=None, p=None, times=None,
                  lower_bounds=DEFAULT_LOWER_BOUNDS, upper_bounds=DEFAULT_UPPER_BOUNDS, ci=False, threads=1,
                  num_attemps=3,
                  **kwargs):
    """
    Infers BDSKY model parameters from a given forest.

    :param forest: list of one or more trees
    :param T: total time span
    :param n_intervals: number of intervals in the skyline model
    :param la: transmission rate(s) - either a single value for all intervals or a list of values
    :param psi: removal rate(s) - either a single value for all intervals or a list of values
    :param p: sampling probability(ies) - either a single value for all intervals or a list of values
    :param times: time points defining interval boundaries - should be n_intervals-1 values
    :param lower_bounds: array of lower bounds for parameter values (la, psi, p)
    :param upper_bounds: array of upper bounds for parameter values (la, psi, p)
    :param ci: whether to calculate the CIs or not
    :return: tuple(vs, cis) of estimated parameter values vs=[la_1, ..., la_n, psi_1, ..., psi_n, rho_1, ..., rho_n, t_1, ..., t_{n-1}]
        and CIs ci=[[la_1_min, la_1_max], ..., [t_{n-1}_min, t_{n-1}_max]].
        In the case when CIs were not set to be calculated,
        their values would correspond exactly to the parameter values.
    """
    if la is None and psi is None and p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    # Convert single values to lists of appropriate length
    def ensure_list(param, length):
        if param is None:
            return [None] * length
        if not isinstance(param, (list, tuple, np.ndarray)):
            return [param] * length
        if len(param) != length:
            raise ValueError(f"Expected {length} values for parameter, got {len(param)}")
        return list(param)

    # Ensure parameter lists have the right length
    la_list = ensure_list(la, n_intervals)
    psi_list = ensure_list(psi, n_intervals)
    p_list = ensure_list(p, n_intervals)

    # For times, if not provided, we'll set them to None and they'll be optimized
    if times is None:
        times_list = [None] * (n_intervals - 1)
    else:
        if len(times) != n_intervals - 1:
            raise ValueError(f"Expected {n_intervals - 1} time points for {n_intervals} intervals, got {len(times)}")
        times_list = list(times)

    # Create the input parameters vector
    input_params = []
    input_params.extend(la_list)
    input_params.extend(psi_list)
    input_params.extend(p_list)
    input_params.extend(times_list)

    # Create bounds for all parameters
    bounds = []
    for i in range(n_intervals):
        # Bounds for la
        bounds.append([lower_bounds[0], upper_bounds[0]])

    for i in range(n_intervals):
        # Bounds for psi
        bounds.append([lower_bounds[1], upper_bounds[1]])

    for i in range(n_intervals):
        # Bounds for p (rho)
        bounds.append([lower_bounds[2], upper_bounds[2]])

    for i in range(n_intervals - 1):
        # Bounds for time points (between 0 and T)
        bounds.append([0, T])

    bounds = np.array(bounds)

    # Create start parameters
    # For rates and probabilities, we'll use the BD start parameters logic
    bd_start = get_start_parameters(forest, la=la_list[0], psi=psi_list[0], rho=p_list[0])

    start_parameters = []
    # la values
    for i in range(n_intervals):
        start_parameters.append(bd_start[0])
    # psi values
    for i in range(n_intervals):
        start_parameters.append(bd_start[1])
    # rho values
    for i in range(n_intervals):
        start_parameters.append(bd_start[2])

    # For time points, distribute them evenly in [0, T]
    for i in range(n_intervals - 1):
        if times_list[i] is not None:
            start_parameters.append(times_list[i])
        else:
            # Distribute time points evenly
            start_parameters.append((i + 1) * T / n_intervals)

    start_parameters = np.array(start_parameters)

    print(f'Lower bounds are set to:\t{format_parameters_skyline(bounds[:, 0], n_intervals)}')
    print(f'Upper bounds are set to:\t{format_parameters_skyline(bounds[:, 1], n_intervals)}\n')
    print(
        f'Starting BDSKY parameters:\t{format_parameters_skyline(start_parameters, n_intervals, fixed=input_params)}\tloglikelihood=N/A')

    # Wrap the loglikelihood function to include n_intervals
    def loglikelihood_wrapper(forest, *parameters, T=T, threads=threads, u=-1):
        return loglikelihood(forest, *parameters, T=T, threads=threads, u=u, n_intervals=n_intervals)

    vs, lk = optimize_likelihood_params(forest, T=T, input_parameters=input_params,
                                        loglikelihood_function=loglikelihood_wrapper, bounds=bounds,
                                        start_parameters=start_parameters, threads=threads,
                                        formatter=lambda _: format_parameters_skyline(_, n_intervals),
                                        num_attemps=num_attemps)

    print(f'Estimated BDSKY parameters:\t{format_parameters_skyline(vs, n_intervals)};\tloglikelihood={lk}')

    if ci:
        cis = estimate_cis(T, forest, input_parameters=input_params, loglikelihood_function=loglikelihood_wrapper,
                           optimised_parameters=vs, bounds=bounds, threads=threads)
        print(f'Estimated CIs:\n\tlower:\t{format_parameters_skyline(cis[:, 0], n_intervals)}\n'
              f'\tupper:\t{format_parameters_skyline(cis[:, 1], n_intervals)}')
    else:
        cis = None

    return vs, cis


def format_parameters_skyline(params, n_intervals, fixed=None, epi=True):
    """Format BDSKY parameters for display"""
    result = []

    # Extract parameters
    la_values = params[:n_intervals]
    psi_values = params[n_intervals:2 * n_intervals]
    rho_values = params[2 * n_intervals:3 * n_intervals]

    # Extract fixed indicators if provided
    if fixed is not None:
        fixed_la = fixed[:n_intervals]
        fixed_psi = fixed[n_intervals:2 * n_intervals]
        fixed_rho = fixed[2 * n_intervals:3 * n_intervals]
        fixed_times = fixed[3 * n_intervals:] if len(fixed) > 3 * n_intervals else []

    # Format rate and probability parameters
    for i in range(n_intervals):
        # Lambda (transmission rate)
        if fixed is None:
            result.append(f"λ_{i + 1}={la_values[i]:.6f}")
        else:
            result.append(f"λ_{i + 1}={la_values[i]:.6f}{'' if fixed_la[i] is None else ' (fixed)'}")

        # Psi (removal rate)
        if fixed is None:
            result.append(f"ψ_{i + 1}={psi_values[i]:.6f}")
        else:
            result.append(f"ψ_{i + 1}={psi_values[i]:.6f}{'' if fixed_psi[i] is None else ' (fixed)'}")

        # Rho (sampling probability)
        if fixed is None:
            result.append(f"ρ_{i + 1}={rho_values[i]:.6f}")
        else:
            result.append(f"ρ_{i + 1}={rho_values[i]:.6f}{'' if fixed_rho[i] is None else ' (fixed)'}")

        # Add epidemiological parameters if requested
        if epi:
            # R0
            R0 = la_values[i] / psi_values[i]
            if fixed is None:
                result.append(f"R0_{i + 1}={R0:.6f}")
            else:
                result.append(f"R0_{i + 1}={R0:.6f}{'' if fixed_la[i] is None or fixed_psi[i] is None else ' (fixed)'}")

            # Infectious time
            inf_time = 1 / psi_values[i]
            if fixed is None:
                result.append(f"1/ψ_{i + 1}={inf_time:.6f}")
            else:
                result.append(f"1/ψ_{i + 1}={inf_time:.6f}{'' if fixed_psi[i] is None else ' (fixed)'}")

    # Format time points
    if len(params) > 3 * n_intervals:
        time_points = params[3 * n_intervals:]
        for i, t in enumerate(time_points):
            if fixed is None:
                result.append(f"t_{i + 1}={t:.6f}")
            else:
                result.append(
                    f"t_{i + 1}={t:.6f}{'' if i >= len(fixed_times) or fixed_times[i] is None else ' (fixed)'}")

    return ', '.join(result)


def save_results_skyline(vs, cis, log, n_intervals, ci=False):
    """Save BDSKY results to a CSV file"""
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)

    # Extract parameters
    la_values = vs[:n_intervals]
    psi_values = vs[n_intervals:2 * n_intervals]
    rho_values = vs[2 * n_intervals:3 * n_intervals]

    if len(vs) > 3 * n_intervals:
        time_points = vs[3 * n_intervals:]
    else:
        time_points = []

    # Create dataframe columns
    columns = []
    data = {'parameter': ['value']}

    if ci:
        data['parameter'].extend(['CI_min', 'CI_max'])

    # Add parameters for each interval
    for i in range(n_intervals):
        # Basic parameters
        columns.extend([
            f'lambda_{i + 1}', f'psi_{i + 1}', f'rho_{i + 1}',
            f'R0_{i + 1}', f'infectious_time_{i + 1}'
        ])

        # Calculate derived parameters
        R0 = la_values[i] / psi_values[i]
        inf_time = 1 / psi_values[i]

        # Add values
        row_data = [la_values[i], psi_values[i], rho_values[i], R0, inf_time]

        # Add to data dictionary
        for j, col in enumerate(columns[-5:]):
            data[col] = [row_data[j]]

        # Add CIs if requested
        if ci:
            (la_min, la_max) = cis[i, :]
            (psi_min, psi_max) = cis[n_intervals + i, :]
            (rho_min, rho_max) = cis[2 * n_intervals + i, :]

            # Calculate derived CIs
            R0_min, R0_max = la_min / psi_max, la_max / psi_min  # Conservative approach
            inf_time_min, inf_time_max = 1 / psi_max, 1 / psi_min

            data[f'lambda_{i + 1}'].extend([la_min, la_max])
            data[f'psi_{i + 1}'].extend([psi_min, psi_max])
            data[f'rho_{i + 1}'].extend([rho_min, rho_max])
            data[f'R0_{i + 1}'].extend([R0_min, R0_max])
            data[f'infectious_time_{i + 1}'].extend([inf_time_min, inf_time_max])

    # Add time points
    for i, t in enumerate(time_points):
        col = f't_{i + 1}'
        columns.append(col)
        data[col] = [t]

        if ci and len(cis) > 3 * n_intervals + i:
            (t_min, t_max) = cis[3 * n_intervals + i, :]
            data[col].extend([t_min, t_max])


def main():
    """
    Entry point for tree parameter estimation with the BDSKY model with command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Estimate BDSKY parameters.")

    # Required arguments
    parser.add_argument('--nwk', required=True, type=str, help="Input tree file")
    parser.add_argument('--log', required=True, type=str, help="Output log file")

    # Optional fixed parameters - can be single values or lists
    parser.add_argument('--la', required=False, type=float, nargs='+',
                        help="Transmission rate(s) - either a single value for all intervals or one per interval")
    parser.add_argument('--psi', required=False, type=float, nargs='+',
                        help="Removal rate(s) - either a single value for all intervals or one per interval")
    parser.add_argument('--p', required=False, type=float, nargs='+',
                        help="Sampling probability(ies) - either a single value for all intervals or one per interval")
    parser.add_argument('--times', required=False, type=float, nargs='+',
                        help="Time points specifying interval boundaries (determines number of intervals)")

    # Parameter estimation options
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=3,
                        help="Upper bounds for parameters (la, psi, p)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=3,
                        help="Lower bounds for parameters (la, psi, p)", default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="Calculate the confidence intervals")
    parser.add_argument('--threads', required=False, type=int, default=1, help="Number of threads to use")
    parser.add_argument('--attempts', required=False, type=int, default=3,
                        help="Number of optimization attempts with different starting values")

    params = parser.parse_args()

    # Determine number of intervals from time points
    if params.times is not None:
        n_intervals = len(params.times) + 1
    else:
        # Default to 1 interval if no time points are provided
        n_intervals = 1

    # Ensure at least one parameter is specified
    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    # Load the forest
    forest = read_forest(params.nwk)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)
    print(
        f'Read a forest of {len(forest)} trees with {sum(len(_) for _ in forest)} tips in total, evolving over time {T}')

    # Process parameter values
    la = params.la
    psi = params.psi
    p = params.p
    times = params.times

    # Check parameter dimensions and replicate single values if needed
    for param_name, param_value in [('la', la), ('psi', psi), ('p', p)]:
        if param_value is not None:
            if len(param_value) == 1:
                # Replicate a single value across all intervals
                globals()[param_name] = param_value * n_intervals
            elif len(param_value) != n_intervals:
                raise ValueError(
                    f"If providing multiple values for {param_name}, must provide exactly {n_intervals} values (one for each interval)")

    # Estimate parameters
    vs, cis = infer_skyline(
        forest,
        T,
        n_intervals=n_intervals,
        la=la,
        psi=psi,
        p=p,
        times=times,
        lower_bounds=params.lower_bounds,
        upper_bounds=params.upper_bounds,
        ci=params.ci,
        threads=params.threads,
        num_attemps=params.attempts
    )

    # Save results
    save_results_skyline(vs, cis, params.log, n_intervals, ci=params.ci)
    print(f"Results have been saved to {params.log}")


def loglikelihood_skyline_main():
    """
    Entry point for tree likelihood estimation with the BDSKY model with command-line arguments.
    This function calculates the likelihood of a tree or forest under fixed parameter values.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate BDSKY likelihood on a given forest for given parameter values.")

    # Required arguments
    parser.add_argument('--nwk', required=True, type=str, help="Input tree file")

    # Parameters for each interval
    parser.add_argument('--la', required=True, type=float, nargs='+',
                        help="Transmission rate(s) - either a single value for all intervals or one per interval")
    parser.add_argument('--psi', required=True, type=float, nargs='+',
                        help="Removal rate(s) - either a single value for all intervals or one per interval")
    parser.add_argument('--p', required=True, type=float, nargs='+',
                        help="Sampling probability(ies) - either a single value for all intervals or one per interval")
    parser.add_argument('--times', required=False, type=float, nargs='+',
                        help="Time points specifying interval boundaries (determines number of intervals)")

    # Optional arguments
    parser.add_argument('--u', required=False, type=int, default=-1,
                        help="Number of hidden trees (estimated by default)")
    parser.add_argument('--threads', required=False, type=int, default=1,
                        help="Number of threads to use")

    params = parser.parse_args()

    # Determine number of intervals
    if params.times is not None:
        n_intervals = len(params.times) + 1
    else:
        # Default to 1 interval if no time points are provided
        n_intervals = 1

    # Ensure parameter lists have the right length
    def ensure_list(param_name, param_values, expected_len):
        if len(param_values) == 1:
            return param_values * expected_len
        elif len(param_values) != expected_len:
            raise ValueError(f"Expected either 1 or {expected_len} values for {param_name}, got {len(param_values)}")
        return param_values

    la_values = ensure_list("la", params.la, n_intervals)
    psi_values = ensure_list("psi", params.psi, n_intervals)
    p_values = ensure_list("p", params.p, n_intervals)

    # Time points are required for skyline models with multiple intervals
    if n_intervals > 1 and params.times is None:
        raise ValueError(f"Time points are required for models with {n_intervals} intervals")

    times = params.times if params.times is not None else []

    # Create parameter vector
    parameter_vector = []
    parameter_vector.extend(la_values)
    parameter_vector.extend(psi_values)
    parameter_vector.extend(p_values)
    parameter_vector.extend(times)

    # Load forest
    forest = read_forest(params.nwk)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)

    # Calculate likelihood
    lk = loglikelihood(forest, *parameter_vector, T=T, threads=params.threads, u=params.u, n_intervals=n_intervals)

    # Print results
    print(f"Log-likelihood: {lk}")
    print(f"Parameters:")
    print(format_parameters_skyline(parameter_vector, n_intervals))

    return lk


if __name__ == '__main__':
    import sys

    # Check if the first argument indicates which main function to run
    if len(sys.argv) > 1 and sys.argv[1] == "--calculate-likelihood":
        # Remove the flag before passing to the argument parser
        sys.argv.pop(1)
        loglikelihood_skyline_main()
    else:
        main()
