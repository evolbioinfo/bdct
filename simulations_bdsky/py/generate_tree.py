import multiprocessing
import re
import time

import numpy as np
from treesimulator import save_forest, save_log

from treesimulator.mtbd_models import BirthDeathModel
from treesimulator.generator import generate

TIMEOUT = int(5 * 60) # seconds


def random_float(rng: np.random.Generator, min_value=0, max_value=1, n=1):
    """
    Generate n random floats in [min_value, max_value[
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + rng.random(size=n) * (max_value - min_value)


def generate_tree(params, results):
    tree_id = int(re.findall(r'[0-9]+', params.log)[-1])
    rng = np.random.default_rng(seed=int(time.time() + TIMEOUT * tree_id))

    total_tips = rng.integers(params.min_tips, params.max_tips + 1)
    n_intervals = rng.integers(params.min_intervals, params.max_intervals + 1) \
        if params.min_intervals < params.max_intervals else params.min_intervals

    Rs = random_float(rng, params.min_R, params.max_R, n=n_intervals)
    ds = random_float(rng, params.min_d, params.max_d, n=n_intervals)
    rhos = random_float(rng, params.min_rho, params.max_rho, n=n_intervals)
    psis = 1 / ds
    las = psis * Rs
    models = []
    for (la, psi, rho) in zip(las, psis, rhos):
        models.append(BirthDeathModel(la=la, psi=psi, p=rho))

    while True:
        skyline_times = []
        for i in range(n_intervals):
            n_tips = int(total_tips * (i + 1) / n_intervals)
            [tree], (_, _, T, observed_freqs), _ = generate(models[:i + 1], min_tips=n_tips, max_tips=n_tips,
                                                            skyline_times=skyline_times)

            # If wanted to generate i intervals but ended up with less, then restart
            if skyline_times and skyline_times[-1] > T:
                skyline_times = []
                break
            if i != n_intervals - 1:
                skyline_times.append(T)
        if len(skyline_times) == n_intervals - 1:
            results[tree_id] = tree, models, skyline_times, T, observed_freqs
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates parameters for BDEISS-CT simulations.")
    parser.add_argument('--log', required=True, type=str, help="output parameters")
    parser.add_argument('--nwk', required=True, type=str, help="output trees")

    parser.add_argument('--min_R', default=0.5, type=float, help="min R0 (included)")
    parser.add_argument('--max_R', default=10., type=float, help="max R0 (excluded)")
    parser.add_argument('--min_d', default=0.5, type=float, help="min infection time (included)")
    parser.add_argument('--max_d', default=12., type=float, help="max infection time (excluded)")
    parser.add_argument('--min_rho', default=0.01, type=float, help="min rho (included)")
    parser.add_argument('--max_rho', default=0.75, type=float, help="max rho (excluded)")
    parser.add_argument('--min_intervals', default=2, type=int, help='min skyline intervals (included)')
    parser.add_argument('--max_intervals', default=2, type=int, help='max skyline intervals (included)')
    parser.add_argument('--min_tips', default=100, type=int, help="min tips (included)")
    parser.add_argument('--max_tips', default=200, type=int, help="max tips (included)")
    params = parser.parse_args()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    while True:
        p = multiprocessing.Process(target=generate_tree, args=(params, return_dict))
        p.start()

        # Wait for TIMEOUT seconds or until process finishes
        p.join(TIMEOUT)

        # If thread is still active
        if p.is_alive():
            print("Tree generation took too long, restarting...")
            # Terminate - may not work if process is stuck for good
            p.terminate()
            # OR Kill - will work for sure, no chance for process to finish nicely however
            # p.kill()
        else:
            print("Generated a tree...")
            break
    tree_id = int(re.findall(r'[0-9]+', params.log)[-1])
    tree, models, skyline_times, T, obs = return_dict[tree_id]
    save_log(models, skyline_times, len(tree), T, u=0, log=params.log, kappa=0, observed_frequencies=obs)
    save_forest([tree], params.nwk)



