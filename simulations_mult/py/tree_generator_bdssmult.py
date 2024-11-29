import logging

import numpy as np
from treesimulator import save_log, save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import BirthDeathWithSuperSpreadingModel


def random_float(min_value=0, max_value=1):
    """
    Generate a random float in ]min_value, max_value]
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + (1 - np.random.random(size=1)[0]) * (max_value - min_value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates random BDSS-MULT tree.")
    parser.add_argument('--log', default='/home/azhukova/projects/bdpn/simulations_mult/medium/BDSSMULT/tree.1.log', type=str, help="Nwk")
    parser.add_argument('--nwk', default='/home/azhukova/projects/bdpn/simulations_mult/medium/BDSSMULT/tree.1.nwk', type=str, help="Log")
    parser.add_argument('--min_tips', default=200, type=int, help="Min number of tips")
    parser.add_argument('--max_tips', default=500, type=int, help="Max number of tips")
    parser.add_argument('--rho', default=None, type=float, help="Sampling probability")
    parser.add_argument('--la', default=None, type=float, help="Transmission rate")
    parser.add_argument('--psi', default=None, type=float, help="Removal rate")
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=params.nwk.replace('.nwk', '.txt'), filemode='w+')

    if params.psi is None:
        if params.la is None:
            R0 = random_float(1, 5)
            psi = random_float(1 / 20, 1 / 5)
            la = psi * R0
        else:
            R0 = random_float(1, 5)
            la = params.la
            psi = la / R0
    elif params.la is None:
        psi = params.psi
        R0 = random_float(1, 5)
        la = psi * R0
    else:
        psi = params.psi
        la = params.la
    if params.rho is None:
        rho = random_float(0.1, 0.9)
    else:
        rho = params.rho
    pi_N = random_float(0.5, 0.9)
    pi_S = 1 - pi_N
    r_N = random_float(min_value=1, max_value=10)
    r_N_R_S = random_float(min_value=1, max_value=25)
    r_S = r_N * r_N_R_S

    print(la, psi, rho, pi_N, r_N, r_S)

    model = BirthDeathWithSuperSpreadingModel(p=rho, la=la, psi=psi,
                                              la_nn=la * pi_N, la_ns=la * pi_S,
                                              la_sn=la * pi_N, la_ss=la * pi_S,
                                              n_recipients=[r_N, r_S])

    forest, (total_tips, u, T), _ = generate(model, params.min_tips, params.max_tips)

    save_forest(forest, params.nwk)
    save_log(model, total_tips, T, u, params.log)


