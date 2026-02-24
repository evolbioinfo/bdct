import multiprocessing
import re
import time

import numpy as np
from treesimulator import save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import BirthDeathModel, BirthDeathWithSuperSpreadingModel

TIMEOUT = int(10 * 60)  # seconds

RHO = 'rho'
REPRODUCTIVE_NUMBER = 'R'
INFECTION_DURATION = 'd'

F_S = 'f_S'
X_S = 'X_S'

N_RECIPIENTS = 'r'


def random_float(rng: np.random.Generator, min_value: float = 0, max_value: float = 1, size=1) -> float:
    """
    Generate a random float in [min_value, max_value[
    :param rng: random generator
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + rng.random(size=None if 1 == size else size) * (max_value - min_value)


def generate_tree(params, pid, results, i, rep):
    rng = np.random.default_rng(seed=int(time.time()) + (i * rep))

    model_name = params.model

    R = random_float(rng, params.min_R, params.max_R)
    d = random_float(rng, params.min_d, params.max_d)
    r_I = random_float(rng, params.min_recipients, params.max_recipients) \
        if params.max_recipients > params.min_recipients else params.min_recipients
    if 'SS' in model_name:
        f_ss = random_float(rng, params.min_fss, params.max_fss)
        x_ss = random_float(rng, params.min_xss, params.max_xss)
    else:
        f_ss, x_ss = 0, 1
    r_S = r_I * x_ss
    r = r_I * (1 - f_ss) + r_S * f_ss
    rho = random_float(rng, params.min_rho, params.max_rho)
    la = R / r / d
    psi = 1 / d

    print(f'la={la}, psi={psi}, rho={rho}, r={r}, f_ss={f_ss}, x_ss={x_ss}')
    if 'SS' in model_name:
        model = BirthDeathWithSuperSpreadingModel(p=rho, psi=psi,
                                                  la_nn=la * (1 - f_ss), la_ns=la * f_ss,
                                                  la_sn=la * (1 - f_ss), la_ss=la * f_ss,
                                                  n_recipients=[r_I, r_S])
    else:
        model = BirthDeathModel(p=rho, la=la, psi=psi, n_recipients=[r])


    tips = rng.integers(params.min_tips, params.max_tips + 1)
    print(f'n_tips={tips}')
    epidemic = generate([model], min_tips=tips, max_tips=tips,
                        return_stats=True)

    R_o, d_o = epidemic.R_e, epidemic.d

    print('Base model\'s R and d:', model.get_avg_R(), model.get_avg_d(), 'vs hoped for:', R, d, 'vs observed:', R_o, d_o)

    results[pid] = epidemic.sampled_forest[0], (R, d, rho, r, f_ss, x_ss, R_o, d_o)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates trees under BD(SS)-MULT models.")
    parser.add_argument('--log', required=True, type=str, help="output parameters")
    parser.add_argument('--nwk', required=True, type=str, help="output trees")

    parser.add_argument('--min_R', default=1, type=float, help="min R0 (included)")
    parser.add_argument('--max_R', default=10., type=float, help="max R0 (excluded)")
    parser.add_argument('--min_d', default=1, type=float, help="min infection time (included)")
    parser.add_argument('--max_d', default=31., type=float, help="max infection time (excluded)")
    parser.add_argument('--min_rho', default=0.01, type=float, help="min rho (included)")
    parser.add_argument('--max_rho', default=0.75, type=float, help="max rho (excluded)")
    parser.add_argument('--min_fss', default=0., type=float, help="min superspreading fraction (included)")
    parser.add_argument('--max_fss', default=0.5, type=float, help="max superspreading fraction (excluded)")
    parser.add_argument('--min_xss', default=2., type=float, help="min superspreading rate ratio (included)")
    parser.add_argument('--max_xss', default=25., type=float, help="max superspreading rate ratio (excluded)")
    parser.add_argument('--min_recipients', default=1., type=float, help="min number of recipients of the standard infectious individual (included)")
    parser.add_argument('--max_recipients', default=2., type=float, help="min number of recipients of the standard infectious individual  (excluded)")
    parser.add_argument('--min_tips', default=200, type=int, help="min tips (included)")
    parser.add_argument('--max_tips', default=500, type=int, help="max tips (included)")
    parser.add_argument('--model', choices=['BD', 'BDSS'], type=str, help='tree model to use for generation')
    parser.add_argument('--n', default=20, type=int, help="number of trees to generate")
    params = parser.parse_args()

    indices = [int(_) for _ in re.findall(r'[0-9]+', params.nwk)]
    i = ((indices[-1] if len(indices) else 0) + 1) + max(0, indices[-2] if len(indices) > 1 else 0) * 128

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    rep = 0
    for pid in range(params.n):
        while True:
            if pid in return_dict:
                print("Generated a tree...")
                break

            p = multiprocessing.Process(target=generate_tree, args=(params, pid, return_dict, i, rep))
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
            rep += 1

    forest = []
    with open(params.log, 'w+') as f:
        f.write(
            f'{REPRODUCTIVE_NUMBER},{INFECTION_DURATION},{RHO},{N_RECIPIENTS},{F_S},{X_S},tips,R_observed,d_observed\n')

        for tree, (R, d, p, r, f_ss, x_ss, R_o, d_o) in return_dict.values():
            f.write(f'{R},{d},{p},{r},{f_ss},{x_ss},{len(tree)},{R_o},{d_o}\n')
            forest.append(tree)
    save_forest(forest, params.nwk)
