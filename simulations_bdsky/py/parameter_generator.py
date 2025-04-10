import numpy as np


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

    parser = argparse.ArgumentParser(description="Generates parameters for MTBD-CT simulations.")
    parser.add_argument('--log', default='/home/azhukova/projects/bdpn/simulations/parameters.101.log', type=str, help="parameter settings")
    parser.add_argument('--min_R0', default=1, type=float, help="min R0 (excluded)")
    parser.add_argument('--max_R0', default=10, type=float, help="max R0 (included)")
    parser.add_argument('--min_rho', default=0.05, type=float, help="min rho (excluded)")
    parser.add_argument('--max_rho', default=0.75, type=float, help="max rho (included)")
    parser.add_argument('--min_di', default=1, type=float, help="min infectious time (excluded)")
    parser.add_argument('--max_di', default=21, type=float, help="max infectious time (included)")
    params = parser.parse_args()

    R0 = random_float(params.min_R0, params.max_R0)
    d = random_float(params.min_di, params.max_di)
    psi = 1 / d
    la = psi * R0
    rho = random_float(params.min_rho, params.max_rho)

    with open(params.log, 'w+') as fo:
        fo.write('la,psi,rho,R0,d\n')
        fo.write(f'{la},{psi},{rho},{R0},{d}\n')


