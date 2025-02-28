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
    parser.add_argument('--min_rho_ups', default=0.01, type=float, help="min rho * upsilon (excluded)")
    parser.add_argument('--max_ups', default=0.75, type=float, help="max upsilon (included)")
    parser.add_argument('--min_phi_by_psi', default=10, type=float, help="min phi / psi (excluded)")
    parser.add_argument('--max_phi_by_psi', default=250, type=float, help="max phi / psi (included)")
    parser.add_argument('--min_f', default=0.01, type=float, help="min SS fraction (included)")
    parser.add_argument('--max_f', default=0.5, type=float, help="max SS fraction (excluded)")
    parser.add_argument('--min_x', default=10, type=float, help="min SS rate ratio (excluded)")
    parser.add_argument('--max_x', default=100, type=float, help="max SS rate ratio (included)")
    parser.add_argument('--min_de_by_di_de', default=0.2, type=float, help="min incubation by incubation plus infectious time (excluded)")
    parser.add_argument('--max_de_by_di_de', default=0.8, type=float, help="max incubation by incubation plus infectious time (included)")
    parser.add_argument('--kappa', default=0, type=int, help="max number of notified contacts")
    params = parser.parse_args()

    R0 = random_float(params.min_R0, params.max_R0)
    d = random_float(params.min_di, params.max_di)
    psi = 1 / d
    la = psi * R0
    rho = random_float(params.min_rho, params.max_rho)
    upsilon = random_float(params.min_rho_ups / rho, params.max_ups) if params.kappa else 0
    phi = psi * random_float(params.min_phi_by_psi, params.max_phi_by_psi) if params.kappa else psi
    f = params.min_f + (params.max_f - params.min_f) * np.random.random(size=1)[0]
    x = random_float(params.min_x, params.max_x)
    inc_perc = random_float(params.min_de_by_di_de, params.max_de_by_di_de)
    mu = psi * (1 / inc_perc - 1)

    la_ss = la / (1 + (1 / f - 1) / x)
    la_nn = la - la_ss
    la_ns = la_ss / x
    la_sn = x * la_nn

    with open(params.log, 'w+') as fo:
        fo.write('la,psi,rho,phi,upsilon,mu,f,x,la_ss,la_nn,la_sn,la_ns,R0,d,kappa\n')
        fo.write(f'{la},{psi},{rho},{phi},{upsilon},{mu},{f},{x},{la_ss},{la_nn},{la_sn},{la_ns},{R0},{d},{params.kappa}\n')


