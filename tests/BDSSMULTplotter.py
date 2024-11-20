from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from treesimulator import DIST_TO_START

from bdpn import bdssmult_model

from treesimulator.mtbd_models import BirthDeathWithSuperSpreadingModel
from treesimulator.generator import simulate_tree_gillespie

REPETITIONS = 50000


def random_bt_0_and_1():
    return 1 - np.random.random(size=1)[0]


if __name__ == '__main__':

    rho = random_bt_0_and_1()
    R0 = random_bt_0_and_1() * 5
    psi_initial = random_bt_0_and_1()
    la_initial = psi_initial * R0
    print(la_initial, psi_initial, rho)
    # la_initial, psi_initial, rho = 1.627328726998328, 0.924748168494683, 0.11874047409503463
    pi_N = random_bt_0_and_1()
    # la_initial, psi_initial, rho = 2.9760137144967165, 0.9625550346746248, 0.5117843385922632
    # pi_N = 0.7

    pi_S = 1 - pi_N

    r_N = int(1 + random_bt_0_and_1() * 5)

    # T = 1000
    # tt, logdt = get_tt(T)

    # t = random_bt_0_and_1() * T
    #
    # print(tt, t, tt[get_index_t(t, tt, logdt, T)])
    # # print(len(tt), tt[0], tt[-1])
    # plt.plot(tt, np.arange(0, N_INTERVALS + 1), '*')
    # plt.show()

    for T_initial in (0.01, 0.1, 1, 10, 100, 1000):
        # for T_initial in (1,):
        T = 1000
        scaling_factor = T / T_initial
        la, psi = la_initial / scaling_factor, psi_initial / scaling_factor
        # T, la, psi = T_initial, la_initial, psi_initial

        N = 100
        delta_t = T_initial / N
        global_tt = np.arange(T_initial + delta_t, step=delta_t)

        # for r in [1, 2, 3, 10]:
        # for r_N, r_S in [(1, 2), (1, 5), (2, 4), (2, 10)]:
        for r_S in [r_N + 1, r_N * 2, r_N * 3, r_N * 10]:
            model = BirthDeathWithSuperSpreadingModel(la_nn=la_initial * pi_N,
                                                      la_ns=la_initial * pi_S,
                                                      la_sn=la_initial * pi_N,
                                                      la_ss=la_initial * pi_S,
                                                      psi=psi_initial,
                                                      p=rho, n_recipients=[r_N, r_S])

            Us_N = np.zeros(N + 1, dtype=float)
            Us_S = np.zeros(N + 1, dtype=float)
            for _ in range(REPETITIONS):
                tree, max_time = simulate_tree_gillespie(model, max_time=T_initial, ltt=False, max_sampled=1,
                                                         min_sampled=1, root_state=model.states[0])
                if not tree:
                    Us_N += 1
                else:
                    sampled_dist = min(getattr(_, DIST_TO_START) for _ in tree)
                    Us_N[: int(sampled_dist / delta_t + 1)] += 1

                tree, max_time = simulate_tree_gillespie(model, max_time=T_initial, ltt=False, max_sampled=1,
                                                         min_sampled=1, root_state=model.states[1])
                if not tree:
                    Us_S += 1
                else:
                    sampled_dist = min(getattr(_, DIST_TO_START) for _ in tree)
                    # Us[: int(sampled_dist * scaling_factor / (T/100) + 1)] += 1
                    Us_S[: int(sampled_dist / delta_t + 1)] += 1

            print(Us_S)
            plt.plot((T_initial - global_tt), Us_N / REPETITIONS, '*', label=f'U_N_sim_{r_N},{r_S}', alpha=0.5)
            plt.plot((T_initial - global_tt), Us_S / REPETITIONS, '*', label=f'U_S_sim_{r_N},{r_S}', alpha=0.5)
            #     plt.plot(tt_p, Ps[mask] / num_n, '*', label='P_sim_{}'.format(r), alpha=0.5)
            #
            Us, tt, t2index = bdssmult_model.get_u_function(T, la, psi, rho, pi_N, r_N, r_S)
            # plt.plot(tt_ / scaling_factor, Us, '--', label='U_BD_{}(t)'.format(r), alpha=0.5)
            plt.plot(tt / scaling_factor, Us[0, :], '--', label=f'U_N_{r_N},{r_S}', alpha=0.5)
            plt.plot(tt / scaling_factor, Us[1, :], '--', label=f'U_S_{r_N},{r_S}', alpha=0.5)
        #
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.show()
        #
        # plt.clf()

        # ti = random_bt_0_and_1() * T
        #
        # E_ti = get_E(c1, c2, ti, T)
        # tt = [t for t in tt if t < ti]
        # Ps = [np.exp(bd_model.get_log_p(c1, t, ti, get_E(c1, c2, t, T), E_ti)) for t in tt]
        # plt.plot(tt, Ps, label='P_BD(t)', alpha=0.5)
        # Us = bdmult_model.precalc_u(T, dt, la, psi, rho, r=1.1)
        # plt.plot(tt, [np.exp(bdmult_model.get_log_p(t, ti, dt, la, psi, r=1.1, Us=Us)) for t in tt],
        #          label='P_BD_{}(t)'.format(1.1), alpha=0.5)
        # for r in range(2, 3):
        #     Us = bdmult_model.precalc_u(T, dt, la, psi, rho, r=r)
        #     plt.plot(tt, [np.exp(bdmult_model.get_log_p(t, ti, dt, la, psi, r=r, Us=Us)) for t in tt],
        #              label='P_BD_{}(t)'.format(r), alpha=0.5)
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        # plt.show()
        # plt.clf()
