import logging

import numpy as np
from matplotlib import pyplot as plt

from bdpn import bd_model, bdmult_model
from bdpn.bdmult_model import get_tt, get_index_t, N_INTERVALS, rescale_parameters

from bdpn.formulas import get_c1, get_E, get_c2

from treesimulator.mtbd_models import BirthDeathModel
from treesimulator.generator import simulate_tree_gillespie

REPETITIONS = 10000


def random_bt_0_and_1():
    return 1 - np.random.random(size=1)[0]


if __name__ == '__main__':

    # rho = random_bt_0_and_1()
    # R0 = random_bt_0_and_1() * 5
    # psi_initial = random_bt_0_and_1()
    # la_initial = psi_initial * R0
    # print(la_initial, psi_initial, rho)
    # la_initial, psi_initial, rho = 1.627328726998328, 0.924748168494683, 0.11874047409503463
    la_initial, psi_initial, rho = 2.9760137144967165, 0.9625550346746248, 0.5117843385922632

    # T = 1000
    # tt, logdt = get_tt(T)

    # t = random_bt_0_and_1() * T
    #
    # print(tt, t, tt[get_index_t(t, tt, logdt, T)])
    # # print(len(tt), tt[0], tt[-1])
    # plt.plot(tt, np.arange(0, N_INTERVALS + 1), '*')
    # plt.show()

    for T_initial in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        T = 100
        scaling_factor = T / T_initial
        la, psi = la_initial / scaling_factor, psi_initial / scaling_factor

        ti = random_bt_0_and_1() * T
        tp = random_bt_0_and_1() * ti
        tt = np.arange(T + T / 100, step=T/100)

        c1 = get_c1(la, psi, rho)
        c2 = get_c2(la, psi, c1)
        Us = [bd_model.get_u(la, psi, c1, get_E(c1, c2, t, T)) for t in tt]
        plt.plot(tt / scaling_factor, Us, label='U_BD(t)', alpha=0.5)

        # E_ti = get_E(c1, c2, ti, T)
        # step = max((ti - tp) / 100, 1e-4)
        # tt_p = np.arange(tp, ti + step, step=step)
        # Ps = [np.exp(bd_model.get_log_p(c1, t, ti, get_E(c1, c2, t, T), E_ti)) for t in tt_p]
        # plt.plot(tt_p / scaling_factor, Ps, label='P_BD(t)', alpha=0.5)

        for r in [1, 2, 3, 10]:
            model = BirthDeathModel(p=rho, la=la, psi=psi, n_recipients=[r])

            Us = np.zeros(101, dtype=float)
            for _ in range(REPETITIONS):
                tree, max_time = simulate_tree_gillespie(model, max_time=T, ltt=False, max_sampled=1, min_sampled=1)
                if not tree:
                    Us += 1
                else:
                    Us[: int(tree.dist / (T/100) + 1)] += 1
            plt.plot((T - tt) / scaling_factor, Us / REPETITIONS, '*', label='U_sim_{}'.format(r), alpha=0.5)

            tt_, _ = get_tt(T)
            Us = bdmult_model.precalc_u(T, tt_, la, psi, rho, r)
            plt.plot(tt_ / scaling_factor, Us, '--', label='U_BD_{}(t)'.format(r), alpha=0.5)
            # plt.plot(tt_p / scaling_factor, [np.exp(bdmult_model.get_log_p(t, ti, tt, logdt, T, la, psi, r, Us)) for t in tt_p],
            #          '--', label='P_BD_{}(t)'.format(r), alpha=0.5)

        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.show()

        plt.clf()

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
