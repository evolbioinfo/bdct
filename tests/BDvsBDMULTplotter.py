import logging
from collections import Counter

import numpy as np
from ete3 import TreeNode
# from matplotlib import pyplot as plt

from treesimulator import DIST_TO_START

from bdpn import bd_model, bdmult_model

from treesimulator.mtbd_models import BirthDeathModel
from treesimulator.generator import simulate_tree_gillespie

from bdpn.tree_manager import annotate_forest_with_time, tree2vector, vector2tree, annotate_tree, rescale_forest, TIME

REPETITIONS = 1000000


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

    # for T_initial in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
    for T_initial in (1,):
        T = 100
        scaling_factor = T / T_initial
        la, psi = la_initial / scaling_factor, psi_initial / scaling_factor
        # T, la, psi = T_initial, la_initial, psi_initial

        N = 1000
        delta_t = T / N
        tt = np.arange(T + delta_t, step=delta_t)

        ti = np.random.choice(tt[-10:-7], 1)[0]
        tp = tt[0]  #np.random.choice(tt[tt < ti], 1)[0]

        # c1 = get_c1(la, psi, rho)
        # c2 = get_c2(la, psi, c1)
        # # Us = [bd_model.get_u(la, psi, c1, get_E(c1, c2, t, T)) for t in tt]
        # # plt.plot(tt / scaling_factor, Us, label='U_BD(t)', alpha=0.5)
        #
        # E_ti = get_E(c1, c2, ti, T)
        # mask = (tp <= tt) & (tt <= ti)
        # tt_p = tt[mask]
        # Ps = [np.exp(bd_model.get_log_p(c1, t, ti, get_E(c1, c2, t, T), E_ti)) for t in tt_p]
        # # plt.plot(tt_p / scaling_factor, Ps, label='P_BD(t)', alpha=0.5)
        # plt.plot(tt_p, Ps, label='P_BD(t)', alpha=0.5)
        # print()
        # U_ti = bd_model.get_u(la, psi, c1, get_E(c1, c2, ti, T))
        # plt.plot(tt_p, np.exp((psi + la - 2 * la * U_ti) * (tt_p - ti)), '--', label='limit1', alpha=0.5)
        # U_tp = bd_model.get_u(la, psi, c1, get_E(c1, c2, tp, T))
        # print(U_ti, U_tp, (la + psi) / 2 / la)
        # plt.plot(tt_p, np.exp((psi + la - 2 * la * U_tp) * (tt_p - ti)), '*', label='limit2', alpha=0.5)

        # for r in [1, 2, 3, 10]:
        for r in [1, 2]:
            model = BirthDeathModel(p=rho, la=la_initial, psi=psi_initial, n_recipients=[r])
            delta_t = T / 100

            conf2c = Counter()
            Us = np.zeros(N + 1, dtype=float)
            for _ in range(REPETITIONS):
                tree, max_time = simulate_tree_gillespie(model, max_time=T_initial, ltt=False, max_sampled=np.inf,
                                                         min_sampled=1)
                if not tree:
                    Us += 1
                else:
                    sampled_dist = min(getattr(_, DIST_TO_START) for _ in tree)
                    # Us[: int(sampled_dist * scaling_factor / (T/100) + 1)] += 1
                    Us[: int(sampled_dist / delta_t + 1)] += 1
                    rescale_forest([tree], T_target=T, T=T_initial)
                    conf2c[tuple(tree2vector(annotate_tree(tree), lambda _: int(_ / delta_t)))] += 1

            lk_sim, lk_est, n_tips, n_tips_real = [], [], [], []
            for conf, count in conf2c.items():
                n_tips.append(len(conf))
                if n_tips[-1] == 1:
                    tree = vector2tree(list(conf), None, lambda _: _ * delta_t + delta_t / 2)
                    n_nodes = sum(1 for _ in tree.traverse())
                    n_tips_real.append(n_nodes)
                    print(conf, tree.get_ascii(attributes=[TIME]))
                    lk_sim.append(np.log(count) - np.log(REPETITIONS))
                    lk_est.append(bdmult_model.loglikelihood([tree], la, psi, rho, r, T)
                                  + sum(1 for _ in tree.traverse()) * np.log(delta_t))
            lk_sim = np.array(lk_sim)
            lk_est = np.array(lk_est)
            print(lk_sim - lk_est)

        #                     # Ps[int(n_tp * scaling_factor / (T/100) + 1): int(n_ti * scaling_factor / (T/100) + 1)] += 1
        #                     Ps[int(n_tp / delta_t + 1): int(n_ti / delta_t + 1)] += 1
        #     print(num_n)
        #     # plt.plot((T - tt) / scaling_factor, Us / REPETITIONS, '*', label='U_sim_{}'.format(r), alpha=0.5)
        #     plt.plot(tt_p, Ps[mask] / num_n, '*', label='P_sim_{}'.format(r), alpha=0.5)
        #
        #     tt_, logdt = get_tt(T)
        #     Us = bdmult_model.precalc_u(T, tt_, la, psi, rho, r)
        #     # plt.plot(tt_ / scaling_factor, Us, '--', label='U_BD_{}(t)'.format(r), alpha=0.5)
        #     plt.plot(tt_p,
        #              [np.exp(bdmult_model.get_log_p(t, ti, tt_, logdt, T, la, psi, r, Us))
        #               for t in tt_p], '--', label='P_BD_{}(t)'.format(r), alpha=0.5)
        #
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        # plt.show()
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
