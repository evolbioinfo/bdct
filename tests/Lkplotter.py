import numpy as np
from matplotlib import pyplot as plt



from bdpn.bdmult_model import loglikelihood
from bdpn.tree_manager import annotate_forest_with_time, rescale_forest, TIME, \
    read_forest, get_T

cmaps = {'Perceptually Uniform Sequential': [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'], 'Sequential': [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'], 'Sequential (2)': [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper'], 'Diverging': [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'], 'Cyclic': ['twilight', 'twilight_shifted', 'hsv'], 'Qualitative': [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'], 'Miscellaneous': [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar']}



def random_bt_0_and_1():
    return 1 - np.random.random(size=1)[0]


if '__main1__' == __name__:
    forest = read_forest('/home/azhukova/projects/bdpn/simulations_mult/minitree5.nwk')
    rs = list(range(2, 20))
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    # node = TreeNode(dist=0.1)
    # node.add_feature(TIME, 0.1)
    # forest = [node]
    # lks_log = [loglikelihood(forest, 2, 1, 0.4, r, T_initial, as_log=True) for r in rs]
    lks = [loglikelihood(forest, 2, 1, 0.4, r, T_initial, as_log=False) for r in rs]
    T = 100
    # T = T_initial
    print(forest[0].get_ascii(attributes=['dist', TIME]))
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    n = sum(sum(1 for _ in tree.traverse()) for tree in forest)
    lks2_log = [loglikelihood(forest, 2 / scaling_factor, 1 / scaling_factor, 0.4, r, T, as_log=True) + n * np.log(scaling_factor) for r in rs]
    lks2 = [loglikelihood(forest, 2 / scaling_factor, 1 / scaling_factor, 0.4, r, T, as_log=False) + n * np.log(scaling_factor) for r in rs]
    # plot(rs, lks_log, '-', label='lk')
    plt.plot(rs, lks2_log, '*', label='lk_log')
    plt.plot(rs, lks, '-', label='lk_resc')
    plt.plot(rs, lks2, '--', label='lk_resc_log')
    plt.legend(loc='best')
    plt.xlabel('r')
    plt.show()

if '__main2__' == __name__:
    forest = read_forest('/home/azhukova/projects/bdpn/simulations_mult/minitree5.nwk')
    psis = np.arange(0.5, 10, step=0.5)
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    # node = TreeNode(dist=0.1)
    # node.add_feature(TIME, 0.1)
    # forest = [node]
    # lks_log = [loglikelihood(forest, 2, 1, 0.4, r, T_initial, as_log=True) for r in rs]
    lks = [loglikelihood(forest, la, 1, 0.4, 5, T_initial, as_log=False) for la in psis]
    T = 100
    # T = T_initial
    print(forest[0].get_ascii(attributes=['dist', TIME]))
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    n = sum(sum(1 for _ in tree.traverse()) for tree in forest)
    lks2_log = [loglikelihood(forest, la / scaling_factor, 1 / scaling_factor, 0.4, 5, T, as_log=True) + n * np.log(scaling_factor) for la in psis]
    lks2 = [loglikelihood(forest, la / scaling_factor, 1 / scaling_factor, 0.4, 5, T, as_log=False) + n * np.log(scaling_factor) for la in psis]
    # plot(rs, lks_log, '-', label='lk')
    plt.plot(psis, lks2_log, '*', label='lk_log')
    plt.plot(psis, lks, '-', label='lk_resc')
    plt.plot(psis, lks2, '--', label='lk_resc_log')
    plt.legend(loc='best')
    plt.xlabel('la')
    plt.show()


if '__main2__' == __name__:
    forest = read_forest('/home/azhukova/projects/bdpn/simulations_mult/minitree5.nwk')
    psis = np.arange(0.2, 6, step=0.2)
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    # node = TreeNode(dist=0.1)
    # node.add_feature(TIME, 0.1)
    # forest = [node]
    # lks_log = [loglikelihood(forest, 2, 1, 0.4, r, T_initial, as_log=True) for r in rs]
    # lks = [loglikelihood(forest, 2, psi, 1, 5, T_initial, as_log=False) for psi in psis]
    T = 100
    # T = T_initial
    print(forest[0].get_ascii(attributes=['dist', TIME]))
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    n = sum(sum(1 for _ in tree.traverse()) for tree in forest)
    lks2_log = [loglikelihood(forest, 2 / scaling_factor, psi / scaling_factor, 1, 5, T, as_log=True) + n * np.log(scaling_factor) for psi in psis]
    # lks2 = [loglikelihood(forest, 2 / scaling_factor, psi / scaling_factor, 1, 5, T, as_log=False) + n * np.log(scaling_factor) for psi in psis]
    # plot(rs, lks_log, '-', label='lk')
    for rho in np.arange(0.1, 1.1, 0.1):
        plt.plot(psis, [loglikelihood(forest, 2 / scaling_factor, psi / scaling_factor, rho, 5, T, as_log=True) + n * np.log(scaling_factor) for psi in psis], '-', label=f'lk_log-rho={rho}')
    plt.legend(loc='best')
    plt.xlabel('psi')
    plt.show()


if '__main4__' == __name__:
    forest = read_forest('/home/azhukova/projects/bdpn/simulations_mult/minitree5.nwk')
    ps = np.arange(0.1, 1.1, step=0.1)
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    # node = TreeNode(dist=0.1)
    # node.add_feature(TIME, 0.1)
    # forest = [node]
    # lks_log = [loglikelihood(forest, 2, 1, 0.4, r, T_initial, as_log=True) for r in rs]
    lks = [loglikelihood(forest, 2, 1, p, 5, T_initial, as_log=False) for p in ps]
    T = 100
    # T = T_initial
    print(forest[0].get_ascii(attributes=['dist', TIME]))
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    n = sum(sum(1 for _ in tree.traverse()) for tree in forest)
    lks2_log = [loglikelihood(forest, 2 / scaling_factor, 1 / scaling_factor, p, 5, T, as_log=True) + n * np.log(scaling_factor) for p in ps]
    lks2 = [loglikelihood(forest, 2 / scaling_factor, 1 / scaling_factor, p, 5, T, as_log=False) + n * np.log(scaling_factor) for p in ps]
    # plot(rs, lks_log, '-', label='lk')
    plt.plot(ps, lks2_log, '*', label='lk_log')
    plt.plot(ps, lks, '-', label='lk_resc')
    plt.plot(ps, lks2, '--', label='lk_resc_log')
    plt.legend(loc='best')
    plt.xlabel('rho')
    plt.show()



if '__main0__' == __name__:
    forest = read_forest('/home/azhukova/projects/bdpn/simulations_mult/minitree5.nwk')
    ps = np.arange(1.1, 10, step=0.25)
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    # node = TreeNode(dist=0.1)
    # node.add_feature(TIME, 0.1)
    # forest = [node]
    # lks_log = [loglikelihood(forest, 2, 1, 0.4, r, T_initial, as_log=True) for r in rs]
    # lks = [loglikelihood(forest, 2, psi, 1, 5, T_initial, as_log=False) for psi in psis]
    T = 1000
    # T = T_initial
    print(forest[0].get_ascii(attributes=['dist', TIME]))
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    n = sum(sum(1 for _ in tree.traverse()) for tree in forest)
    # for p2 in np.arange(0.5, 7.5, step=0.5):
    #     lks2_log = [loglikelihood(forest, p2 / scaling_factor, 1 / scaling_factor, 0.4, p1, T, as_log=True) + n * np.log(scaling_factor) for p1 in ps]
    #     print(p2, ps[np.argmax(lks2_log)], np.max(lks2_log))
    #     plt.plot(ps, lks2_log, '-', label=f'lk_log, p2={p2}')
    # plt.legend(loc='best')
    # plt.xlabel('r')
    # plt.show()

    LA = np.arange(0.5, 12.5, 1.5)
    PSI = np.arange(0.5, 4.5, 0.5)
    LA, PSI = np.meshgrid(LA, PSI)
    def lk_fun(la, psi, r):
        return loglikelihood(forest, la / scaling_factor, psi / scaling_factor, 0.4, r, T, as_log=True) + n * np.log(scaling_factor)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for r, cmap in zip([1.5, 2, 4, 5, 6], cmaps['Sequential']):
        vfunc = np.vectorize(lambda la, psi: lk_fun(la, psi, r))
        Z = vfunc(LA, PSI)

        # Plot the surface.
        surf = ax.plot_surface(LA, PSI, Z, linewidth=0, antialiased=False, alpha=0.6, cmap=cmap)
        # surf.actor.property.opacity = 0.5

        # # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, label=f'r={r}')
    ax.set_xlabel('lambda')
    ax.set_ylabel('psi')
    ax.set_zlabel('likelihood')

    plt.show()




if '__main__' == __name__:
    forest = read_forest('/home/azhukova/projects/bdpn/simulations_mult/minitree5.nwk')
    ps = np.arange(1.1, 10, step=0.25)
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    # node = TreeNode(dist=0.1)
    # node.add_feature(TIME, 0.1)
    # forest = [node]
    # lks_log = [loglikelihood(forest, 2, 1, 0.4, r, T_initial, as_log=True) for r in rs]
    # lks = [loglikelihood(forest, 2, psi, 1, 5, T_initial, as_log=False) for psi in psis]
    T = 1000
    # T = T_initial
    print(forest[0].get_ascii(attributes=['dist', TIME]))
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    n = sum(sum(1 for _ in tree.traverse()) for tree in forest)
    # for p2 in np.arange(0.5, 7.5, step=0.5):
    #     lks2_log = [loglikelihood(forest, p2 / scaling_factor, 1 / scaling_factor, 0.4, p1, T, as_log=True) + n * np.log(scaling_factor) for p1 in ps]
    #     print(p2, ps[np.argmax(lks2_log)], np.max(lks2_log))
    #     plt.plot(ps, lks2_log, '-', label=f'lk_log, p2={p2}')
    # plt.legend(loc='best')
    # plt.xlabel('r')
    # plt.show()

    R = np.arange(1.25, 7.25, step=.25)
    PSI = np.arange(0.25, 8.25, 0.25)
    PSI, R = np.meshgrid(PSI, R)
    def lk_fun(la, psi, r):
        return loglikelihood(forest, la / scaling_factor, psi / scaling_factor, 0.4, r, T, as_log=True) + n * np.log(scaling_factor)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # for la, cmap in zip([1, 2, 4], cmaps['Sequential']):
    for la, cmap in zip([2], ['rainbow']):
        vfunc = np.vectorize(lambda psi, r: lk_fun(la, psi, r))
        Z = vfunc(PSI, R)

        # Plot the surface.
        surf = ax.plot_surface(PSI, R, Z, linewidth=0, antialiased=False, alpha=0.6, cmap=cmap)
        # surf.actor.property.opacity = 0.5

        # # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, label=f'la={la}')
    ax.set_xlabel('psi')
    ax.set_ylabel('r')
    ax.set_zlabel('likelihood')

    plt.show()



if '__main0__' == __name__:
    forest = read_forest('/home/azhukova/projects/bdpn/simulations_mult/minitree5.nwk')
    annotate_forest_with_time(forest)
    T_initial = get_T(T=None, forest=forest)
    T = 1000
    scaling_factor = rescale_forest(forest, T_target=T, T=T_initial)
    n = sum(sum(1 for _ in tree.traverse()) for tree in forest)

    R0 = 3
    r = np.arange(3, 5.25, step=.25)
    def lk_fun(r0, r, la):
        psi = la / r0
        return loglikelihood(forest, la / scaling_factor, psi / scaling_factor, 0.4, r, T, as_log=True) + n * np.log(scaling_factor)


    for la in [1, 2 , 4]:
        vfunc = np.vectorize(lambda r: lk_fun(R0, r, la))
        print(vfunc(r))
    exit()



    R = np.arange(1.25, 7.25, step=.25)
    R0 = np.arange(1.001, 5.002, step=0.5)
    R0, R = np.meshgrid(R0, R)
    def lk_fun(r0, r, la):
        psi = la / r0
        return loglikelihood(forest, la / scaling_factor, psi / scaling_factor, 0.4, r, T, as_log=True) + n * np.log(scaling_factor)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for la, cmap in zip([1, 2, 4], cmaps['Sequential']):
    # for la, cmap in zip([2], ['rainbow']):
        vfunc = np.vectorize(lambda r0, r: lk_fun(r0, r, la))
        Z = vfunc(R0, R)

        # Plot the surface.
        surf = ax.plot_surface(R0, R, Z, linewidth=0, antialiased=False, alpha=0.6, cmap=cmap)
        # surf.actor.property.opacity = 0.5

        # # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, label=f'la={la}')
    ax.set_xlabel('R0')
    ax.set_ylabel('r')
    ax.set_zlabel('likelihood')

    plt.show()