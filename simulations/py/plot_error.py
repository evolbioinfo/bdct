from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from statsmodels.stats.weightstats import CompareMeans

import re

RATE_PARAMETERS = ['lambda', 'psi', 'phi', 'upsilon']
EPIDEMIOLOGIC_PARAMETERS = ['R_naught', 'infectious_time', 'partner_removal_time', 'upsilon']
par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'phi': u'\u03c6', 'p': '\u03c1', 'upsilon': '\u03c5',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8',
             'infectious_time': '1' + u'\u002F\u03c8', 'partner_removal_time': '1' + u'\u002F\u03c6',
             'phi_by_psi': u'\u03c6 / \u03c8',
             'x': 'x', 'f_ss': 'f_ss', 'f_inc': 'f_i'}


EST_ORDER = ['bd', 'bddl', 'bdct', 'bdct1dl', 'bdct2dl', 'bdct2000dl',
             'bdeidl', 'bdeict1dl', 'bdeict2dl', 'bdeict2000dl',
             'bdssdl', 'bdssct1dl', 'bdssct2dl', 'bdssct2000dl',
             'mfdl']
PARAMETERS = RATE_PARAMETERS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    parser.add_argument('--tab', type=str, help="error table")
    parser.add_argument('--only_trees', action='store_true')
    # parser.add_argument('--upsilon_min', type=float, default=0., help="Only display cases with upsilon greater or equal to this value")
    # parser.add_argument('--upsilon_max', type=float, default=1, help="Only display cases with upsilon smaller than this value")
    params = parser.parse_args()


    model, kappa = re.findall(r'(BDEI|BDSS|BD)CT(\d+)', params.estimates)[0]
    kappa = int(kappa)
    fig_title = f'{model}{f"-CT({kappa})" if kappa > 0 else ""}'
    if kappa == 0:
        PARAMETERS.remove('phi')

    print(f'\n\n==========================={fig_title}==============\n')

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    # real_df = real_df[(real_df['upsilon'] * real_df['p'] >= params.upsilon_min) & (real_df['upsilon'] * real_df['p'] < params.upsilon_max)]

    df = df.loc[df['type'] != 'real', :]
    estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: EST_ORDER.index(_)) if 'real' != _]
    data_types = sorted(df['data_type'].unique(), key=lambda _: _ == 'forest')
    if params.only_trees:
        data_types = [_ for _ in data_types if _ != 'forest']

    for estimator_type in estimator_types:
        mask = df['type'] == estimator_type
        idx = df.loc[mask, :].index
        for par in PARAMETERS:
            # df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par]) / real_df[par]
            if par != 'p' and par != 'upsilon' and par != 'f_inc' and par != 'f_ss':
                df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                # if par == 'phi' or par == 'partner_removal_time':
                #     zero_ups_mask = pd.isna(df['upsilon']) | (df['upsilon'] == 0)
                #     df.loc[mask & zero_ups_mask, '{}_error'.format(par)] = np.nan
            else:
                df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df.loc[idx, par])

    error_columns = [col for col in df.columns if 'error' in col]
    df[['type'] + PARAMETERS + error_columns].to_csv(params.tab, sep='\t')

    plt.clf()

    rc = {'font.size': 14, 'axes.labelsize': 12, 'legend.fontsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    # sns.set(style="whitegrid")
    sns.axes_style(style="whitegrid", rc=rc)

    abs_error_or_1 = lambda _: min(abs(_), 1)
    error_or_1 = lambda _: max(min(_, 1), -1)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig, ax1 = plt.subplots(1, 1, figsize=(25 if kappa else 17, 8))
    # for pars, ax in ((RATE_PARAMETERS, ax1), (EPIDEMIOLOGIC_PARAMETERS, ax2)):
    # for pars, ax in ((['R_naught', 'infectious_time'], ax1), (['upsilon', 'partner_removal_time'], ax2)):
    for pars, ax in ((RATE_PARAMETERS, ax1),):
        data = []
        par2type2avg_error = defaultdict(lambda: dict())
        par2type2bias = defaultdict(lambda: dict())

        est_labels = []
        for estimator_type in estimator_types:
            for data_type in data_types:
                estimator_type_label = f'{estimator_type} - {data_type}'
                est_labels.append(estimator_type_label)

                for par in pars:
                    cur_mask = (df['type'] == estimator_type) & (df['data_type'] == data_type)
                    # data.extend([[par2greek[par], _, estimator_type_label]
                    #              for _ in df.loc[cur_mask,
                    #     '{}_error'.format(par)].apply(abs_error_or_1)])
                    data.extend([[par2greek[par], _, estimator_type_label]
                                 for _ in df.loc[cur_mask,
                        '{}_error'.format(par)].apply(error_or_1)])
                    par2type2avg_error[par][estimator_type_label] = \
                        f'{np.mean(np.abs(df.loc[cur_mask, f"{par}_error"])):.2f}'
                    par2type2bias[par][estimator_type_label] = \
                        f'{np.mean(df.loc[cur_mask, f"{par}_error"]):.2f}'

        # n_types = len(est_labels)
        #
        # type_vs_type2pars = defaultdict(list)
        # par2types2pval = defaultdict(lambda: dict())
        # for par in pars:
        #     for i in range(n_types):
        #         type_1 = est_labels[i]
        #         estimator_type_1, data_type_1 = type_1.split(' - ')
        #         mask1 = (df['type'] == estimator_type_1) & (df['data_type'] == data_type_1)
        #         for j in range(i + 1, n_types):
        #             type_2 = est_labels[j]
        #             estimator_type_2, data_type_2 = type_2.split(' - ')
        #             mask2 = (df['type'] == estimator_type_2) & (df['data_type'] == data_type_2)
        #             pval_abs = \
        #                 CompareMeans.from_data(data1=df.loc[mask1, '{}_error'.format(par)].apply(np.abs),
        #                                        data2=df.loc[mask2, '{}_error'.format(par)].apply(np.abs)).ztest_ind()[1]
        #             par2types2pval[par][(type_1, type_2)] = pval_abs
        #             if pval_abs < 0.05:
        #                 type_vs_type2pars[(type_1, type_2)].append(par)
        #
        # for type_vs_type, pars in type_vs_type2pars.items():
        #     print(f"{' vs '.join(type_vs_type)}:\t{'all' if len(pars) == 4 else ', '.join(pars)}")



        ERROR_COL = 'relative error' if 'upsilon' not in pars else 'relative or absolute (for {}) error'.format(par2greek['upsilon'])
        plot_df = pd.DataFrame(data=data, columns=['parameter', ERROR_COL, 'config'])

        tree_palette = sns.color_palette()
        if params.only_trees:
            palette = [tree_palette[0], tree_palette[4], tree_palette[2], tree_palette[1], tree_palette[3]] + tree_palette[5:]
        else:
            forest_palette = sns.color_palette("pastel")
            palette = [item for pair in zip([tree_palette[0], tree_palette[4], tree_palette[2], tree_palette[1], tree_palette[3]] + tree_palette[5:],
                                            [forest_palette[0], forest_palette[4], forest_palette[2], forest_palette[1], forest_palette[3]] + forest_palette[5:]) for item in pair]

        ax = sns.stripplot(x="parameter", y=ERROR_COL, palette=palette, data=plot_df, alpha=.75, hue="config", ax=ax,
                           dodge=True, jitter=0.4, size=5, edgecolor='grey', linewidth=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        min_error = min(min(df['{}_error'.format(_)]) for _ in pars)
        max_error = max(max(df['{}_error'.format(_)]) for _ in pars)
        abs_error = max(max_error, abs(min_error))
        # ax.set_yticks(list(np.arange(0, min(1.1, abs_error + 0.1), step=0.2 if abs_error >= 1 else 0.1)))


        # ticks = list(np.arange(max(-1, (min_error // 0.1) * 0.1), min(1.1, max_error + 0.1),
        #                    step=0.2 if abs_error >= 1 else 0.1))
        # if abs_error >= 1:
        #     ax.set_yticklabels([u"\u22641" if np.abs(tick + 1) <= 1e-3 else (u"\u22651" if np.abs(tick - 1) <= 1e-3 else f'{tick:.1f}')
        #                         for tick in ticks])
        # ax.set_ylim(0, min(1.1, abs_error + 0.1))
        # y_min = max(-1.1, (min_error // 0.1) * 0.1 - 0.1)
        # y_max = min(1.1, max_error + 0.1)
        # print(y_min, y_max, min_error, max_error)

        ax.axhline(y=0, xmin=0, xmax=1)
        # ax.set_facecolor('w')
        ticks = list(np.arange(-1, 1.1, 0.1).astype(float))
        ax.set_yticks(ticks)
        ax.set_yticklabels([u"\u22641"] + [f'{tick:.1f}' for tick in ticks[1:-1]] + [u"\u22651"])
        y_min, y_max = -1.1, 1.1

        ax.set_ylim(y_min, y_max)
        ax.yaxis.grid()

        def get_xbox(par):

            def get_ta(color, text):
                return TextArea(text,
                                textprops=dict(color=color, ha='center', va='center', fontsize=10,
                                               fontweight='bold'))

            return HPacker(children=[VPacker(children=[get_ta(color, text_err),
                                                       get_ta(color, text_bias)],
                                             align="center", pad=1, sep=4)
                                     for (text_err, text_bias, color)
                                     in zip((par2type2avg_error[par][_] for _ in est_labels),
                                            (par2type2bias[par][_] for _ in est_labels),
                                            palette)],
                           align="center", pad=0, sep=0)




        xbox = HPacker(children=[get_xbox(par) for par in pars], align="center", pad=0, sep=8)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                          bbox_to_anchor=(0, -0.12),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.set_xlabel('')
        ax.add_artist(anchored_xbox)

        # def get_pbox(par):
        #     EMPTY = ' ' * 10
        #     LONG_DASH = u"\u2014"
        #     FILLED = LONG_DASH * 10
        #     boxes = []
        #     for i in range(n_types - 1):
        #         type_1 = est_labels[i]
        #         s = EMPTY * max(0, i)
        #         for j in range(i + 1, n_types):
        #             type_2 = est_labels[j]
        #             pval = par2types2pval[par][(type_1, type_2)]
        #             if pval < 0.05:
        #                 print(par, type_1, type_2, pval)
        #                 boxes.append(TextArea(s + LONG_DASH * 3 + '{:g}'.format(pval) + LONG_DASH * 3 + EMPTY * (n_types - j - 1),
        #                                       textprops=dict(color='black', ha='center', va='center',
        #                                                      fontsize='x-small', fontweight='bold', family='monospace')))
        #             else:
        #                 boxes.append(TextArea(EMPTY * n_types,
        #                                       textprops=dict(color='black', ha='center', va='center',
        #                                                      fontsize='x-small', fontweight='bold', family='monospace')))
        #             s += FILLED
        #     return VPacker(children=list(reversed(boxes)), mode='equal', pad=0, sep=3) if len(boxes) > 1 else boxes[0]
        #
        # xbox = HPacker(children=[get_pbox(par) for par in pars], align="center", pad=0, sep=20)
        # anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
        #                                   bbox_to_anchor=(0, 1),
        #                                   bbox_transform=ax.transAxes, borderpad=0.)
        # ax.add_artist(anchored_xbox)
        # ax.set_xlabel('')

        leg = ax.legend()
        # if pars != RATE_PARAMETERS:
        #     leg.remove()

    # plt.tight_layout()
    # fig.set_size_inches(9, 9)
    # plt.show()
    plt.title(fig_title)
    plt.savefig(params.pdf, dpi=300)
