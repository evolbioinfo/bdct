from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from statsmodels.stats.weightstats import CompareMeans

import re

par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'phi': u'\u03c6', 'p': '\u03c1', 'upsilon': '\u03c5',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8',
             'infectious_time': '1' + u'\u002F\u03c8', 'partner_removal_time': '1' + u'\u002F\u03c6',
             'phi_by_psi': u'\u03c6 / \u03c8',
             'x': 'x' + u'\u209B\u209B', 'f_ss': 'f' + u'\u209B\u209B', 'f_inc': 'f' + u'\u1D62'}


EST_ORDER = ['bd', 'bddl', 'bdct', 'bdct1dl', 'bdct2dl', 'bdct2000dl',
             'bdeidl', 'bdeict1dl', 'bdeict2dl', 'bdeict2000dl',
             'bdssdl', 'bdssct1dl', 'bdssct2dl', 'bdssct2000dl',
             'mfdl']

rc = {'font.size': 16, 'axes.labelsize': 14, 'legend.fontsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 18,
      'ytick.labelsize': 14}
sns.axes_style(style="whitegrid", rc=rc)

BIAS_COL = 'bias'
ERROR_COL = 'error'


BD_PARAMETERS = ['lambda', 'psi', 'upsilon', 'phi']
BDEI_PARAMETERS = ['lambda', 'psi', 'f_inc', 'upsilon', 'phi']
BDSS_PARAMETERS = ['lambda', 'psi', 'f_ss', 'x', 'upsilon', 'phi']
ALL_PARAMETERS = ['lambda', 'psi', 'f_inc', 'f_ss', 'x', 'upsilon', 'phi']

model2params = {'BD': BD_PARAMETERS, 'BDEI': BDEI_PARAMETERS, 'BDSS': BDSS_PARAMETERS, 'ALL': ALL_PARAMETERS}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    parser.add_argument('--only_trees', action='store_true')
    params = parser.parse_args()

    palette = sns.color_palette("colorblind")

    n_models = len(params.estimates)
    mtbd_models = {re.findall(r'(BDEI|BDSS|BD)CT', estimates)[0] for estimates in params.estimates}
    n_params = len(BDSS_PARAMETERS if 'BDSS' in mtbd_models else BDEI_PARAMETERS if 'BDEI' in mtbd_models else BD_PARAMETERS)
    estimators = set()
    for m in mtbd_models:
        estimators |= {_ for _ in EST_ORDER if (m.lower() == _.split('ct')[0].replace('dl', '').replace('ml', '') or 'mf' in _)}
    n_estimators = len(estimators)
    n_ct_estimators = len([_ for _ in estimators if 'mf' in _ or 'ct' in _])
    fig, axes = plt.subplots(2, 2, figsize=(2 * 0.2 * ((n_params - 2) * n_estimators + 2 * n_ct_estimators) + 0.05 * (n_params + 1), 3 * n_models / 2))
    axes = np.array(axes).flatten()

    for ax, estimates in zip(axes, params.estimates):

        model, kappa = re.findall(r'(BDEI|BDSS|BD)CT(\d+)', estimates)[0]
        kappa = int(kappa)
        model_label = f'{model}{f"-CT({kappa})" if kappa > 0 else ""}'
        ax.set_title(model_label)

        parameters = list(model2params[model])
        # if kappa == 0:
        #     parameters.remove('phi')

        df = pd.read_csv(estimates, sep='\t', index_col=0)
        df['phi_by_psi'] = df['phi'] / df['psi']
        real_df = df.loc[df['type'] == 'real', :]
        df = df.loc[df['type'] != 'real', :]
        estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: EST_ORDER.index(_)) if 'real' != _ and (model.lower() == _.split('ct')[0].replace('dl', '').replace('ml', '') or 'mf' in _)]

        for estimator_type in estimator_types:
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            for par in parameters:
                if ('phi' in par or 'upsilon' in par) and estimator_type in ('bd', 'bddl', 'bdeidl', 'bdssdl'):
                    continue
                if ('f_ss' in par or 'x' in par) and ('bdss' not in estimator_type and 'mf' not in estimator_type):
                    continue
                if ('f_inc' in par) and ('bdei' not in estimator_type and 'mf' not in estimator_type):
                    continue
                if par != 'p' and par != 'upsilon' and par != 'f_inc' and par != 'f_ss':
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                    if 'phi' in par:
                        df.loc[mask, f'{par}_error'] = np.where(np.abs(df.loc[mask, 'upsilon']) <= 1e-3, 0, df.loc[mask, f'{par}_error'])
                    if 'x' in par:
                        df.loc[mask, f'{par}_error'] = np.where(np.abs(df.loc[mask, 'f_ss']) <= 1e-3, 0, df.loc[mask, f'{par}_error'])
                else:
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par])

        data = []
        par2type2avg_error = defaultdict(lambda: defaultdict(None))
        par2type2bias = defaultdict(lambda: defaultdict(None))

        for estimator_type in estimator_types:
            for par in parameters:
                if (('phi' in par or 'upsilon' in par) and estimator_type in ('bd', 'bddl', 'bdeidl', 'bdssdl')) \
                        or (('f_ss' in par or 'x' in par) and ('bdss' not in estimator_type and 'mf' not in estimator_type)) \
                        or (('f_inc' in par) and ('bdei' not in estimator_type and 'mf' not in estimator_type)):
                    par2type2avg_error[par][estimator_type] = ''
                    par2type2bias[par][estimator_type] = ''
                else:
                    cur_mask = df['type'] == estimator_type
                    data.extend([[par2greek[par], _, estimator_type] for _ in df.loc[cur_mask, f'{par}_error']])
                    par2type2avg_error[par][estimator_type] = f'{np.mean(np.abs(df.loc[cur_mask, f"{par}_error"])):.2f}'
                    par2type2bias[par][estimator_type] = f'{np.mean(df.loc[cur_mask, f"{par}_error"]):.2f}'

        # n_types = len(estimator_types)
        # type_vs_type2pars = defaultdict(list)
        # par2types2pval = defaultdict(lambda: dict())
        # type2par2errs = defaultdict(lambda: dict())
        # for i in range(n_types):
        #     estimator_type = estimator_types[i]
        #     mask = (df['type'] == estimator_type)
        #     for par in parameters:
        #         err = df.loc[mask, '{}_error'.format(par)]
        #         type2par2errs[estimator_type][par] = err
        #
        #
        # for par in parameters:
        #     for i in range(n_types):
        #         estimator_type_1 = estimator_types[i]
        #         err1 = type2par2errs[estimator_type_1][par].apply(np.abs)
        #         for j in range(i + 1, n_types):
        #             estimator_type_2 = estimator_types[j]
        #             err2 = type2par2errs[estimator_type_2][par].apply(np.abs)
        #             pval_abs = CompareMeans.from_data(data1=err1, data2=err2).ztest_ind()[1]
        #             par2types2pval[par][(estimator_type_1, estimator_type_2)] = pval_abs
        #             if pval_abs < 0.01:
        #                 type_vs_type2pars[(estimator_type_1, estimator_type_2)].append(par)
        #
        # # print(f'===={model_label}===:')
        # for estimator_type in estimator_types:
        #     print(f'{estimator_type}:\t{', '.join(f'{par}: {err.apply(np.abs).mean()} ({err.mean()})' for (par, err) in type2par2errs[estimator_type].items())}')
        #     print(f'{estimator_type}:\tmax upsilon:{type2par2errs[estimator_type]['upsilon'].max()}\tmax upsilon_min:{type2par2errs[estimator_type]['upsilon_min'].max()}\n')
        #
        # for type_vs_type, pars in type_vs_type2pars.items():
        #     print(f"{' vs '.join(type_vs_type)}:\t{'all' if len(pars) == 4 else ', '.join(pars)}")

        plot_df = pd.DataFrame(data=data, columns=['parameter', BIAS_COL, 'config'])
        plot_df[ERROR_COL] = np.abs(plot_df[BIAS_COL])

        ax = sns.barplot(data=plot_df, x="parameter", y=ERROR_COL, order=[par2greek[_] for _ in parameters],
                         hue="config", estimator='mean', palette=palette, ax=ax, errorbar=None)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ticks = list(np.arange(0, 1.1, 0.1).astype(float))
        ax.set_yticks(ticks)
        ax.set_yticklabels(['0'] + [f'{tick:.1f}' for tick in ticks[1:-1]] + [u"\u22651"])
        ax.set_ylim(0, 1.1)
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
                                     in zip((par2type2avg_error[par][_] for _ in estimator_types if _ in par2type2avg_error[par]),
                                            (par2type2bias[par][_] for _ in estimator_types if _ in par2type2avg_error[par]),
                                            palette)],
                           align="center", pad=0, sep=1)

        xbox = HPacker(children=[get_xbox(par) for par in parameters], align="center", pad=0, sep=30)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False, bbox_to_anchor=(0.02, -0.3),
                                          bbox_transform=ax.transAxes, borderpad=0.4)
        ax.set_xlabel('')
        ax.add_artist(anchored_xbox)

        leg = ax.legend()
        if ax != axes[0]:
            leg.remove()

    plt.tight_layout()
    plt.savefig(params.pdf, dpi=300)
