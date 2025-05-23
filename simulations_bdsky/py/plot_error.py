from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker

import re

RATE_PARAMETERS = ['lambda1', 'psi1', 'lambda2', 'psi2', 'T1']
EPIDEMIOLOGIC_PARAMETERS = ['R_naught1', 'infectious_time1', 'R_naught2', 'infectious_time2']
par2greek = {'lambda1': u'\u03bb1', 'psi1': u'\u03c81', 'p1': '\u03c11',
             'lambda2': u'\u03bb2', 'psi2': u'\u03c82', 'p2': '\u03c12',
             'R_naught1': u'\u0052\u20801' + '=' + u'\u03bb1\u002F\u03c81',
             'R_naught2': u'\u0052\u20802' + '=' + u'\u03bb2\u002F\u03c82',
             'infectious_time1': '1' + u'\u002F\u03c81',
             'infectious_time2': '1' + u'\u002F\u03c82',
             'T1': 'T1'}


EST_ORDER = ['bdsky', 'bdsky_t']
PARAMETERS = RATE_PARAMETERS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    parser.add_argument('--tab', type=str, help="error table")
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    df = df.loc[df['type'] != 'real', :]
    estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: EST_ORDER.index(_)) if 'real' != _]

    for estimator_type in estimator_types:
        mask = df['type'] == estimator_type
        idx = df.loc[mask, :].index
        for par in PARAMETERS:
            # df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par]) / real_df[par]
            if par != 'p1' and par != 'p2':
                df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
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
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    # for pars, ax in ((RATE_PARAMETERS, ax1), (EPIDEMIOLOGIC_PARAMETERS, ax2)):
    # for pars, ax in ((['R_naught', 'infectious_time'], ax1), (['upsilon', 'partner_removal_time'], ax2)):
    for pars, ax in ((RATE_PARAMETERS, ax1),):
        data = []
        par2type2avg_error = defaultdict(lambda: dict())
        par2type2bias = defaultdict(lambda: dict())

        for estimator_type in estimator_types:
            for par in pars:
                cur_mask = (df['type'] == estimator_type)
                data.extend([[par2greek[par], _, estimator_type]
                             for _ in df.loc[cur_mask, '{}_error'.format(par)].apply(error_or_1)])
                par2type2avg_error[par][estimator_type] = \
                    f'{np.mean(np.abs(df.loc[cur_mask, f"{par}_error"])):.2f}'
                par2type2bias[par][estimator_type] = \
                    f'{np.mean(df.loc[cur_mask, f"{par}_error"]):.2f}'


        ERROR_COL = 'error'
        plot_df = pd.DataFrame(data=data, columns=['parameter', ERROR_COL, 'config'])

        palette = sns.color_palette()

        ax = sns.stripplot(x="parameter", y=ERROR_COL, palette=palette, data=plot_df, alpha=.75, hue="config", ax=ax,
                           dodge=True, jitter=0.4, size=5, edgecolor='grey', linewidth=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        min_error = min(min(df['{}_error'.format(_)]) for _ in pars)
        max_error = max(max(df['{}_error'.format(_)]) for _ in pars)
        abs_error = max(max_error, abs(min_error))

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
                                     in zip((par2type2avg_error[par][_] for _ in estimator_types),
                                            (par2type2bias[par][_] for _ in estimator_types),
                                            palette)],
                           align="center", pad=0, sep=2)




        xbox = HPacker(children=[get_xbox(par) for par in pars], align="center", pad=0, sep=20)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                          bbox_to_anchor=(0.02, -0.18),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.set_xlabel('')
        ax.add_artist(anchored_xbox)

        leg = ax.legend()
        leg.remove()

    plt.tight_layout()
    # fig.set_size_inches(6, 4.5)
    # plt.show()
    plt.title('BDSKY estimator performance')
    plt.savefig(params.pdf, dpi=300)
