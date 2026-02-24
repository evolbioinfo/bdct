import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

# PARAMETERS = ['lambda', 'avg la', 'psi', 'avg psi', 'avg psi 2', 'R', 'R2'] #['f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'pi_E', 'pi_I', 'pi_S', 'pi_E-C', 'pi_I-C', 'pi_S-C']
PARAMETERS = ['lambda', 'psi', 'r', 'X_S', 'f_S'] #, 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C'] #, 'f_E', 'f_S', 'X_S', 'X_C', 'upsilon']
# PARAMETERS = ['f_E', 'f_S', 'upsilon', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C'] #, 'f_E', 'f_S', 'X_S', 'X_C', 'upsilon']
# PARAMETERS = ['avg la', 'R', 'd', 'd1'] #, 'f_E', 'f_S', 'X_S', 'X_C', 'upsilon']
par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'R': 'R', 'd': 'd', 'd_E': 'd_E', 'phi': u'\u03c6',
             'p': '\u03c1', 'upsilon': '\u03c5',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8', 'r': 'r',
             'infectious_time': '1' + u'\u002F\u03c8', 'partner_removal_time': '1' + u'\u002F\u03c6'}
for p in PARAMETERS:
    if p not in par2greek:
        par2greek[p] = p

greek2par = {v: k for k, v in par2greek.items()}

EST_ORDER = ['bd', 'bdmult',] # 'bdssmult'] #, 'bdei', 'bdeidl', 'bdssdl', 'bdeissdl', 'bdctdl', 'bdeictdl', 'bdssctdl', 'bdeissctdl']
# EST_ORDER = ['bd', 'bddl', 'bdeidl', 'bdssdl', 'bdeissdl', 'bdctdl', 'bdeictdl', 'bdssctdl', 'bdeissctdl']

BIAS_COL = 'bias'
ERROR_COL = 'error'

total_palette = sns.color_palette("colorblind")


def pertinent_estimators(model, palette):
    ei_model = 'ei' in model.lower()
    ss_model = 'ss' in model.lower()
    ct_model = 'ct' in model.lower()

    result = []
    res_palette = []

    for estimator, col in zip(EST_ORDER, palette):
        ei_estimator = 'ei' in estimator.lower()
        ss_estimator = 'ss' in estimator.lower()
        ct_estimator = 'ct' in estimator.lower()
        bd_estimator = estimator.lower() in {'bd', 'bddl'}

        # if not bd_estimator:
        #     if (ei_model and not ss_model) and not ei_estimator \
        #             or (ss_model and not ei_model) and not ss_estimator \
        #             or (not ei_model and ct_model) and (ei_estimator and not ct_estimator)\
        #             or (not ss_model and ct_model) and (ss_estimator and not ct_estimator):
        #         continue
        result.append(estimator)
        res_palette.append(col)
    return result, res_palette


def need_to_skip(par, estimator_type, model):

    if estimator_type.lower() in ['bd', 'bddl'] and par.startswith('pi'):
        return True
    if ('X_C' in par or 'upsilon' in par or par.startswith('pi') and par.endswith('C')) and ('ct' not in estimator_type.lower() or 'ct' not in model.lower()):
        return True
    if ('d_E' in par or 'f_E' in par or par.startswith('pi_E')) and ('ei' not in estimator_type.lower() or 'ei' not in model.lower()):
        return True
    if ('f_S' in par or 'X_S' in par or par.startswith('pi_S')) and ('ss' not in estimator_type.lower() or 'ss' not in model.lower()):
        return True
    return False

folder = '/home/azhukova/projects/bdct/simulations_mult/500_1000'
estimate_files = [f'{folder}/{model}/estimates.tab' for model in ['BD', 'BDMULT', 'BDSSMULT']]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', default=estimate_files, type=str, nargs='+', help="estimated parameters")
    parser.add_argument('--pdf', default=f'{folder}/estimates.svg', type=str, help="plot")
    params = parser.parse_args()

    plt.clf()
    rc = {'font.size': 24, 'axes.labelsize': 24, 'legend.fontsize': 20, 'axes.titlesize': 30, 'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    # sns.set(style="whitegrid")
    sns.axes_style(style="whitegrid", rc=rc)
    fig, axs = plt.subplots(len(params.estimates), 2, figsize=(20, 2 * len(params.estimates)))


    order = ['', ' ']
    for i, _ in enumerate(PARAMETERS, start=1):
        order.append(par2greek[_])
        order.append(' ' * (2 * i))
        order.append(' ' * (2 * i + 1))


    for num_est, estimate in enumerate(params.estimates):

        model = re.findall(r'BDSSMULT|BDMULT|BD', estimate)[0]
        fig_title = model



        print(f'\n\n==========================={fig_title}==============\n')

        df = pd.read_csv(estimate, sep='\t', index_col=0)
        real_df = df.loc[df['type'] == 'real', :]
        df = df.loc[df['type'] != 'real', :]
        estimator_types, palette = pertinent_estimators(model, total_palette)
        # estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: EST_ORDER.index(_)) if 'real' != _]

        for estimator_type in estimator_types:
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            for par in PARAMETERS:
                if need_to_skip(par, estimator_type, model):
                    continue

                df.loc[mask, f'{par}_error'] = df.loc[mask, par] - real_df.loc[idx, par]
                if par != 'upsilon' and par != 'f_E' and par != 'f_S' and not par.startswith('pi'):
                    df.loc[mask, f'{par}_error'] /= np.where(real_df.loc[idx, par] > 0, real_df.loc[idx, par], 1)

                # if par != 'p' and par != 'upsilon' and par != 'f_E' and par != 'f_S' and not par.startswith('pi'):
                #     df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                # else:
                #     df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par])


        data = []
        par2type2avg_error = defaultdict(lambda: dict())
        par2type2bias = defaultdict(lambda: dict())

        est_labels = []
        for estimator_type in estimator_types:
                estimator_type_label = f'{estimator_type}'
                est_labels.append(estimator_type_label)

                for par in PARAMETERS:
                    if need_to_skip(par, estimator_type, model):
                        par2type2avg_error[par][estimator_type_label] = '___'
                        par2type2bias[par][estimator_type_label] = '___'
                    else:
                        cur_mask = (df['type'] == estimator_type)
                        if 'X_C' in par:
                            data.extend([[par2greek[par], _, estimator_type_label]
                                         for _ in np.where(df.loc[cur_mask, 'upsilon'] <= 0.001, 0,
                                                           df.loc[cur_mask, f'{par}_error'])])
                        else:
                            data.extend([[par2greek[par], _, estimator_type_label]
                                         for _ in df.loc[cur_mask, f'{par}_error']])
                        if 'X_C' in par:
                            cur_mask &= df['upsilon'] > 0.001
                        if cur_mask.sum() == 0:
                            par2type2avg_error[par][estimator_type_label] = '___'
                            par2type2bias[par][estimator_type_label] = '___'
                        else:
                            par2type2avg_error[par][estimator_type_label] = \
                                f'{100 * np.mean(np.abs(df.loc[cur_mask, f"{par}_error"])):3.0f}'
                            par2type2bias[par][estimator_type_label] = \
                                f'{100 * np.mean(df.loc[cur_mask, f"{par}_error"]):3.0f}'

        plot_df = pd.DataFrame(data=data, columns=['parameter', BIAS_COL, 'config'])
        plot_df[ERROR_COL] = np.abs(plot_df[BIAS_COL])


        for ax, col in zip(axs[num_est] if len(params.estimates) > 1 else axs, (ERROR_COL, BIAS_COL)):
            ax = sns.barplot(data=plot_df, x="parameter", y=col, hue="config", estimator='mean', palette=total_palette,
                             ax=ax, order=order, errorbar='ci', gap=0.1, width=2.4, hue_order=EST_ORDER)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Add hatches on CT bars
            plot_pars = [t.get_text() for t in ax.get_xticklabels() if t.get_text().strip()]
            plot_estimators = [t.get_text() for t in ax.get_legend().get_texts()]
            for par_bars, plot_est in zip(ax.containers, plot_estimators):
                if plot_est not in estimator_types:
                    continue
                bar_idx = 0
                for plot_par in plot_pars:
                    if not need_to_skip(greek2par[plot_par], plot_est, model):
                        hatch = ''
                        if 'ct' in plot_est.lower():
                            hatch += '----'
                        if 'dl' not in plot_est.lower():
                            hatch += '////'
                        # if 'ss' in plot_est.lower():
                        #     hatch += '\\\\\\\\'
                        if hatch:
                            par_bars[bar_idx].set_hatch(hatch)
                        bar_idx += 1

            if BIAS_COL == col:
                ax.axhline(y=0, xmin=0, xmax=1)
            ticks = list(np.arange(-0.6 if BIAS_COL == col else 0, 0.61 if BIAS_COL == col else 0.7, 0.2 if BIAS_COL == col else 0.1).astype(float))
            ax.set_yticks(ticks)
            ax.set_ylim(-0.61 if BIAS_COL == col else 0, 0.61 if BIAS_COL == col else 0.71)
            ax.set_yticklabels([f'{100 * _:.0f}%' for _ in ticks])

            def get_xbox(par):

                def get_ta(color, text):
                    return TextArea(text,
                                    textprops=dict(color=color, ha='center', va='center', fontsize=11,
                                                   fontweight='bold'))

                return HPacker(children=[get_ta(color, text_err) if col == ERROR_COL else get_ta(color, text_bias)
                                         for (text_err, text_bias, color)
                                         in zip((par2type2avg_error[par][_] for _ in est_labels),
                                                (par2type2bias[par][_] for _ in est_labels),
                                                palette + palette)],
                               align="center", pad=0, sep=0)




            xbox = HPacker(children=[get_xbox(par) for par in PARAMETERS], align="center", pad=0, sep=30)
            anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                              bbox_to_anchor=(0.15, -0.15),
                                              bbox_transform=ax.transAxes, borderpad=0.)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.add_artist(anchored_xbox)
            ax.set_xticks([])

            leg = ax.legend()
            if num_est > 0 or col == BIAS_COL:
                leg.remove()

            ax.set_title(fig_title, loc='center', y=0.75)

    # plt.tight_layout()
    # fig.set_size_inches(9, 9)
    # plt.show()
    # plt.title(fig_title)
    plt.savefig(params.pdf, dpi=100)
