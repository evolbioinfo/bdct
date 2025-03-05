import os

import numpy as np
import pandas as pd

CI_WIDTH_REL = 'CI_width_relative'
CI_WIDTH_ABS = 'CI_width_absolute'

WITHIN_CI = 'percent_within_CIs'

RATE_PARAMETERS = ['lambda', 'psi', 'phi']
PROBS = ['upsilon']
PARAMETERS = RATE_PARAMETERS + PROBS


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Access CIs.")
    parser.add_argument('--estimates', type=str, help="estimated parameters",
                        default=os.path.join(os.path.dirname(__file__), '..', 'trees', 'BDCT', 'estimates.tab'))
    parser.add_argument('--log', type=str, help="output log",
                        default=os.path.join(os.path.dirname(__file__), '..', 'trees', 'BDCT', 'CIs.log'))
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    df = df.loc[df['type'] != 'real', :]

    estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: (len(_), _)) if 'real' != _]
    data_types = sorted(df['data_type'].unique(), key=lambda _: _ == 'forest')



    result_df = pd.DataFrame(index=[f'{data_type}.{p}' for p in PARAMETERS for data_type in data_types] ,
                             columns=['type', 'parameter']
                                     + [f'{estimator_type}.{aspect}' for estimator_type in estimator_types
                                        for aspect in (WITHIN_CI, CI_WIDTH_REL, CI_WIDTH_ABS)
                                        ])
    for estimator_type in estimator_types:
        for data_type in data_types:
            mask = (df['type'] == estimator_type) & (df['data_type'] == data_type)
            idx = df.loc[mask, :].index
            n_observations = sum(mask)
            for par in PARAMETERS:
                result_df.loc[f'{data_type}.{par}', 'type'] = data_type
                result_df.loc[f'{data_type}.{par}', 'parameter'] = par
                if par == 'phi':
                    mask = (df['type'] == estimator_type) & (df['data_type'] == data_type) & (df['upsilon'] >= 1e-3)
                    idx = df.loc[mask, :].index
                    n_observations = sum(mask)
                if n_observations:
                    min_label = f'{par}_min'
                    max_label = f'{par}_max'
                    perc = 100 * sum((np.less_equal(df.loc[mask, min_label], real_df.loc[idx, par])
                                   | np.less_equal(np.abs(real_df.loc[idx, par] - df.loc[mask, min_label]), 1e-3)) \
                                  & (np.less_equal(real_df.loc[idx, par], df.loc[mask, max_label])
                                     | np.less_equal(np.abs(df.loc[mask, max_label] - real_df.loc[idx, par]), 1e-3))) \
                        / n_observations
                    result_df.loc[f'{data_type}.{par}', f'{estimator_type}.{WITHIN_CI}'] = \
                        f'{perc:.0f}%'
                    within_med = (100 * (df.loc[mask, max_label] - df.loc[mask, min_label]) \
                         / real_df.loc[idx, par]).median()
                    result_df.loc[f'{data_type}.{par}', f'{estimator_type}.{CI_WIDTH_REL}'] = \
                        f'{within_med:.1f}%'
                    if par in PROBS:
                        result_df.loc[f'{data_type}.{par}', f'{estimator_type}.{CI_WIDTH_ABS}'] = \
                            f'{(df.loc[mask, max_label] - df.loc[mask, min_label]).median():.2f}*'
    result_df.to_csv(params.log, sep='\t', index=False)