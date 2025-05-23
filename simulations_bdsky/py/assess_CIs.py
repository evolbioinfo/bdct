import pandas as pd

CI_WIDTH_REL = 'CI_width_relative'
CI_WIDTH_ABS = 'CI_width_absolute'

WITHIN_CI = 'percent_within_CIs'

PARAMETERS = ['lambda1', 'psi1', 'lambda2', 'psi2', 'T1']
PROBS = ['p1', 'p2']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Access CIs.")
    parser.add_argument('--estimates', default='/home/azhukova/projects/bdct/simulations_bdsky/trees/estimates.tab',
                        type=str, help="estimated parameters")
    parser.add_argument('--log', type=str, help="output log",
                        default='/home/azhukova/projects/bdct/simulations_bdsky/trees/CIs.log')
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    df = df.loc[df['type'] != 'real', :]

    estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: (len(_), _)) if 'real' != _]



    result_df = pd.DataFrame(index=PARAMETERS,
                             columns=['parameter']
                                     + [f'{estimator_type}.{aspect}' for estimator_type in estimator_types
                                        for aspect in (WITHIN_CI, CI_WIDTH_REL) #, CI_WIDTH_ABS)
                                        ])
    for estimator_type in estimator_types:
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            n_observations = sum(mask)
            for par in PARAMETERS:
                result_df.loc[par, 'parameter'] = par
                if n_observations:
                    min_label = f'{par}_min'
                    max_label = f'{par}_max'

                    mask_within = (real_df.loc[idx, par] - df.loc[mask, min_label] >= -1e-3) & (df.loc[mask, max_label] - real_df.loc[idx, par] >= -1e-3)

                    perc = 100 * sum(mask_within) / n_observations
                    result_df.loc[par, f'{estimator_type}.{WITHIN_CI}'] = \
                        f'{perc:.0f}%'
                    within_med = (100 * (df.loc[mask, max_label] - df.loc[mask, min_label]) \
                         / real_df.loc[idx, par]).median()
                    result_df.loc[par, f'{estimator_type}.{CI_WIDTH_REL}'] = \
                        f'{within_med:.1f}%'
                    # if par in PROBS:
                    #     result_df.loc[par, f'{estimator_type}.{CI_WIDTH_ABS}'] = \
                    #         f'{(df.loc[mask, max_label] - df.loc[mask, min_label]).median():.2f}*'
    result_df.to_csv(params.log, sep='\t', index=False)