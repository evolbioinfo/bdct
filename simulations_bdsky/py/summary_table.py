import re

import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--estimates_bdsky', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdsky_t', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--real', nargs='+', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    df = pd.DataFrame(columns=['type', 'tips',
                               'lambda1', 'lambda1_min', 'lambda1_max',
                               'psi1', 'psi1_min', 'psi1_max',
                               'p1', 'p1_min', 'p1_max',
                               'lambda2', 'lambda2_min', 'lambda2_max',
                               'psi2', 'psi2_min', 'psi2_max',
                               'p2', 'p2_min', 'p2_max',
                               'T1', 'T1_min', 'T1_max',
                               ])

    for real in params.real:
        i = int(re.findall(r'[0-9]+', real)[-1])
        ddf = pd.read_csv(real)
        psi1, psi2 = ddf.loc[:, 'psi_I']
        rho1, rho2 = ddf.loc[:, 'p_I']
        la1, la2 = ddf.loc[:, 'la_II']
        tips = ddf.loc[1, 'tips']
        T1 = ddf.loc[0, 'end_time']

        df.loc[f'{i}.real',
        ['lambda1', 'psi1', 'p1', 'lambda2', 'psi2', 'p2', 'T1', 'tips', 'type']] \
            = [la1, psi1, rho1, la2, psi2, rho2, T1, tips, 'real']

    for est_label, estimates in (('bdsky', params.estimates_bdsky), ('bdsky_t', params.estimates_bdsky_t)):
        for est in estimates:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=[0, 1])

            _, _, rho1, la1, psi1, T1 = ddf.loc[(0, 'value'), :]
            _, _, rho2, la2, psi2, T = ddf.loc[(1, 'value'), :]
            df.loc[f'{i}.{est_label}',
            ['lambda1', 'psi1', 'p1', 'lambda2', 'psi2', 'p2', 'T1', 'type']] \
                = [la1, psi1, rho1, la2, psi2, rho2, T1, est_label]
            if (0, 'CI_min') in ddf.index:
                _, _, rho1, la1, psi1, T1 = ddf.loc[(0, 'CI_min'), :]
                _, _, rho2, la2, psi2, T = ddf.loc[(1, 'CI_min'), :]
                df.loc[f'{i}.{est_label}',
                ['lambda1_min', 'psi1_min', 'p1_min', 'lambda2_min', 'psi2_min', 'p2_min', 'T1_min', 'type']] \
                    = [la1, psi1, rho1, la2, psi2, rho2, T1, est_label]
                _, _, rho1, la1, psi1, T1 = ddf.loc[(0, 'CI_max'), :]
                _, _, rho2, la2, psi2, T = ddf.loc[(1, 'CI_max'), :]
                df.loc[f'{i}.{est_label}',
                ['lambda1_max', 'psi1_max', 'p1_max', 'lambda2_max', 'psi2_max', 'p2_max', 'T1_max', 'type']] \
                    = [la1, psi1, rho1, la2, psi2, rho2, T1, est_label]

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.to_csv(params.tab, sep='\t')
