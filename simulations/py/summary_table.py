
import logging
import re

import pandas as pd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize estimates.")
    parser.add_argument('--estimates_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdct', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--real', nargs='*', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(columns=['data_type', 'type', 'sampled_tips',
                               'lambda', 'lambda_min', 'lambda_max',
                               'psi', 'psi_min', 'psi_max',
                               'upsilon', 'upsilon_min', 'upsilon_max',
                               'phi', 'phi_min', 'phi_max'])

    for real in params.real:
        i = int(re.findall(r'[0-9]+', real)[-1])
        tree_type = re.findall('tree|forest', real)[0]
        kappa = int(re.findall(r'[0-9]+', real)[-2])
        ddf = pd.read_csv(real)

        tips = ddf.loc[0, 'tips']
        p = ddf.loc[0, 'sampling probability']
        psi = 1 / ddf.loc[0, 'infectious time']
        la = ddf.loc[0, 'R0'] * psi
        upsilon = ddf.loc[0, 'notification probability'] if kappa > 0 else 0
        phi = 1 / ddf.loc[0, 'removal time after notification'] if kappa > 0 else psi

        df.loc[f'{i}.{tree_type}.real',
        ['lambda', 'psi', 'phi', 'p', 'upsilon', 'sampled_tips', 'type', 'data_type']] \
            = [la, psi, phi, p, upsilon, tips, 'real', tree_type]

    if params.estimates_bdct:
        for est in params.estimates_bdct:
            i = int(re.findall(r'[0-9]+', est)[-1])
            tree_type = re.findall('tree|forest', est)[0]
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bdct'
            R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['value', :]
            df.loc[f'{i}.{tree_type}.{est_label}',
            ['lambda', 'psi', 'phi', 'p', 'upsilon', 'type', 'data_type']] \
                = [la, psi, phi, rho, upsilon, est_label, tree_type]
            if 'CI_min' in ddf.index:
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_min', :]
                df.loc[f'{i}.{tree_type}.{est_label}',
                ['lambda_min', 'psi_min', 'phi_min', 'p_min', 'upsilon_min', 'type', 'data_type']] \
                    = [la, psi, phi, rho, upsilon, est_label, tree_type]
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
                df.loc[f'{i}.{tree_type}.{est_label}',
                ['lambda_max', 'psi_max', 'phi_max', 'p_max', 'upsilon_max', 'type', 'data_type']] \
                    = [la, psi, phi, rho, upsilon, est_label, tree_type]

    if params.estimates_bd:
        for est in params.estimates_bd:
            i = int(re.findall(r'[0-9]+', est)[-1])
            tree_type = re.findall('tree|forest', est)[0]
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bd'
            R0, rt, rho, la, psi = ddf.loc['value', :]
            df.loc[f'{i}.{tree_type}.{est_label}',
            ['lambda', 'psi', 'p', 'type', 'data_type']] \
                = [la, psi, rho, est_label, tree_type]
            if 'CI_min' in ddf.index:
                R0, rt, rho, la, psi = ddf.loc['CI_min', :]
                df.loc[f'{i}.{tree_type}.{est_label}',
                ['lambda_min', 'psi_min', 'p_min', 'type', 'data_type']] \
                    = [la, psi, rho, est_label, tree_type]
                R0, rt, rho, la, psi = ddf.loc['CI_max', :]
                df.loc[f'{i}.{tree_type}.{est_label}',
                ['lambda_max', 'psi_max', 'p_max', 'type', 'data_type']] \
                    = [la, psi, rho, est_label, tree_type]

    df.loc[pd.isna(df['upsilon']), 'upsilon'] = 0
    df.loc[pd.isna(df['phi']), 'phi'] = df['psi']

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.index = df.index.map(str) + '.' + df['data_type']
    df.to_csv(params.tab, sep='\t')

