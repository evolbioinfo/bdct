import os.path
import re

import pandas as pd
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--estimates_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdmult', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdssmult', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--real', nargs='*', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    df = pd.DataFrame(columns=['type',
                               'R',
                               'd',
                               'p',
                               'r',
                               'f_S',
                               'X_S'])

    for real in params.real:
        i = int(re.findall(r'[0-9]+', real)[-1])
        ddf = pd.read_csv(real)
        # R,d,rho,r,f_S,X_S,tips,R_observed,d_observed
        R0, it, p, r, f_S, X_S, tips, R_obs, d_obs \
            = ddf.loc[next(iter(ddf.index)), :]
        df.loc[f'{i}.real', ['type', 'R', 'd', 'p', 'r', 'f_S', 'X_S']] = ['real', R0, it, p, r, f_S, X_S]

    if params.estimates_bdmult:
        for est in params.estimates_bdmult:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bdmult'
            # ,R0,infectious time,sampling probability,transmission rate,removal rate,avg number of recipients
            R, d, rho, la, psi, r = ddf.loc['value', :]
            df.loc[f'{i}.{est_label}', ['R', 'd', 'p', 'r', 'f_S', 'X_S', 'type']] = [R, d, rho, r, 0, 1, est_label]
            # if 'CI_min' in ddf.index:
            #     R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_min', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_min', 'psi_min', 'p_min', 'd_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]
            #     R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'd_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]

    if params.estimates_bd:
        for est in params.estimates_bd:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bd'
            # ,R0,infectious time,sampling probability,transmission rate,removal rate
            R, d, rho, la, psi = ddf.loc['value', :]
            df.loc[f'{i}.{est_label}',['R', 'd', 'p', 'r', 'f_S', 'X_S', 'type']] = [R, d, rho, 1, 0, 1, est_label]
            # if 'CI_min' in ddf.index:
            #     R0, rt, rho, la, psi = ddf.loc['CI_min', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_min', 'psi_min', 'p_min', 'd_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]
            #     R0, rt, rho, la, psi = ddf.loc['CI_max', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'd_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]



    if params.estimates_bdssmult:
        for est in params.estimates_bdssmult:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bdssmult'
            # ,R0,infectious time,sampling probability,transmission rate,removal rate,N fraction,S fraction,avg number of recipients N,avg number of recipients S
            R, d, rho, la, psi, f_N, f_S, r_N, r_S = ddf.loc['value', :]
            X_S = r_S / r_N
            r = f_N * r_N + f_S * r_S
            df.loc[f'{i}.{est_label}', ['R', 'd', 'p', 'r', 'f_S', 'X_S', 'type']] = [R, d, rho, r, 0, 1, est_label]
            # if 'CI_min' in ddf.index:
            #     R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_min', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_min', 'psi_min', 'p_min', 'd_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]
            #     R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'd_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]


    df['lambda'] = df['R'] / df['d']  / df['r']
    df['psi'] = 1 / df['d']

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.to_csv(params.tab, sep='\t')
