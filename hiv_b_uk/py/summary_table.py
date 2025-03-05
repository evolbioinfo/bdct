import os
import re

import pandas as pd
from ete3 import Tree


def read_forest(tree_path):
    with open(tree_path, 'r') as f:
        return [Tree(_ + ';') for _ in f.read().replace('\n', '').strip().split(';')[:-1]]


def latexify_values(ci=True):
    df['p'] = ' $' + df['p'].apply(str) + '$'
    if ci:
        df['infectious_time'] = ' $' + df['infectious_time'].apply(str) + '\;[' + df['infectious_time_min'].apply(
            str) + '-' + df['infectious_time_max'].apply(str) + ']$'
        df['partner_removal_time'] = ' $' + df['partner_removal_time'].apply(str) + '\;[' + df[
            'partner_removal_time_min'].apply(
            str) + '-' + df['partner_removal_time_max'].apply(str) + ']$'
        df['R'] = ' $' + df['R'].apply(str) + '\;[' + df['R_min'].apply(str) + '-' + df['R_max'].apply(str) + ']$'
        df['lambda'] = ' $' + df['lambda'].apply(str) + '\;[' + df['lambda_min'].apply(str) + '-' + df['lambda_max'].apply(
            str) + ']$'
        df['psi'] = ' $' + df['psi'].apply(str) + '\;[' + df['psi_min'].apply(str) + '-' + df['psi_max'].apply(
            str) + ']$ '
        df['phi'] = ' $' + df['phi'].apply(str) + '\;[' + df['phi_min'].apply(str) + '-' + df['phi_max'].apply(
            str) + ']$ '
        df['upsilon'] = ' $' + df['upsilon'].apply(str) + '\;[' + df['upsilon_min'].apply(str) + '-' + df['upsilon_max'].apply(
            str) + ']$ \\\\'
    else:
        df['infectious_time'] = ' $' + df['infectious_time'].apply(str) + '$'
        df['partner_removal_time'] = ' $' + df['partner_removal_time'].apply(str) + '$  '
        df['R'] = ' $' + df['R'].apply(str) + '$'
        df['lambda'] = ' $' + df['lambda'].apply(str) + '$'
        df['psi'] = ' $' + df['psi'].apply(str) + '$ '
        df['phi'] = ' $' + df['phi'].apply(str) + '$ '
        df['upsilon'] = ' $' + df['upsilon'].apply(str) + '$   \\\\'
    df['repetition'] = ' $' + df['repetition'].apply(str) + '$ '
    df['observed_trees'] = ' $' + df['observed_trees'].apply(str) + '$ '
    df['sampled_tips'] = ' $' + df['sampled_tips'].apply(str) + '$ '


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combines the estimates from different forests/settings into one table.")
    parser.add_argument('--forests', nargs='+', type=str, help="forests")
    parser.add_argument('--estimates', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    df = pd.DataFrame(columns=['repetition', 'sampled_tips', 'observed_trees', 'hidden_trees',
                               'lambda', 'lambda_min', 'lambda_max',
                               'psi', 'psi_min', 'psi_max',
                               'phi', 'phi_min', 'phi_max',
                               'R', 'R_min', 'R_max',
                               'infectious_time', 'infectious_time_min', 'infectious_time_max',
                               'partner_removal_time', 'partner_removal_time_min', 'partner_removal_time_max',
                               'p',
                               'upsilon', 'upsilon_min', 'upsilon_max'])

    i2stats = {}
    for nwk in params.forests:
        forest = read_forest(nwk)
        rep = int(re.findall(r'\.(\d+)\.nwk', os.path.basename(nwk))[0]) + 1
        i2stats[rep] = (len(forest), sum(len(_) for _ in forest))

    for file in params.estimates:
        basename = os.path.basename(file)

        rep = int(re.findall(r'\.(\d+)\.[a-z]+[=]', basename)[0]) + 1
        o_trees, tips = i2stats[rep]

        estimates = pd.read_csv(file, index_col=0)
        est_label = '{i}.p={p}'.format(i=rep, p=estimates.loc['value', 'sampling probability'])

        df.loc[est_label, ['lambda', 'psi', 'phi', 'R', 'infectious_time', 'partner_removal_time', 'p', 'upsilon']] \
            = estimates.loc['value', ['transmission rate', 'removal rate', 'notified sampling rate',
                                      'R0', 'infectious time', 'removal time after notification',
                                      'sampling probability', 'contact tracing probability']].tolist()

        df.loc[est_label, ['lambda_min', 'psi_min', 'phi_min', 'R_min', 'infectious_time_min',
                           'partner_removal_time_min', 'upsilon_min']] \
            = estimates.loc['CI_min', ['transmission rate', 'removal rate', 'notified sampling rate',
                                       'R0', 'infectious time', 'removal time after notification',
                                       'contact tracing probability']].tolist()

        df.loc[est_label, ['lambda_max', 'psi_max', 'phi_max', 'R_max', 'infectious_time_max',
                           'partner_removal_time_max', 'upsilon_max']] \
            = estimates.loc['CI_max', ['transmission rate', 'removal rate', 'notified sampling rate',
                                       'R0', 'infectious time', 'removal time after notification',
                                       'contact tracing probability']].tolist()
        df.loc[est_label, ['repetition', 'sampled_tips', 'observed_trees']] = [rep, tips, o_trees]

    df.sort_values(by=['repetition', 'p'], inplace=True)

    for col in ['lambda', 'lambda_min', 'lambda_max',
                'psi', 'psi_min', 'psi_max',
                'phi', 'phi_min', 'phi_max',
                'R', 'R_min', 'R_max',
                'infectious_time', 'infectious_time_min', 'infectious_time_max',
                'p']:
        df[col] = df[col].apply(lambda _: '{:.2f}'.format(_))
    for col in ['upsilon', 'upsilon_min', 'upsilon_max']:
        df[col] = df[col].apply(lambda _: '{:.3f}'.format(_))
    for col in ['partner_removal_time', 'partner_removal_time_min', 'partner_removal_time_max']:
        df[col] = df[col].apply(lambda _: '{:.0f}'.format(365 * _))
    df[['repetition', 'sampled_tips', 'observed_trees',
        'p',
        'R', 'R_min', 'R_max',
        'infectious_time', 'infectious_time_min', 'infectious_time_max',
        'partner_removal_time', 'partner_removal_time_min', 'partner_removal_time_max',
        'upsilon', 'upsilon_min', 'upsilon_max',
        'lambda', 'lambda_min', 'lambda_max',
        'psi', 'psi_min', 'psi_max',
        'phi', 'phi_min', 'phi_max',
        ]].to_csv(params.tab, sep='\t', index=False)

    latexify_values(ci=True)

    df = df[['repetition', 'sampled_tips', 'observed_trees', 'p',
             'R', 'infectious_time', 'partner_removal_time', 'upsilon']]
    df.columns = ['repetition', 'sampled tips', 'observed trees', 'p',
                  'R', 'infectious time (years)', 'partner removal time (days)', 'upsilon']
    df.to_csv(params.tab + '.latex', sep='&', index=False)
