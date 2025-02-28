import re

import numpy as np
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize likelihood ratios.")
    parser.add_argument('--ct', nargs='+', type=str, help="BD-CT model likelihoods")
    parser.add_argument('--no_ct', nargs='+', type=str, help="BD model likelihoods")
    parser.add_argument('--tab', type=str, help="summary table")
    params = parser.parse_args()

    df = pd.DataFrame(columns=['lk', 'lk_CT'])

    for log in params.no_pn:
        i = int(re.findall(r'[0-9]+', log)[-1])
        with open(log, 'r') as f:
            val = float(f.readline().strip('\n'))
        df.loc[i, 'lk'] = val

    for log in params.pn:
        i = int(re.findall(r'[0-9]+', log)[-1])
        with open(log, 'r') as f:
            val = float(f.readline().strip('\n'))
        df.loc[i, 'lk_CT'] = val

    df['lk_ratio'] = -2 * (df['lk'] - df['lk_CT'])
    df['above_threshold'] = df['lk_ratio'] > 5.99

    df.sort_index(inplace=True)
    mean_lk_r = df['lk_ratio'].mean()
    median_lk_r = np.median(df['lk_ratio'])

    df.loc['total', 'above_threshold'] = sum(df['above_threshold'])
    df.loc['mean', 'lk_ratio'] = mean_lk_r
    df.loc['median', 'lk_ratio'] = median_lk_r

    df.to_csv(params.tab, sep='\t', index=True, index_label='tree')

