import re
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize BDEI tests.")
    parser.add_argument('--logs', nargs='+', type=str, help="BDEI test results")
    parser.add_argument('--tab', type=str, help="summary table")
    params = parser.parse_args()

    df = pd.DataFrame(columns=['model', 'tree/forest', 'id', 'num_cherries', 'BDEI_test', 'infectious_vs_notified_time',
                               'upsilon_rho', 'upsilon', 'rho', 'tips_in_cherries', 'num_tips', 'result'])

    for log in params.logs:
        i = int(re.findall(r'[0-9]+', log)[-1])
        pval, num_cherries = pd.read_csv(log, sep='\t', header=0).iloc[0, :]

        # Extract model info for BDEI test
        if 'BDCT0' in log:
            model, kappa = 'BDCT', '0'
        elif 'BDEICT0' in log:
            model, kappa = 'BDEICT', '0'
        else:
            print(f"Warning: Could not extract model from {log}")
            continue

        data_type = 'tree'  # Assuming tree data based on your structure
        df.loc[f'{model}({kappa}).{data_type}.{i}', ['model', 'tree/forest', 'id', 'num_cherries', 'BDEI_test']] \
            = [f'{model}({kappa})', data_type, i, num_cherries, pval]

        # Try to read additional parameters if available
        try:
            param_log = log.replace('.bdei_test', '.log')
            if re.search(r'BDEICT0', log):  # Has epidemiological intervention
                param_df = pd.read_csv(param_log, header=0)
                rho = param_df.loc[0, 'sampling probability']
                upsilon = param_df.loc[0, 'notification probability']
                tips = param_df.loc[0, 'tips']
                psit = param_df.loc[0, 'infectious time']
                phit = param_df.loc[0, 'removal time after notification']
                df.loc[f'{model}({kappa}).{data_type}.{i}', ['infectious_vs_notified_time', 'upsilon_rho', 'upsilon',
                                                             'rho', 'tips_in_cherries', 'num_tips', 'result']] \
                    = [psit / phit, upsilon * rho, upsilon, rho, num_cherries * 2 / tips, tips,
                       'TP' if pval < 0.05 else 'FN']
            else:  # BDCT0 - no epidemiological intervention
                param_df = pd.read_csv(param_log, header=0)
                rho = param_df.loc[0, 'sampling probability']
                tips = param_df.loc[0, 'tips']
                df.loc[f'{model}({kappa}).{data_type}.{i}', ['infectious_vs_notified_time', 'upsilon_rho', 'upsilon',
                                                             'rho', 'tips_in_cherries', 'num_tips', 'result']] \
                    = [1, 0, 0, rho, num_cherries * 2 / tips, tips, 'TN' if pval >= 0.05 else 'FP']
        except:
            # If parameter files don't exist, set basic values
            df.loc[f'{model}({kappa}).{data_type}.{i}', ['infectious_vs_notified_time', 'upsilon_rho', 'upsilon', 'rho',
                                                         'tips_in_cherries', 'num_tips', 'result']] \
                = [1, 0, 0, 1, 0, 0, 'TP' if 'BDEICT' in model and pval < 0.05 else (
                'TN' if 'BDCT' in model and pval >= 0.05 else ('FN' if 'BDEICT' in model else 'FP'))]

    TP = len(df[df['result'] == 'TP'])
    TN = len(df[df['result'] == 'TN'])
    FP = len(df[df['result'] == 'FP'])
    FN = len(df[df['result'] == 'FN'])

    print(f'Global:')
    print(
        f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}, sensitivity={TP / (TP + FN) if (TP + FN) > 0 else "NA"}, specificity={TN / (TN + FP) if (TN + FP) > 0 else "NA"}')
    print('==============\n')

    for model in df['model'].unique():
        for data_type in df['tree/forest'].unique():
            ddf = df[(df['model'] == model) & (df['tree/forest'] == data_type)]
            if len(ddf) == 0:
                continue

            mean_pval = ddf['BDEI_test'].mean()
            min_pval = ddf['BDEI_test'].min()
            max_pval = ddf['BDEI_test'].max()
            mean_nc = ddf['num_cherries'].mean()
            min_nc = ddf['num_cherries'].min()
            max_nc = ddf['num_cherries'].max()
            num_sign = len(ddf[ddf['BDEI_test'] < 0.05])
            percentage_significant = 100 * num_sign / len(ddf)

            TP = len(ddf[ddf['result'] == 'TP'])
            TN = len(ddf[ddf['result'] == 'TN'])
            FP = len(ddf[ddf['result'] == 'FP'])
            FN = len(ddf[ddf['result'] == 'FN'])

            tips_min, tips_mean, tips_max = ddf['num_tips'].min(), ddf['num_tips'].mean(), ddf['num_tips'].max()
            phi_psi_min, phi_psi_mean, phi_psi_max = ddf['infectious_vs_notified_time'].min(), ddf[
                'infectious_vs_notified_time'].mean(), ddf['infectious_vs_notified_time'].max()
            ups_min, ups_mean, ups_max = ddf['upsilon'].min(), ddf['upsilon'].mean(), ddf['upsilon'].max()
            rho_min, rho_mean, rho_max = ddf['rho'].min(), ddf['rho'].mean(), ddf['rho'].max()
            ups_rho_min, ups_rho_mean, ups_rho_max = ddf['upsilon_rho'].min(), ddf['upsilon_rho'].mean(), ddf[
                'upsilon_rho'].max()
            tic_min, tic_mean, tic_max = ddf['tips_in_cherries'].min(), ddf['tips_in_cherries'].mean(), ddf[
                'tips_in_cherries'].max()

            print(f'{model} on {data_type}s:')

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 'NA'
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 'NA'
            print(f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}, sensitivity={sensitivity}, specificity={specificity}')

            print('--------all:')
            print(f'\tp-val < 0.05\t{percentage_significant}%')
            print(f'\tavg p-val\t{mean_pval}\t[{min_pval}-{max_pval}]')
            print(f'\tavg num cherries\t{mean_nc}\t[{min_nc}-{max_nc}]')
            print(f'\tavg num tips\t{tips_mean}\t[{tips_min}-{tips_max}]')
            print(f'\tavg num tips in cherries\t{tic_mean}\t[{tic_min}-{tic_max}]')
            print(f'\tavg infectious_vs_notified_time\t{phi_psi_mean}\t[{phi_psi_min}-{phi_psi_max}]')
            print(f'\tavg ups\t{ups_mean}\t[{ups_min}-{ups_max}]')
            print(f'\tavg rho\t{rho_mean}\t[{rho_min}-{rho_max}]')
            print(f'\tavg rho*ups\t{ups_rho_mean}\t[{ups_rho_min}-{ups_rho_max}]')

            print('--------FN/FP:')
            ddff = ddf[(ddf['result'] == 'FN') | (ddf['result'] == 'FP')]

            if len(ddff) > 0:
                num_cherries = '\t'.join(f'{_}' for _ in ddff['num_cherries'].astype(int).to_list())
                print(f'\tnum cherries\t{num_cherries}')
                inf_vs_notified_times = '\t'.join(
                    f'{_:.1f}' for _ in ddff['infectious_vs_notified_time'].astype(float).to_list())
                print(f'\tinfectious_vs_notified_time\t{inf_vs_notified_times}')
                rho = '\t'.join(f'{_:.3f}' for _ in ddff['rho'].astype(float).to_list())
                print(f'\trho\t{rho}')
                rho_ups = '\t'.join(f'{_:.3f}' for _ in ddff['upsilon_rho'].astype(float).to_list())
                print(f'\trho*ups\t{rho_ups}')

            df.loc[f'{model}.{data_type}.mean',
            ['model', 'tree/forest', 'id', 'num_cherries', 'BDEI_test', 'infectious_vs_notified_time', 'upsilon_rho',
             'upsilon', 'rho', 'tips_in_cherries', 'num_tips']] \
                = [model, data_type, 'mean', mean_nc, mean_pval, phi_psi_mean, ups_rho_mean, ups_mean, rho_mean,
                   tic_mean, tips_mean]
            df.loc[f'{model}.{data_type}.min',
            ['model', 'tree/forest', 'id', 'num_cherries', 'BDEI_test', 'infectious_vs_notified_time', 'upsilon_rho',
             'upsilon', 'rho', 'tips_in_cherries', 'num_tips']] \
                = [model, data_type, 'min', min_nc, min_pval, phi_psi_min, ups_rho_min, ups_min, rho_min, tic_min,
                   tips_min]
            df.loc[f'{model}.{data_type}.max',
            ['model', 'tree/forest', 'id', 'num_cherries', 'BDEI_test', 'infectious_vs_notified_time', 'upsilon_rho',
             'upsilon', 'rho', 'tips_in_cherries', 'num_tips']] \
                = [model, data_type, 'max', max_nc, max_pval, phi_psi_max, ups_rho_max, ups_max, rho_max, tic_max,
                   tips_max]
            df.loc[f'{model}.{data_type}.perc', ['model', 'tree/forest', 'id', 'num_cherries', 'BDEI_test']] \
                = [model, data_type, 'percent < 0.05', num_sign, percentage_significant]

            print('==============\n')

    df.sort_values(by=['model', 'tree/forest', 'id'], inplace=True)
    df.to_csv(params.tab, sep='\t', index=False)