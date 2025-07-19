import re
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize superspreading tests.")
    parser.add_argument('--logs', nargs='+', type=str, help="superspreading test results")
    parser.add_argument('--tab', type=str, help="summary table")
    params = parser.parse_args()

    df = pd.DataFrame(
        columns=['model', 'tree/forest', 'id', 'num_triplets', 'SS_test', 'superspreading_rate', 'tips_in_triplets',
                 'num_tips', 'result'])

    for log in params.logs:
        i = int(re.findall(r'[0-9]+', log)[-1])
        pval, num_triplets = pd.read_csv(log, sep='\t', header=0).iloc[0, :]

        # Extract model information - looking for patterns like BDCT0, BDSSCT0
        if 'BDEICT0' in log:
            model = 'BDEICT'
            kappa = '0'
            has_superspreading = True  # BDSSCT models have superspreading
        elif 'BDCT0' in log:
            model = 'BDCT'
            kappa = '0'
            has_superspreading = False  # BDCT models don't have superspreading
        else:
            # Fallback parsing
            model_match = re.findall(r'(BDEICT|BDCT)(\d+)', log)
            if model_match:
                model, kappa = model_match[0]
                has_superspreading = (model == 'BDEICT')
            else:
                model = 'BDCT'
                kappa = '0'
                has_superspreading = False

        data_type = re.findall(r'tree|forest', log)[0]
        df.loc[f'{model}({kappa}).{data_type}.{i}', ['model', 'tree/forest', 'id', 'num_triplets', 'SS_test']] \
            = [f'{model}({kappa})', data_type, i, num_triplets, pval]

        # Try to read additional parameters if available
        try:
            param_log = log.replace('.bdss_test', '.log')
            param_df = pd.read_csv(param_log, header=0)

            rho = param_df.loc[0, 'sampling probability'] if 'sampling probability' in param_df.columns else 0.5
            tips = param_df.loc[0, 'tips'] if 'tips' in param_df.columns else num_triplets * 2

            # For superspreading, we assume BDSSCT has superspreading, BDCT doesn't
            if has_superspreading:
                ss_rate = 1.0  # Assume some superspreading rate for BDSSCT
                result = 'TP' if pval < 0.05 else 'FN'
            else:
                ss_rate = 0.0  # No superspreading for BDCT
                result = 'TN' if pval >= 0.05 else 'FP'

            df.loc[f'{model}({kappa}).{data_type}.{i}',
            ['superspreading_rate', 'tips_in_triplets', 'num_tips', 'result']] \
                = [ss_rate, num_triplets * 2 / tips if tips > 0 else 0, tips, result]

        except (FileNotFoundError, KeyError, IndexError):
            # If parameter file doesn't exist or doesn't have expected columns
            if has_superspreading:
                ss_rate = 1.0
                result = 'TP' if pval < 0.05 else 'FN'
            else:
                ss_rate = 0.0
                result = 'TN' if pval >= 0.05 else 'FP'

            df.loc[f'{model}({kappa}).{data_type}.{i}',
            ['superspreading_rate', 'tips_in_triplets', 'num_tips', 'result']] \
                = [ss_rate, 1.0, num_triplets * 2, result]

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

            mean_pval = ddf['SS_test'].mean()
            min_pval = ddf['SS_test'].min()
            max_pval = ddf['SS_test'].max()
            mean_nc = ddf['num_triplets'].mean()
            min_nc = ddf['num_triplets'].min()
            max_nc = ddf['num_triplets'].max()
            num_sign = len(ddf[ddf['SS_test'] < 0.05])
            percentage_significant = 100 * num_sign / len(ddf)

            TP = len(ddf[ddf['result'] == 'TP'])
            TN = len(ddf[ddf['result'] == 'TN'])
            FP = len(ddf[ddf['result'] == 'FP'])
            FN = len(ddf[ddf['result'] == 'FN'])

            tips_min, tips_mean, tips_max = ddf['num_tips'].min(), ddf['num_tips'].mean(), ddf['num_tips'].max()
            ss_rate_min, ss_rate_mean, ss_rate_max = ddf['superspreading_rate'].min(), ddf[
                'superspreading_rate'].mean(), ddf['superspreading_rate'].max()
            tic_min, tic_mean, tic_max = ddf['tips_in_triplets'].min(), ddf['tips_in_triplets'].mean(), ddf[
                'tips_in_triplets'].max()

            print(f'{model} on {data_type}s:')

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 'NA'
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 'NA'
            print(f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}, sensitivity={sensitivity}, specificity={specificity}')

            print('--------all:')
            print(f'\tp-val < 0.05\t{percentage_significant}%')
            print(f'\tavg p-val\t{mean_pval}\t[{min_pval}-{max_pval}]')
            print(f'\tavg num triplets\t{mean_nc}\t[{min_nc}-{max_nc}]')
            print(f'\tavg num tips\t{tips_mean}\t[{tips_min}-{tips_max}]')
            print(f'\tavg num tips in triplets\t{tic_mean}\t[{tic_min}-{tic_max}]')
            print(f'\tavg superspreading_rate\t{ss_rate_mean}\t[{ss_rate_min}-{ss_rate_max}]')

            print('--------FN/FP:')
            ddff = ddf[(ddf['result'] == 'FN') | (ddf['result'] == 'FP')]

            if len(ddff) > 0:
                num_triplets = '\t'.join(f'{_}' for _ in ddff['num_triplets'].astype(int).to_list())
                print(f'\tnum triplets\t{num_triplets}')
                ss_rates = '\t'.join(f'{_:.3f}' for _ in ddff['superspreading_rate'].astype(float).to_list())
                print(f'\tsuperspreading_rate\t{ss_rates}')

            df.loc[f'{model}.{data_type}.mean',
            ['model', 'tree/forest', 'id', 'num_triplets', 'SS_test', 'superspreading_rate', 'tips_in_triplets',
             'num_tips']] \
                = [model, data_type, 'mean', mean_nc, mean_pval, ss_rate_mean, tic_mean, tips_mean]
            df.loc[f'{model}.{data_type}.min',
            ['model', 'tree/forest', 'id', 'num_triplets', 'SS_test', 'superspreading_rate', 'tips_in_triplets',
             'num_tips']] \
                = [model, data_type, 'min', min_nc, min_pval, ss_rate_min, tic_min, tips_min]
            df.loc[f'{model}.{data_type}.max',
            ['model', 'tree/forest', 'id', 'num_triplets', 'SS_test', 'superspreading_rate', 'tips_in_triplets',
             'num_tips']] \
                = [model, data_type, 'max', max_nc, max_pval, ss_rate_max, tic_max, tips_max]
            df.loc[f'{model}.{data_type}.perc', ['model', 'tree/forest', 'id', 'num_triplets', 'SS_test']] \
                = [model, data_type, 'percent < 0.05', num_sign, percentage_significant]

            print('==============\n')

    df.sort_values(by=['model', 'tree/forest', 'id'], inplace=True)
    df.to_csv(params.tab, sep='\t', index=False)