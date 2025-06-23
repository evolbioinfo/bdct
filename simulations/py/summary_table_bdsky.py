import re
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize BD-Skyline tests.")
    parser.add_argument('--logs', nargs='+', type=str, help="BD-Skyline test results")
    parser.add_argument('--tab', type=str, help="summary table")
    params = parser.parse_args()

    df = pd.DataFrame(
        columns=['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence', 'evidence_binary', 'best_internal_ks',
                 'best_external_ks', 'num_tips', 'result'])

    for log in params.logs:
        try:
            with open(log, 'r') as f:
                content = f.read()

            # Extract tree ID from filename
            i = int(re.findall(r'[0-9]+', log)[-1])

            # Determine model type from path
            if 'bdsky_trees' in log or 'final_tree' in log:
                model = 'BDSKY'
                has_skyline = True  # BDSKY trees should have skyline evidence
                data_type = 'tree'
            elif 'bdct_trees' in log or 'BDCT0' in log:
                model = 'BDCT'
                has_skyline = False  # BDCT trees should not have skyline evidence
                data_type = 'tree'
            else:
                model = 'UNKNOWN'
                has_skyline = False
                data_type = 'tree'

            # Extract test results from log content
            evidence_line = [line for line in content.split('\n') if 'Evidence of skyline model:' in line]
            tests_line = [line for line in content.split('\n') if 'Number of tests performed:' in line]
            internal_ks_line = [line for line in content.split('\n') if 'Best internal KS statistic:' in line]
            external_ks_line = [line for line in content.split('\n') if 'Best external KS statistic:' in line]

            if evidence_line and tests_line:
                evidence_text = evidence_line[0].split(':')[1].strip()
                evidence_binary = 1 if evidence_text.lower() == 'yes' else 0
                num_tests = int(tests_line[0].split(':')[1].strip())

                # Extract KS statistics if available
                best_internal_ks = 0.0
                best_external_ks = 0.0
                if internal_ks_line:
                    try:
                        best_internal_ks = float(internal_ks_line[0].split(':')[1].strip())
                    except:
                        pass
                if external_ks_line:
                    try:
                        best_external_ks = float(external_ks_line[0].split(':')[1].strip())
                    except:
                        pass

                # Determine result classification
                if has_skyline:  # BDSKY trees
                    result = 'TP' if evidence_binary == 1 else 'FN'
                else:  # BDCT trees (control)
                    result = 'TN' if evidence_binary == 0 else 'FP'

                # Estimate number of tips (rough approximation from number of tests)
                estimated_tips = max(100, num_tests * 10)  # Conservative estimate

                df.loc[f'{model}.{data_type}.{i}',
                ['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence', 'evidence_binary',
                 'best_internal_ks', 'best_external_ks', 'num_tips', 'result']] = [
                    model, data_type, i, num_tests, evidence_text, evidence_binary,
                    best_internal_ks, best_external_ks, estimated_tips, result
                ]

        except Exception as e:
            print(f"Error processing {log}: {e}")
            continue

    # Global statistics
    TP = len(df[df['result'] == 'TP'])
    TN = len(df[df['result'] == 'TN'])
    FP = len(df[df['result'] == 'FP'])
    FN = len(df[df['result'] == 'FN'])

    print(f'Global:')
    print(
        f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}, sensitivity={TP / (TP + FN) if (TP + FN) > 0 else "NA"}, specificity={TN / (TN + FP) if (TN + FP) > 0 else "NA"}')
    print('==============\n')

    # Statistics by model
    for model in df['model'].unique():
        for data_type in df['tree/forest'].unique():
            ddf = df[(df['model'] == model) & (df['tree/forest'] == data_type)]
            if len(ddf) == 0:
                continue

            mean_evidence = ddf['evidence_binary'].mean()
            mean_tests = ddf['num_tests'].mean()
            min_tests = ddf['num_tests'].min()
            max_tests = ddf['num_tests'].max()
            mean_internal_ks = ddf['best_internal_ks'].mean()
            mean_external_ks = ddf['best_external_ks'].mean()

            num_detected = len(ddf[ddf['evidence_binary'] == 1])
            percentage_detected = 100 * num_detected / len(ddf)

            TP = len(ddf[ddf['result'] == 'TP'])
            TN = len(ddf[ddf['result'] == 'TN'])
            FP = len(ddf[ddf['result'] == 'FP'])
            FN = len(ddf[ddf['result'] == 'FN'])

            tips_min, tips_mean, tips_max = ddf['num_tips'].min(), ddf['num_tips'].mean(), ddf['num_tips'].max()

            print(f'{model} on {data_type}s:')

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 'NA'
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 'NA'
            print(f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}, sensitivity={sensitivity}, specificity={specificity}')

            print('--------all:')
            print(f'\tSkyline detected\t{percentage_detected:.1f}%')
            print(f'\tavg evidence rate\t{mean_evidence:.3f}')
            print(f'\tavg num tests\t{mean_tests:.1f}\t[{min_tests}-{max_tests}]')
            print(f'\tavg internal KS\t{mean_internal_ks:.4f}')
            print(f'\tavg external KS\t{mean_external_ks:.4f}')
            print(f'\tavg num tips\t{tips_mean:.0f}\t[{tips_min}-{tips_max}]')

            print('--------FN/FP:')
            ddff = ddf[(ddf['result'] == 'FN') | (ddf['result'] == 'FP')]

            if len(ddff) > 0:
                num_tests_list = '\t'.join(f'{int(_)}' for _ in ddff['num_tests'].to_list())
                print(f'\tnum tests\t{num_tests_list}')
                evidence_list = '\t'.join(f'{_}' for _ in ddff['BDSKY_evidence'].to_list())
                print(f'\tevidence\t{evidence_list}')

            # Summary statistics
            df.loc[f'{model}.{data_type}.mean',
            ['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence', 'evidence_binary',
             'best_internal_ks', 'best_external_ks', 'num_tips']] = [
                model, data_type, 'mean', mean_tests, f'{mean_evidence:.3f}', mean_evidence,
                mean_internal_ks, mean_external_ks, tips_mean
            ]

            df.loc[f'{model}.{data_type}.min',
            ['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence', 'evidence_binary',
             'best_internal_ks', 'best_external_ks', 'num_tips']] = [
                model, data_type, 'min', min_tests, f'{ddf["evidence_binary"].min():.0f}', ddf['evidence_binary'].min(),
                ddf['best_internal_ks'].min(), ddf['best_external_ks'].min(), tips_min
            ]

            df.loc[f'{model}.{data_type}.max',
            ['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence', 'evidence_binary',
             'best_internal_ks', 'best_external_ks', 'num_tips']] = [
                model, data_type, 'max', max_tests, f'{ddf["evidence_binary"].max():.0f}', ddf['evidence_binary'].max(),
                ddf['best_internal_ks'].max(), ddf['best_external_ks'].max(), tips_max
            ]

            df.loc[f'{model}.{data_type}.perc',
            ['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence']] = [
                model, data_type, 'percent detected', num_detected, f'{percentage_detected:.1f}%'
            ]

            print('==============\n')

    df.sort_values(by=['model', 'tree/forest', 'id'], inplace=True)
    df.to_csv(params.tab, sep='\t', index=False)