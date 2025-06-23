import re
import pandas as pd
import logging  # Import logging for consistent output

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize BD-Skyline tests.")
    parser.add_argument('--logs', nargs='+', type=str, help="BD-Skyline test results log files")
    parser.add_argument('--tab', type=str, help="Output summary table file (TSV format)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for debugging parsing issues")
    params = parser.parse_args()

    # Set up basic logging for this script
    log_level = logging.INFO
    if params.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    df = pd.DataFrame(
        columns=['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence', 'evidence_binary', 'best_internal_ks',
                 'best_external_ks', 'num_tips', 'result'])

    for log in params.logs:
        logging.debug(f"Processing log file: {log}")
        try:
            with open(log, 'r') as f:
                content = f.read()

            # Extract tree ID from filename
            # This regex assumes the tree ID is the last sequence of digits in the filename.
            match_id = re.findall(r'[0-9]+', log)
            if not match_id:
                logging.warning(f"Could not extract tree ID from filename: {log}. Skipping.")
                continue
            i = int(match_id[-1])

            # Determine model type from path or log content
            model = 'UNKNOWN'
            # Initialize has_skyline based on default expectation for the model,
            # this will be overridden if 'Evidence of skyline model:' is found.
            if 'bdsky_trees' in log or 'final_tree' in log:
                model = 'BDSKY'
            elif 'bdct_trees' in log or 'BDCT0' in log:
                model = 'BDCT'
            data_type = 'tree'  # Default data type

            # Initialize all parsed variables to default values
            evidence_text = "No"
            evidence_binary = 0
            num_tests = 0
            best_internal_ks = 0.0
            best_external_ks = 0.0
            num_tips = 0  # Default to 0, will be updated if parsed

            # Extract test results from log content
            for line in content.split('\n'):
                line_stripped = line.strip()
                logging.debug(f"Reading line: {line_stripped}")

                if 'Evidence of skyline model:' in line_stripped:
                    try:
                        evidence_text = line_stripped.split(':')[1].strip()
                        evidence_binary = 1 if evidence_text.lower() == 'yes' else 0
                        logging.debug(f"Parsed evidence: {evidence_text}, binary: {evidence_binary}")
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Failed to parse evidence from line: '{line_stripped}' in {log}. Error: {e}")
                elif 'Number of interval pairs tested:' in line_stripped:
                    try:
                        # Improved regex to be more flexible with spaces
                        match = re.search(r'Number of interval pairs tested:\s*(\d+)', line_stripped)
                        if match:
                            num_tests = int(match.group(1))
                            logging.debug(f"Parsed num_tests: {num_tests}")
                        else:
                            logging.warning(
                                f"Could not find num_tests pattern in line: '{line_stripped}' in {log}. Setting to 0.")
                    except ValueError as e:
                        logging.warning(
                            f"Could not convert parsed number of tests to int from '{line_stripped}' in {log}. Setting to 0. Error: {e}")
                elif 'Best internal KS statistic:' in line_stripped:
                    try:
                        # Improved regex to capture float value
                        match = re.search(r'Best internal KS statistic:\s*([\d.]+)', line_stripped)
                        if match:
                            best_internal_ks = float(match.group(1))
                            logging.debug(f"Parsed best_internal_ks: {best_internal_ks}")
                        else:
                            logging.warning(
                                f"Could not find internal KS statistic pattern in line: '{line_stripped}' in {log}. Setting to 0.0.")
                    except ValueError as e:
                        logging.warning(
                            f"Could not convert parsed internal KS statistic to float from '{line_stripped}' in {log}. Setting to 0.0. Error: {e}")
                elif 'Best external KS statistic:' in line_stripped:
                    try:
                        # Improved regex to capture float value
                        match = re.search(r'Best external KS statistic:\s*([\d.]+)', line_stripped)
                        if match:
                            best_external_ks = float(match.group(1))
                            logging.debug(f"Parsed best_external_ks: {best_external_ks}")
                        else:
                            logging.warning(
                                f"Could not find external KS statistic pattern in line: '{line_stripped}' in {log}. Setting to 0.0.")
                    except ValueError as e:
                        logging.warning(
                            f"Could not convert parsed external KS statistic to float from '{line_stripped}' in {log}. Setting to 0.0. Error: {e}")
                elif 'Total tips in tree:' in line_stripped:
                    match = re.search(r'Total tips in tree:\s*(\d+)', line_stripped)
                    if match:
                        try:
                            num_tips = int(match.group(1))
                            logging.debug(f"Parsed num_tips: {num_tips}")
                        except ValueError as e:
                            logging.warning(
                                f"Could not convert parsed total tips '{match.group(1)}' to int in {log}. Setting to 0. Error: {e}")
                    else:
                        logging.warning(
                            f"Could not find total tips pattern in line: '{line_stripped}' in {log}. Setting to 0.")

            # Determine result classification
            # This logic still holds: if the BD-Skyline model is expected but not found, it's FN, etc.
            if model == 'BDSKY':  # Trees from BD-Skyline simulations (should have skyline evidence)
                result = 'TP' if evidence_binary == 1 else 'FN'
            elif model == 'BDCT':  # Trees from simple Birth-Death simulations (should NOT have skyline evidence)
                result = 'TN' if evidence_binary == 0 else 'FP'
            else:  # For UNKNOWN models or cases where detection expectation is unclear
                result = 'UNDEF'  # Undefined, requires manual check

            # Add data to DataFrame
            df.loc[f'{model}.{data_type}.{i}',
            ['model', 'tree/forest', 'id', 'num_tests', 'BDSKY_evidence', 'evidence_binary',
             'best_internal_ks', 'best_external_ks', 'num_tips', 'result']] = [
                model, data_type, i, num_tests, evidence_text, evidence_binary,
                best_internal_ks, best_external_ks, num_tips, result
            ]

        except Exception as e:
            logging.error(f"Top-level error processing {log}: {e}")
            continue

    # --- Global statistics ---
    TP = len(df[df['result'] == 'TP'])
    TN = len(df[df['result'] == 'TN'])
    FP = len(df[df['result'] == 'FP'])
    FN = len(df[df['result'] == 'FN'])

    print(f'Global:')
    # Ensure denominator is not zero to avoid division by zero errors
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
    print(f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}, sensitivity={sensitivity:.4f}, specificity={specificity:.4f}')
    print('==============\n')

    # --- Statistics by model ---
    # Convert columns to numeric, coercing errors to NaN and then filling with 0
    df['evidence_binary'] = pd.to_numeric(df['evidence_binary'], errors='coerce').fillna(0)
    df['num_tests'] = pd.to_numeric(df['num_tests'], errors='coerce').fillna(0)
    df['best_internal_ks'] = pd.to_numeric(df['best_internal_ks'], errors='coerce').fillna(0.0)
    df['best_external_ks'] = pd.to_numeric(df['best_external_ks'], errors='coerce').fillna(0.0)
    df['num_tips'] = pd.to_numeric(df['num_tips'], errors='coerce').fillna(0)

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
            percentage_detected = 100 * num_detected / len(ddf) if len(ddf) > 0 else 0.0

            TP_model = len(ddf[ddf['result'] == 'TP'])
            TN_model = len(ddf[ddf['result'] == 'TN'])
            FP_model = len(ddf[ddf['result'] == 'FP'])
            FN_model = len(ddf[ddf['result'] == 'FN'])

            tips_min, tips_mean, tips_max = ddf['num_tips'].min(), ddf['num_tips'].mean(), ddf['num_tips'].max()

            print(f'{model} on {data_type}s:')

            sensitivity_model = TP_model / (TP_model + FN_model) if (TP_model + FN_model) > 0 else float('nan')
            specificity_model = TN_model / (TN_model + FP_model) if (TN_model + FP_model) > 0 else float('nan')
            print(
                f'\tTP={TP_model}, TN={TN_model}, FP={FP_model}, FN={FN_model}, sensitivity={sensitivity_model:.4f}, specificity={specificity_model:.4f}')

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
                num_tests_list = '\t'.join(f'{int(float(_))}' for _ in ddff['num_tests'].astype(str).to_list())
                print(f'\tnum tests\t{num_tests_list}')
                evidence_list = '\t'.join(f'{_}' for _ in ddff['BDSKY_evidence'].astype(str).to_list())
                print(f'\tevidence\t{evidence_list}')
            else:
                print('\tNo FN/FP entries for this model/data type.')

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

    # Save the dataframe to the specified tab file
    if params.tab:
        try:
            df.to_csv(params.tab, sep='\t', index=False)
            logging.info(f"Summary table saved to {params.tab}")
        except Exception as e:
            logging.error(f"Error saving summary table to {params.tab}: {e}")
    else:
        logging.warning("No output table file specified. Table will not be saved.")
