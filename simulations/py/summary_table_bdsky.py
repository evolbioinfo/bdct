import re
import pandas as pd
import logging

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize BD-Skyline tests (updated for comprehensive diagnostic output).")
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
        columns=['model', 'tree/forest', 'id', 'evidence_type', 'BDSKY_evidence', 'evidence_binary',
                 'internal_u_stat', 'external_u_stat', 'internal_pval', 'external_pval',
                 'internal_T', 'external_T', 'num_tips', 'result',
                 'internal_cohens_d', 'external_cohens_d', 'internal_relative_diff', 'external_relative_diff',
                 'internal_early_mean', 'external_early_mean', 'internal_late_mean', 'external_late_mean',
                 'internal_effect_significant', 'external_effect_significant',
                 'internal_ks_statistic', 'external_ks_statistic', 'internal_ks_pval', 'external_ks_pval',
                 'internal_ks_significant', 'external_ks_significant', 'test_type',
                 'alt_internal_mw_stat', 'alt_external_mw_stat', 'alt_internal_mw_pval', 'alt_external_mw_pval',
                 'alt_internal_ks_stat', 'alt_external_ks_stat', 'alt_internal_ks_pval', 'alt_external_ks_pval'])

    for log in params.logs:
        logging.debug(f"Processing log file: {log}")
        try:
            with open(log, 'r') as f:
                content = f.read()

            # Extract tree ID from filename
            match_id = re.findall(r'[0-9]+', log)
            if not match_id:
                logging.warning(f"Could not extract tree ID from filename: {log}. Skipping.")
                continue
            i = int(match_id[-1])

            # Determine model type from path or log content
            model = 'UNKNOWN'
            if 'bdsky_trees' in log or 'final_tree' in log:
                model = 'BDSKY'
            elif 'bdct_trees' in log or 'BDCT0' in log:
                model = 'BDCT'
            data_type = 'tree'

            # Determine test type - updated for comprehensive test output
            test_type = 'comprehensive'
            if 'COMPREHENSIVE SKY TEST ANALYSIS' in content:
                test_type = 'comprehensive'
            elif 'Effect-size SKY test' in content:
                test_type = 'effect-size'
            elif 'Kolmogorov-Smirnov SKY test' in content:
                test_type = 'ks-test'
            elif 'SKY test' in content:
                test_type = 'original'

            # Initialize all variables
            evidence_binary_uncorrected = 0
            evidence_binary_bonferroni = 0
            internal_u_stat = 0.0
            external_u_stat = 0.0
            internal_pval = 1.0
            external_pval = 1.0
            internal_T = 0.0
            external_T = 0.0
            num_tips = 0

            # Effect size variables
            internal_cohens_d = 0.0
            external_cohens_d = 0.0
            internal_relative_diff = 0.0
            external_relative_diff = 0.0
            internal_early_mean = 0.0
            external_early_mean = 0.0
            internal_late_mean = 0.0
            external_late_mean = 0.0
            internal_effect_significant = False
            external_effect_significant = False

            # KS test variables
            internal_ks_statistic = 0.0
            external_ks_statistic = 0.0
            internal_ks_pval = 1.0
            external_ks_pval = 1.0
            internal_ks_significant = False
            external_ks_significant = False

            # Alternative test variables
            alt_internal_mw_stat = 0.0
            alt_external_mw_stat = 0.0
            alt_internal_mw_pval = 1.0
            alt_external_mw_pval = 1.0
            alt_internal_ks_stat = 0.0
            alt_external_ks_stat = 0.0
            alt_internal_ks_pval = 1.0
            alt_external_ks_pval = 1.0

            # Parse total tips
            tips_match = re.search(r'Total tips in tree:\s*(\d+)', content)
            if tips_match:
                num_tips = int(tips_match.group(1))

            # Parse the content for comprehensive test format
            lines = content.split('\n')
            current_section = None
            current_branch_type = None

            for line in lines:
                line_stripped = line.strip()

                # Identify sections
                if '=== ORIGINAL SKY TEST ===' in line_stripped:
                    current_section = 'original'
                elif '=== ALTERNATIVE SKY TEST ===' in line_stripped:
                    current_section = 'alternative'
                elif '=== SUMMARY ===' in line_stripped:
                    current_section = 'summary'

                # Parse branch type sections in original test
                if current_section == 'original':
                    if '--- INTERNAL BRANCHES ---' in line_stripped:
                        current_branch_type = 'internal'
                    elif '--- EXTERNAL BRANCHES ---' in line_stripped:
                        current_branch_type = 'external'

                    # Parse T values
                    if current_branch_type and 'Time for' in line_stripped and 'branches:' in line_stripped:
                        match = re.search(r'Time for \d+ \w+ branches: ([\d.]+)', line_stripped)
                        if match:
                            if current_branch_type == 'internal':
                                internal_T = float(match.group(1))
                            else:
                                external_T = float(match.group(1))

                    # Parse Mann-Whitney results
                    if 'Mann-Whitney U test:' in line_stripped:
                        u_match = re.search(r'statistic=([\d.]+), p=([\d.e-]+)', line_stripped)
                        if u_match and current_branch_type:
                            if current_branch_type == 'internal':
                                internal_u_stat = float(u_match.group(1))
                                internal_pval = float(u_match.group(2))
                            else:
                                external_u_stat = float(u_match.group(1))
                                external_pval = float(u_match.group(2))

                # Parse alternative test results
                elif current_section == 'alternative':
                    if 'Testing internal branches...' in line_stripped:
                        current_branch_type = 'internal'
                    elif 'Testing external branches...' in line_stripped:
                        current_branch_type = 'external'

                    # Parse Mann-Whitney results from alternative test
                    if 'Mann-Whitney U:' in line_stripped:
                        mw_match = re.search(r'Mann-Whitney U: ([\d.]+), p = ([\d.e-]+)', line_stripped)
                        if mw_match and current_branch_type:
                            if current_branch_type == 'internal':
                                alt_internal_mw_stat = float(mw_match.group(1))
                                alt_internal_mw_pval = float(mw_match.group(2))
                            else:
                                alt_external_mw_stat = float(mw_match.group(1))
                                alt_external_mw_pval = float(mw_match.group(2))

                    # Parse KS results from alternative test
                    if 'Kolmogorov-Smirnov:' in line_stripped:
                        ks_match = re.search(r'Kolmogorov-Smirnov: ([\d.]+), p = ([\d.e-]+)', line_stripped)
                        if ks_match and current_branch_type:
                            if current_branch_type == 'internal':
                                alt_internal_ks_stat = float(ks_match.group(1))
                                alt_internal_ks_pval = float(ks_match.group(2))
                            else:
                                alt_external_ks_stat = float(ks_match.group(1))
                                alt_external_ks_pval = float(ks_match.group(2))

                # Parse summary section for significance results
                elif current_section == 'summary':
                    # Parse significance results for original test
                    if 'Original test:' in line_stripped:
                        if 'INTERNAL BRANCHES:' in lines[max(0, lines.index(line) - 2):lines.index(line)]:
                            current_branch_type = 'internal'
                        elif 'EXTERNAL BRANCHES:' in lines[max(0, lines.index(line) - 2):lines.index(line)]:
                            current_branch_type = 'external'

                        sig_match = re.search(r'significant = (YES|NO), Bonferroni = (YES|NO)', line_stripped)
                        if sig_match:
                            uncorrected_sig = 1 if sig_match.group(1) == 'YES' else 0
                            bonferroni_sig = 1 if sig_match.group(2) == 'YES' else 0

                            # For now, use a simple heuristic: if either branch type is significant,
                            # consider evidence found
                            if current_branch_type == 'internal' or not evidence_binary_uncorrected:
                                evidence_binary_uncorrected = max(evidence_binary_uncorrected, uncorrected_sig)
                                evidence_binary_bonferroni = max(evidence_binary_bonferroni, bonferroni_sig)

            # If we couldn't parse significance from summary, fall back to p-value thresholds
            if evidence_binary_uncorrected == 0 and evidence_binary_bonferroni == 0:
                alpha = 0.05
                bonferroni_alpha = 0.025

                # Check if either branch type shows significance
                uncorrected_sig = (internal_pval < alpha or external_pval < alpha or
                                   alt_internal_mw_pval < alpha or alt_external_mw_pval < alpha or
                                   alt_internal_ks_pval < alpha or alt_external_ks_pval < alpha)
                bonferroni_sig = (internal_pval < bonferroni_alpha or external_pval < bonferroni_alpha or
                                  alt_internal_mw_pval < bonferroni_alpha or alt_external_mw_pval < bonferroni_alpha or
                                  alt_internal_ks_pval < bonferroni_alpha or alt_external_ks_pval < bonferroni_alpha)

                evidence_binary_uncorrected = 1 if uncorrected_sig else 0
                evidence_binary_bonferroni = 1 if bonferroni_sig else 0

            # Set evidence text based on binary values
            evidence_text_uncorrected = "Yes" if evidence_binary_uncorrected else "No"
            evidence_text_bonferroni = "Yes" if evidence_binary_bonferroni else "No"

            # Create entries for both uncorrected and Bonferroni-corrected results
            for evidence_type, evidence_text, evidence_binary in [
                ('uncorrected', evidence_text_uncorrected, evidence_binary_uncorrected),
                ('bonferroni', evidence_text_bonferroni, evidence_binary_bonferroni)
            ]:
                # Determine result classification
                if model == 'BDSKY':
                    result = 'TP' if evidence_binary == 1 else 'FN'
                elif model == 'BDCT':
                    result = 'TN' if evidence_binary == 0 else 'FP'
                else:
                    result = 'UNDEF'

                # Add data to DataFrame
                df.loc[f'{model}.{data_type}.{i}.{evidence_type}',
                ['model', 'tree/forest', 'id', 'evidence_type', 'BDSKY_evidence', 'evidence_binary',
                 'internal_u_stat', 'external_u_stat', 'internal_pval', 'external_pval',
                 'internal_T', 'external_T', 'num_tips', 'result',
                 'internal_cohens_d', 'external_cohens_d', 'internal_relative_diff', 'external_relative_diff',
                 'internal_early_mean', 'external_early_mean', 'internal_late_mean', 'external_late_mean',
                 'internal_effect_significant', 'external_effect_significant',
                 'internal_ks_statistic', 'external_ks_statistic', 'internal_ks_pval', 'external_ks_pval',
                 'internal_ks_significant', 'external_ks_significant', 'test_type',
                 'alt_internal_mw_stat', 'alt_external_mw_stat', 'alt_internal_mw_pval', 'alt_external_mw_pval',
                 'alt_internal_ks_stat', 'alt_external_ks_stat', 'alt_internal_ks_pval', 'alt_external_ks_pval']] = [
                    model, data_type, i, evidence_type, evidence_text, evidence_binary,
                    internal_u_stat, external_u_stat, internal_pval, external_pval,
                    internal_T, external_T, num_tips, result,
                    internal_cohens_d, external_cohens_d, internal_relative_diff, external_relative_diff,
                    internal_early_mean, external_early_mean, internal_late_mean, external_late_mean,
                    internal_effect_significant, external_effect_significant,
                    internal_ks_statistic, external_ks_statistic, internal_ks_pval, external_ks_pval,
                    internal_ks_significant, external_ks_significant, test_type,
                    alt_internal_mw_stat, alt_external_mw_stat, alt_internal_mw_pval, alt_external_mw_pval,
                    alt_internal_ks_stat, alt_external_ks_stat, alt_internal_ks_pval, alt_external_ks_pval
                ]

        except Exception as e:
            logging.error(f"Top-level error processing {log}: {e}")
            continue

    # --- Global statistics ---
    for evidence_type in ['uncorrected', 'bonferroni']:
        df_subset = df[df['evidence_type'] == evidence_type]

        TP = len(df_subset[df_subset['result'] == 'TP'])
        TN = len(df_subset[df_subset['result'] == 'TN'])
        FP = len(df_subset[df_subset['result'] == 'FP'])
        FN = len(df_subset[df_subset['result'] == 'FN'])

        print(f'Global ({evidence_type}):')
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
        specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else float('nan')
        precision = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
        print(f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}')
        print(f'\tSensitivity={sensitivity:.4f}, Specificity={specificity:.4f}')
        print(f'\tAccuracy={accuracy:.4f}, Precision={precision:.4f}')
        print('==============\n')

    # --- Statistics by model ---
    numeric_columns = ['evidence_binary', 'internal_u_stat', 'external_u_stat',
                       'internal_pval', 'external_pval', 'internal_T', 'external_T', 'num_tips',
                       'internal_cohens_d', 'external_cohens_d', 'internal_relative_diff', 'external_relative_diff',
                       'internal_early_mean', 'external_early_mean', 'internal_late_mean', 'external_late_mean',
                       'internal_ks_statistic', 'external_ks_statistic', 'internal_ks_pval', 'external_ks_pval',
                       'alt_internal_mw_stat', 'alt_external_mw_stat', 'alt_internal_mw_pval', 'alt_external_mw_pval',
                       'alt_internal_ks_stat', 'alt_external_ks_stat', 'alt_internal_ks_pval', 'alt_external_ks_pval']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    boolean_columns = ['internal_effect_significant', 'external_effect_significant',
                       'internal_ks_significant', 'external_ks_significant']
    for col in boolean_columns:
        df[col] = df[col].astype(bool)

    for evidence_type in ['uncorrected', 'bonferroni']:
        print(f'\n=== {evidence_type.upper()} RESULTS ===')
        df_evidence = df[df['evidence_type'] == evidence_type]

        for test_type in df_evidence['test_type'].unique():
            df_test_type = df_evidence[df_evidence['test_type'] == test_type]
            print(f'\n--- {test_type.upper()} TEST TYPE ---')

            for model in df_test_type['model'].unique():
                for data_type in df_test_type['tree/forest'].unique():
                    ddf = df_test_type[(df_test_type['model'] == model) & (df_test_type['tree/forest'] == data_type)]
                    if len(ddf) == 0:
                        continue

                    # Basic statistics
                    mean_evidence = ddf['evidence_binary'].mean()
                    mean_internal_u = ddf['internal_u_stat'].mean()
                    mean_external_u = ddf['external_u_stat'].mean()
                    mean_internal_pval = ddf['internal_pval'].mean()
                    mean_external_pval = ddf['external_pval'].mean()
                    mean_internal_T = ddf['internal_T'].mean()
                    mean_external_T = ddf['external_T'].mean()

                    # Alternative test statistics
                    mean_alt_internal_mw_pval = ddf['alt_internal_mw_pval'].mean()
                    mean_alt_external_mw_pval = ddf['alt_external_mw_pval'].mean()
                    mean_alt_internal_ks_pval = ddf['alt_internal_ks_pval'].mean()
                    mean_alt_external_ks_pval = ddf['alt_external_ks_pval'].mean()

                    num_detected = len(ddf[ddf['evidence_binary'] == 1])
                    percentage_detected = 100 * num_detected / len(ddf) if len(ddf) > 0 else 0.0

                    TP_model = len(ddf[ddf['result'] == 'TP'])
                    TN_model = len(ddf[ddf['result'] == 'TN'])
                    FP_model = len(ddf[ddf['result'] == 'FP'])
                    FN_model = len(ddf[ddf['result'] == 'FN'])

                    tips_min, tips_mean, tips_max = ddf['num_tips'].min(), ddf['num_tips'].mean(), ddf['num_tips'].max()

                    print(f'{model} on {data_type}s ({evidence_type}, {test_type}):')

                    sensitivity_model = TP_model / (TP_model + FN_model) if (TP_model + FN_model) > 0 else float('nan')
                    specificity_model = TN_model / (TN_model + FP_model) if (TN_model + FP_model) > 0 else float('nan')
                    accuracy_model = (TP_model + TN_model) / (TP_model + TN_model + FP_model + FN_model) if (
                                                                                                                        TP_model + TN_model + FP_model + FN_model) > 0 else float(
                        'nan')
                    precision_model = TP_model / (TP_model + FP_model) if (TP_model + FP_model) > 0 else float('nan')

                    print(f'\tTP={TP_model}, TN={TN_model}, FP={FP_model}, FN={FN_model}')
                    print(f'\tSensitivity={sensitivity_model:.4f}, Specificity={specificity_model:.4f}')
                    print(f'\tAccuracy={accuracy_model:.4f}, Precision={precision_model:.4f}')

                    print('--------Original test metrics:')
                    print(f'\tSkyline detected\t{percentage_detected:.1f}%')
                    print(f'\tavg evidence rate\t{mean_evidence:.3f}')
                    print(f'\tavg internal U stat\t{mean_internal_u:.4f}')
                    print(f'\tavg external U stat\t{mean_external_u:.4f}')
                    print(f'\tavg internal p-val\t{mean_internal_pval:.6f}')
                    print(f'\tavg external p-val\t{mean_external_pval:.6f}')
                    print(f'\tavg internal T\t{mean_internal_T:.4f}')
                    print(f'\tavg external T\t{mean_external_T:.4f}')

                    print('--------Alternative test metrics:')
                    print(f'\tavg alt internal MW p-val\t{mean_alt_internal_mw_pval:.6f}')
                    print(f'\tavg alt external MW p-val\t{mean_alt_external_mw_pval:.6f}')
                    print(f'\tavg alt internal KS p-val\t{mean_alt_internal_ks_pval:.6f}')
                    print(f'\tavg alt external KS p-val\t{mean_alt_external_ks_pval:.6f}')

                    print(f'\tavg num tips\t{tips_mean:.0f}\t[{tips_min}-{tips_max}]')

                    print('--------FN/FP:')
                    ddff = ddf[(ddf['result'] == 'FN') | (ddf['result'] == 'FP')]

                    if len(ddff) > 0:
                        internal_u_list = '\t'.join(f'{float(_):.4f}' for _ in ddff['internal_u_stat'].to_list())
                        external_u_list = '\t'.join(f'{float(_):.4f}' for _ in ddff['external_u_stat'].to_list())
                        evidence_list = '\t'.join(f'{_}' for _ in ddff['BDSKY_evidence'].astype(str).to_list())
                        print(f'\tinternal U stats\t{internal_u_list}')
                        print(f'\texternal U stats\t{external_u_list}')
                        print(f'\tevidence\t{evidence_list}')
                    else:
                        print('\tNo FN/FP entries for this model/data type.')

                    print('==============\n')

    df.sort_values(by=['model', 'tree/forest', 'id', 'evidence_type'], inplace=True)

    # Save the dataframe to the specified tab file
    if params.tab:
        try:
            df.to_csv(params.tab, sep='\t', index=False)
            logging.info(f"Summary table saved to {params.tab}")
        except Exception as e:
            logging.error(f"Error saving summary table to {params.tab}: {e}")
    else:
        logging.warning("No output table file specified. Table will not be saved.")