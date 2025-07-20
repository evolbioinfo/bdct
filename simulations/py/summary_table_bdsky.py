import re
import pandas as pd
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analyze_fn_case(log_content, T_value, num_tips):
    """
    Analyze FN cases to understand why skyline was not detected.
    Returns a dictionary with analysis metrics.
    """
    analysis = {
        'early_branch_count': 0,
        'late_branch_count': 0,
        'interval_balance': 0.0,
        'short_interval_flag': False,
        'T_percentage': 0.0,
        'extreme_T_flag': False,
        'u_stat_closeness': 'N/A',
        'pval_closeness': 'N/A',
        'fn_explanation': 'Unknown',
        'T_top': 'N/A',
        'T_bottom': 'N/A',
        'T_fallback': 'N/A',
        'dominant_criterion': 'N/A'
    }

    internal_early_count = 0
    internal_late_count = 0
    external_early_count = 0
    external_late_count = 0

    t_top_match = re.search(r'T_top \(tip accumulation\):\s*([\d.]+)', log_content)
    if t_top_match:
        analysis['T_top'] = float(t_top_match.group(1))

    t_bottom_match = re.search(r'T_bottom \(bottom structure\):\s*([\d.]+|N/A)', log_content)
    if t_bottom_match and t_bottom_match.group(1) != 'N/A':
        try:
            analysis['T_bottom'] = float(t_bottom_match.group(1))
        except ValueError:
            analysis['T_bottom'] = 'N/A'

    t_fallback_match = re.search(r'T_fallback \(midpoint\):\s*([\d.]+)', log_content)
    if t_fallback_match:
        analysis['T_fallback'] = float(t_fallback_match.group(1))

    dominant_match = re.search(r'Dominant criterion:\s*([^\n]+)', log_content)
    if dominant_match:
        analysis['dominant_criterion'] = dominant_match.group(1).strip()

    internal_match = re.search(
        r'Internal branches \((?:new strategy|robust strategy)(?: comparison)?\):\s*'
        r'.*?Early subtree interval: .*?\((\d+) branches\)\s*'
        r'.*?Late subtree interval: .*?\((\d+) branches\)',
        log_content, re.DOTALL
    )
    if internal_match:
        internal_early_count = int(internal_match.group(1))
        internal_late_count = int(internal_match.group(2))

    external_match = re.search(
        r'External branches \((?:new strategy|robust strategy)(?: comparison)?\):\s*'
        r'.*?Early subtree interval: .*?\((\d+) branches\)\s*'
        r'.*?Late subtree interval: .*?\((\d+) branches\)',
        log_content, re.DOTALL
    )
    if external_match:
        external_early_count = int(external_match.group(1))
        external_late_count = int(external_match.group(2))

    total_early = internal_early_count + external_early_count
    total_late = internal_late_count + external_late_count

    analysis['early_branch_count'] = total_early
    analysis['late_branch_count'] = total_late

    if total_early > 0 and total_late > 0:
        analysis['interval_balance'] = min(total_early, total_late) / max(total_early, total_late)

    total_branches = total_early + total_late
    if total_branches > 0:
        min_interval_ratio = min(total_early, total_late) / total_branches
        analysis['short_interval_flag'] = min_interval_ratio < 0.1

    tree_height_match = re.search(r'Tree height:\s*([\d.]+)', log_content)
    if tree_height_match and T_value > 0:
        tree_height = float(tree_height_match.group(1))
        if tree_height > 0:
            analysis['T_percentage'] = (T_value / tree_height) * 100
            analysis['extreme_T_flag'] = analysis['T_percentage'] < 10 or analysis['T_percentage'] > 90

    internal_u_match = re.search(r'Internal branches.*?Mann-Whitney U statistic: ([\d.]+)', log_content, re.DOTALL)
    external_u_match = re.search(r'External branches.*?Mann-Whitney U statistic: ([\d.]+)', log_content, re.DOTALL)

    if internal_u_match and external_u_match:
        internal_u = float(internal_u_match.group(1))
        external_u = float(external_u_match.group(1))

        if internal_early_count > 0 and internal_late_count > 0:
            expected_internal_u = (internal_early_count * internal_late_count) / 2
            internal_deviation = abs(
                internal_u - expected_internal_u) / expected_internal_u if expected_internal_u > 0 else 0
        else:
            internal_deviation = 0

        if external_early_count > 0 and external_late_count > 0:
            expected_external_u = (external_early_count * external_late_count) / 2
            external_deviation = abs(
                external_u - expected_external_u) / expected_external_u if expected_external_u > 0 else 0
        else:
            external_deviation = 0

        avg_deviation = (internal_deviation + external_deviation) / 2 if (
                                                                                     internal_deviation + external_deviation) > 0 else 0

        if avg_deviation < 0.05:
            analysis['u_stat_closeness'] = 'Very close (±5%)'
        elif avg_deviation < 0.10:
            analysis['u_stat_closeness'] = 'Close (±10%)'
        elif avg_deviation < 0.20:
            analysis['u_stat_closeness'] = 'Moderate (±20%)'
        else:
            analysis['u_stat_closeness'] = 'Different (>20%)'

    internal_pval_match = re.search(r'Internal branches.*?p-value: ([\d.e-]+)', log_content, re.DOTALL)
    external_pval_match = re.search(r'External branches.*?p-value: ([\d.e-]+)', log_content, re.DOTALL)

    min_pval = 1.0
    if internal_pval_match:
        internal_pval = float(internal_pval_match.group(1))
        min_pval = min(min_pval, internal_pval)
    if external_pval_match:
        external_pval = float(external_pval_match.group(1))
        min_pval = min(min_pval, external_pval)

    if min_pval < 1.0:
        if min_pval > 0.2:
            analysis['pval_closeness'] = 'Far from significant (>0.2)'
        elif min_pval > 0.1:
            analysis['pval_closeness'] = 'Moderate (0.1-0.2)'
        elif min_pval > 0.05:
            analysis['pval_closeness'] = 'Close to significant (0.05-0.1)'
        else:
            analysis['pval_closeness'] = 'Significant (<0.05)'

    explanations = []

    if analysis['short_interval_flag']:
        explanations.append("One interval too short")

    if analysis['extreme_T_flag']:
        if analysis['T_percentage'] < 10:
            explanations.append("T too early in tree")
        else:
            explanations.append("T too late in tree")

    if analysis['u_stat_closeness'] in ['Very close (±5%)', 'Close (±10%)']:
        explanations.append("Branch lengths too similar")

    if analysis['interval_balance'] > 0.8:
        explanations.append("Well-balanced intervals")

    if not explanations:
        explanations.append("Insufficient signal")

    analysis['fn_explanation'] = "; ".join(explanations)

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize results from the new BD-Skyline test (bdsky_test.py log files) with FN analysis.")
    parser.add_argument('--logs', nargs='+', type=str,
                        help="BD-Skyline test results log files generated by bdsky_test.py")
    parser.add_argument('--tab', type=str, help="Output summary table file (TSV format)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for debugging parsing issues")
    params = parser.parse_args()

    df = pd.DataFrame(
        columns=['model', 'tree/forest', 'id', 'BDSKY_evidence', 'evidence_binary',
                 'internal_u_stat', 'external_u_stat', 'internal_pval', 'external_pval',
                 'T_value', 'T_percentage', 'T_top', 'T_bottom', 'T_fallback', 'dominant_criterion',
                 'num_tips', 'result', 'test_type', 'filepath',
                 'early_branches', 'late_branches', 'interval_balance', 'short_interval_flag',
                 'extreme_T_flag', 'u_stat_closeness', 'pval_closeness', 'fn_explanation'])

    if not params.logs:
        logging.error("No log files provided. Please use --logs to specify input log files.")
    else:
        for log_path in params.logs:
            logging.debug(f"Processing log file: {log_path}")
            try:
                with open(log_path, 'r') as f:
                    content = f.read()

                match_id = re.findall(r'(\d+)(?=\.bdsky_test|\.log|$)', os.path.basename(log_path))
                i = int(match_id[-1]) if match_id else 0

                model = 'UNKNOWN'
                path_lower = log_path.lower()
                dirname_lower = os.path.dirname(log_path).lower()
                basename_lower = os.path.basename(log_path).lower()

                if 'bdct0' in dirname_lower:
                    model = 'BD'
                elif 'simulations_bdsky' in dirname_lower or 'bdsky' in dirname_lower:
                    model = 'BDSKY'
                elif 'bd' in dirname_lower and 'bdsky' not in dirname_lower:
                    model = 'BD'
                elif 'bdsky_test' in basename_lower and model == 'UNKNOWN':
                    model = 'BDSKY'
                data_type = 'tree'
                test_type = 'robust_strategy'

                evidence_binary = 0
                internal_u_stat = float('nan')
                external_u_stat = float('nan')
                internal_pval = float('nan')
                external_pval = float('nan')
                T_value = float('nan')
                T_percentage = float('nan')
                T_top = float('nan')
                T_bottom = float('nan')
                T_fallback = float('nan')
                dominant_criterion = 'N/A'
                num_tips = 0

                tips_match = re.search(r'Total tips in tree:\s*(\d+)', content)
                if tips_match:
                    num_tips = int(tips_match.group(1))

                t_match = re.search(r'Selected T:\s*([\d.]+)', content)
                if not t_match:
                    t_match = re.search(r'Time \(T\) based on \d+ tips \(N/\d+\):\s*([\d.]+)', content)
                if not t_match:
                    t_match = re.search(r'Time \(T\) based on \d+ tips:\s*([\d.]+)', content)
                if t_match:
                    T_value = float(t_match.group(1))

                t_perc_match = re.search(r'Selected T:\s*[\d.]+\s*\(([\d.]+)% of tree height\)', content)
                if not t_perc_match:
                    t_perc_match = re.search(
                        r'Time \(T\) based on \d+ tips.*?:\s*[\d.]+\s*\(([\d.]+)% of tree height\)', content)
                if t_perc_match:
                    T_percentage = float(t_perc_match.group(1))

                t_top_match = re.search(r'T_top \(tip accumulation\):\s*([\d.]+)', content)
                if t_top_match:
                    T_top = float(t_top_match.group(1))

                t_bottom_match = re.search(r'T_bottom \(bottom structure\):\s*([\d.]+|N/A)', content)
                if t_bottom_match and t_bottom_match.group(1) != 'N/A':
                    try:
                        T_bottom = float(t_bottom_match.group(1))
                    except ValueError:
                        T_bottom = float('nan')

                t_fallback_match = re.search(r'T_fallback \(midpoint\):\s*([\d.]+)', content)
                if t_fallback_match:
                    T_fallback = float(t_fallback_match.group(1))

                dominant_match = re.search(r'Dominant criterion:\s*([^\n]+)', content)
                if dominant_match:
                    dominant_criterion = dominant_match.group(1).strip()

                internal_section_match = re.search(
                    r'Internal branches \((?:new strategy|robust strategy)(?: comparison)?\):\s*'
                    r'.*?T used = ([\d.]+)\s*'
                    r'.*?Early subtree interval: .*?\n'
                    r'.*?Late subtree interval: .*?\n'
                    r'.*?Mann-Whitney U statistic: ([\d.]+)\s*'
                    r'.*?p-value: ([\d.e-]+)', content, re.DOTALL
                )
                if internal_section_match:
                    internal_u_stat = float(internal_section_match.group(2))
                    internal_pval = float(internal_section_match.group(3))
                else:
                    logging.debug(f"No internal branch section found in {log_path}")
                    if "Internal branches: Not enough data for comparison." in content:
                        internal_pval = 'N/A (Insufficient Data)'

                external_section_match = re.search(
                    r'External branches \((?:new strategy|robust strategy)(?: comparison)?\):\s*'
                    r'.*?T used = ([\d.]+)\s*'
                    r'.*?Early subtree interval: .*?\n'
                    r'.*?Late subtree interval: .*?\n'
                    r'.*?Mann-Whitney U statistic: ([\d.]+)\s*'
                    r'.*?p-value: ([\d.e-]+)', content, re.DOTALL
                )
                if external_section_match:
                    external_u_stat = float(external_section_match.group(2))
                    external_pval = float(external_section_match.group(3))
                else:
                    logging.debug(f"No external branch section found in {log_path}")
                    if "External branches: Not enough data for comparison." in content:
                        external_pval = 'N/A (Insufficient Data)'

                evidence_detected_line = re.search(r'(?:NEW|ENHANCED) SKY test: Evidence of BD-Skyline model detected',
                                                   content)
                no_evidence_detected_line = re.search(
                    r'(?:NEW|ENHANCED) SKY test: No evidence of BD-Skyline model \(consistent with simple BD\)',
                    content)

                if evidence_detected_line:
                    evidence_binary = 1
                elif no_evidence_detected_line:
                    evidence_binary = 0
                else:
                    alpha = 0.05
                    if (isinstance(internal_pval, float) and not pd.isna(internal_pval) and internal_pval < alpha) or \
                            (isinstance(external_pval, float) and not pd.isna(external_pval) and external_pval < alpha):
                        evidence_binary = 1
                    else:
                        evidence_binary = 0

                evidence_text = "Yes" if evidence_binary == 1 else "No"

                result = 'UNDEF'
                if model == 'BDSKY':
                    result = 'TP' if evidence_binary == 1 else 'FN'
                elif model == 'BD':
                    result = 'TN' if evidence_binary == 0 else 'FP'

                fn_analysis = analyze_fn_case(content, T_value, num_tips)

                if params.verbose:
                    logging.debug(f"File: {log_path}")
                    logging.debug(f"Model: {model}")
                    logging.debug(f"ID: {i}")
                    logging.debug(f"Num Tips: {num_tips}")
                    logging.debug(f"T Value: {T_value:.4f}")
                    logging.debug(f"Internal U stat: {internal_u_stat:.4f}, Pval: {internal_pval}")
                    logging.debug(f"External U stat: {external_u_stat:.4f}, Pval: {external_pval}")
                    logging.debug(f"Evidence: {evidence_text} ({evidence_binary})")
                    if result == 'FN':
                        logging.debug(f"FN Analysis: {fn_analysis}")

                df.loc[f'{os.path.basename(log_path)}',
                ['model', 'tree/forest', 'id', 'BDSKY_evidence', 'evidence_binary',
                 'internal_u_stat', 'external_u_stat', 'internal_pval', 'external_pval',
                 'T_value', 'T_percentage', 'T_top', 'T_bottom', 'T_fallback', 'dominant_criterion',
                 'num_tips', 'result', 'test_type', 'filepath',
                 'early_branches', 'late_branches', 'interval_balance', 'short_interval_flag',
                 'extreme_T_flag', 'u_stat_closeness', 'pval_closeness', 'fn_explanation']] = [
                    model, data_type, i, evidence_text, evidence_binary,
                    internal_u_stat, external_u_stat, internal_pval, external_pval,
                    T_value, T_percentage, T_top, T_bottom, T_fallback, dominant_criterion,
                    num_tips, result, test_type, log_path,
                    fn_analysis['early_branch_count'], fn_analysis['late_branch_count'],
                    fn_analysis['interval_balance'], fn_analysis['short_interval_flag'],
                    fn_analysis['extreme_T_flag'], fn_analysis['u_stat_closeness'],
                    fn_analysis['pval_closeness'], fn_analysis['fn_explanation']
                ]

            except Exception as e:
                logging.error(f"Error processing {log_path}: {e}")
                if params.verbose:
                    import traceback

                    logging.error(traceback.format_exc())
                continue

        print("\n--- Models identified from log file paths ---")
        if not df.empty:
            print(df['model'].value_counts().to_string())
        else:
            print("No models identified from any log files.")
        print("-------------------------------------------\n")

    numeric_columns = ['evidence_binary', 'internal_u_stat', 'external_u_stat',
                       'internal_pval', 'external_pval', 'T_value', 'T_percentage', 'T_top', 'T_bottom', 'T_fallback',
                       'num_tips', 'early_branches', 'late_branches', 'interval_balance']
    for col in numeric_columns:
        if col in ['internal_pval', 'external_pval']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print('\n' + '=' * 80)
    print("Summary Statistics")
    print('=' * 80)

    TP = len(df[df['result'] == 'TP'])
    TN = len(df[df['result'] == 'TN'])
    FP = len(df[df['result'] == 'FP'])
    FN = len(df[df['result'] == 'FN'])
    UNDEF = len(df[df['result'] == 'UNDEF'])

    print(f'Global Results:')
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else float('nan')
    precision = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    print(f'\tTP={TP}, TN={TN}, FP={FP}, FN={FN}, UNDEF={UNDEF}')
    print(f'\tSensitivity={sensitivity:.4f}, Specificity={specificity:.4f}')
    print(f'\tAccuracy={accuracy:.4f}, Precision={precision:.4f}')
    print('==============\n')

    fn_cases = df[df['result'] == 'FN']
    if len(fn_cases) > 0:
        print(f'\n=== FALSE NEGATIVE ANALYSIS ({len(fn_cases)} cases) ===')
        print(f'Common explanations for FN:')

        explanations = {}
        for explanation in fn_cases['fn_explanation']:
            for exp in str(explanation).split(';'):
                exp = exp.strip()
                explanations[exp] = explanations.get(exp, 0) + 1

        for exp, count in sorted(explanations.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(fn_cases)) * 100
            print(f'\t{exp}: {count} cases ({percentage:.1f}%)')

        print(f'\nFN Parameter Analysis:')
        print(f'\tAverage T percentage: {fn_cases["T_percentage"].mean():.1f}% of tree height')
        print(f'\tAverage interval balance: {fn_cases["interval_balance"].mean():.3f}')
        print(
            f'\tShort interval cases: {fn_cases["short_interval_flag"].sum()} ({(fn_cases["short_interval_flag"].sum() / len(fn_cases) * 100):.1f}%)')
        print(
            f'\tExtreme T cases: {fn_cases["extreme_T_flag"].sum()} ({(fn_cases["extreme_T_flag"].sum() / len(fn_cases) * 100):.1f}%)')

        print(f'\nRobust T Analysis:')
        criterion_counts = fn_cases['dominant_criterion'].value_counts()
        for criterion, count in criterion_counts.items():
            percentage = (count / len(fn_cases)) * 100
            print(f'\t{criterion}: {count} cases ({percentage:.1f}%)')

        if not fn_cases['T_top'].isnull().all():
            print(f'\tAverage T_top: {fn_cases["T_top"].mean():.3f}')
        if not fn_cases['T_bottom'].isnull().all():
            print(f'\tAverage T_bottom: {fn_cases["T_bottom"].mean():.3f}')
        if not fn_cases['T_fallback'].isnull().all():
            print(f'\tAverage T_fallback: {fn_cases["T_fallback"].mean():.3f}')

        print(f'\nU statistic closeness distribution:')
        u_closeness_counts = fn_cases['u_stat_closeness'].value_counts()
        for closeness, count in u_closeness_counts.items():
            percentage = (count / len(fn_cases)) * 100
            print(f'\t{closeness}: {count} cases ({percentage:.1f}%)')

        print('==============\n')

    print(f'\n=== RESULTS BY MODEL ===')

    for test_type in df['test_type'].unique():
        df_test_type = df[df['test_type'] == test_type]
        print(f'\n--- {test_type.upper()} TEST TYPE ---')

        for model in df_test_type['model'].unique():
            for data_type in df_test_type['tree/forest'].unique():
                ddf = df_test_type[(df_test_type['model'] == model) & (df_test_type['tree/forest'] == data_type)]
                if len(ddf) == 0:
                    continue

                mean_evidence = ddf['evidence_binary'].mean()
                mean_internal_u = ddf['internal_u_stat'].mean()
                mean_external_u = ddf['external_u_stat'].mean()
                mean_internal_pval = ddf['internal_pval'].mean()
                mean_external_pval = ddf['external_pval'].mean()
                mean_T = ddf['T_value'].mean()
                mean_T_perc = ddf['T_percentage'].mean()
                mean_T_top = ddf['T_top'].mean()
                mean_T_bottom = ddf['T_bottom'].mean()
                mean_T_fallback = ddf['T_fallback'].mean()

                num_detected = len(ddf[ddf['evidence_binary'] == 1])
                percentage_detected = 100 * num_detected / len(ddf) if len(ddf) > 0 else 0.0

                TP_model = len(ddf[ddf['result'] == 'TP'])
                TN_model = len(ddf[ddf['result'] == 'TN'])
                FP_model = len(ddf[ddf['result'] == 'FP'])
                FN_model = len(ddf[ddf['result'] == 'FN'])

                tips_min = ddf['num_tips'].min() if not ddf['num_tips'].isnull().all() else float('nan')
                tips_mean = ddf['num_tips'].mean() if not ddf['num_tips'].isnull().all() else float('nan')
                tips_max = ddf['num_tips'].max() if not ddf['num_tips'].isnull().all() else float('nan')

                print(f'{model} on {data_type}s ({test_type}):')

                sensitivity_model = TP_model / (TP_model + FN_model) if (TP_model + FN_model) > 0 else float('nan')
                specificity_model = TN_model / (TN_model + FP_model) if (TN_model + FP_model) > 0 else float('nan')
                accuracy_model = (TP_model + TN_model) / (TP_model + TN_model + FP_model + FN_model) if (
                                                                                                                TP_model + TN_model + FP_model + FN_model) > 0 else float(
                    'nan')
                precision_model = TP_model / (TP_model + FP_model) if (TP_model + FP_model) > 0 else float('nan')

                print(f'\tTP={TP_model}, TN={TN_model}, FP={FP_model}, FN={FN_model}')
                print(f'\tSensitivity={sensitivity_model:.4f}, Specificity={specificity_model:.4f}')
                print(f'\tAccuracy={accuracy_model:.4f}, Precision={precision_model:.4f}')

                print('--------Test metrics:')
                print(f'\tSkyline detected\t{percentage_detected:.1f}%')
                print(f'\tavg evidence rate\t{mean_evidence:.3f}')
                print(f'\tavg internal U stat\t{mean_internal_u:.4f}')
                print(f'\tavg external U stat\t{mean_external_u:.4f}')
                print(f'\tavg internal p-val\t{mean_internal_pval:.6f}')
                print(f'\tavg external p-val\t{mean_external_pval:.6f}')
                print(f'\tavg T\t{mean_T:.4f}')
                print(f'\tavg T percentage\t{mean_T_perc:.1f}%')
                print(f'\tavg T_top\t{mean_T_top:.4f}' if not pd.isna(mean_T_top) else '\tavg T_top\tNaN')
                print(f'\tavg T_bottom\t{mean_T_bottom:.4f}' if not pd.isna(mean_T_bottom) else '\tavg T_bottom\tNaN')
                print(f'\tavg T_fallback\t{mean_T_fallback:.4f}' if not pd.isna(
                    mean_T_fallback) else '\tavg T_fallback\tNaN')
                print(f'\tavg num tips\t{tips_mean:.0f}\t[{tips_min:.0f}-{tips_max:.0f}]' if not pd.isna(
                    tips_mean) else '\tavg num tips\tNaN')

                print('--------FN/FP details:')
                ddff = ddf[(ddf['result'] == 'FN') | (ddf['result'] == 'FP')]

                if len(ddff) > 0:
                    internal_u_list = '\t'.join(
                        f'{float(_):.4f}' if pd.notna(_) else 'nan' for _ in ddff['internal_u_stat'].to_list())
                    external_u_list = '\t'.join(
                        f'{float(_):.4f}' if pd.notna(_) else 'nan' for _ in ddff['external_u_stat'].to_list())
                    evidence_list = '\t'.join(f'{_}' for _ in ddff['BDSKY_evidence'].astype(str).to_list())

                    print(f'\tinternal U stats\t{internal_u_list}')
                    print(f'\texternal U stats\t{external_u_list}')
                    print(f'\tevidence\t{evidence_list}')

                    file_list = '\t'.join(f'{os.path.basename(_)}' for _ in ddff['filepath'].to_list())
                    print(f'\tfilenames\t{file_list}')

                    fn_subset = ddff[ddff['result'] == 'FN']
                    if len(fn_subset) > 0:
                        print(f'\t--- FN Analysis ({len(fn_subset)} cases) ---')

                        t_perc_list = '\t'.join(
                            f'{float(_):.1f}%' if pd.notna(_) else 'nan' for _ in fn_subset['T_percentage'].to_list())
                        print(f'\tT percentages\t{t_perc_list}')

                        t_top_list = '\t'.join(
                            f'{float(_):.3f}' if pd.notna(_) else 'nan' for _ in fn_subset['T_top'].to_list())
                        print(f'\tT_top values\t{t_top_list}')

                        t_bottom_list = '\t'.join(
                            f'{float(_):.3f}' if pd.notna(_) else 'nan' for _ in fn_subset['T_bottom'].to_list())
                        print(f'\tT_bottom values\t{t_bottom_list}')

                        t_fallback_list = '\t'.join(
                            f'{float(_):.3f}' if pd.notna(_) else 'nan' for _ in fn_subset['T_fallback'].to_list())
                        print(f'\tT_fallback values\t{t_fallback_list}')

                        criterion_list = '\t'.join(str(_) for _ in fn_subset['dominant_criterion'].to_list())
                        print(f'\tDominant criteria\t{criterion_list}')

                        balance_list = '\t'.join(
                            f'{float(_):.3f}' if pd.notna(_) else 'nan' for _ in
                            fn_subset['interval_balance'].to_list())
                        print(f'\tInterval balance\t{balance_list}')

                        u_close_list = '\t'.join(str(_) for _ in fn_subset['u_stat_closeness'].to_list())
                        print(f'\tU stat closeness\t{u_close_list}')

                        pval_close_list = '\t'.join(str(_) for _ in fn_subset['pval_closeness'].to_list())
                        print(f'\tP-val closeness\t{pval_close_list}')

                        explanation_list = '\t'.join(str(_) for _ in fn_subset['fn_explanation'].to_list())
                        print(f'\tFN explanations\t{explanation_list}')
                else:
                    print('\tNo FN/FP entries for this model/data type.')

                print('==============\n')

    df.sort_values(by=['model', 'tree/forest', 'id'], inplace=True)


    if params.tab:
        try:
            df.to_csv(params.tab, sep='\t', index=False)
            logging.info(f"Summary table saved to {params.tab}")
        except Exception as e:
            logging.error(f"Error saving summary table to {params.tab}: {e}")
    else:
        logging.warning("No output table file specified. Table will not be saved.")