#!/usr/bin/env python3
"""
Combine balanced BDSKY test results into a summary table.
"""

import argparse
import pandas as pd
import re
from pathlib import Path


def parse_balanced_log(log_file):
    """Parse a balanced skyline test log file."""

    result = {
        'evidence_found': False,
        'bonferroni_evidence': False,
        'internal_T': None,
        'external_T': None,
        'internal_early_count': 0,
        'internal_late_count': 0,
        'external_early_count': 0,
        'external_late_count': 0,
        'internal_u_stat': None,
        'external_u_stat': None,
        'internal_pval': None,
        'external_pval': None,
        'internal_effect_size': None,
        'external_effect_size': None,
        'internal_significant': False,
        'external_significant': False,
        'internal_strong_evidence': False,
        'external_strong_evidence': False,
        'num_tips': None,
        'parameter_set': None
    }

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Extract basic results
        if 'Evidence of BD-Skyline model detected (Bonferroni corrected)' in content:
            result['evidence_found'] = True
            result['bonferroni_evidence'] = True
        elif 'Evidence of BD-Skyline model detected' in content:
            result['evidence_found'] = True
            result['bonferroni_evidence'] = False

        # Extract number of tips
        tips_match = re.search(r'Total tips in tree: (\d+)', content)
        if tips_match:
            result['num_tips'] = int(tips_match.group(1))

        # Extract parameter set from filename
        log_path = Path(log_file)
        param_match = re.search(r'\.balanced_bdsky_test\.(\w+)$', log_path.name)
        if param_match:
            result['parameter_set'] = param_match.group(1)

        # Extract internal branch results
        internal_section = re.search(r'Internal branches \(.*?\):(.*?)(?=External branches|$)', content, re.DOTALL)
        if internal_section:
            internal_text = internal_section.group(1)

            # Extract T value
            t_match = re.search(r'T = ([\d.]+)', internal_text)
            if t_match:
                result['internal_T'] = float(t_match.group(1))

            # Extract interval counts
            early_match = re.search(r'Early interval.*?(\d+) branches', internal_text)
            late_match = re.search(r'Late interval.*?(\d+) branches', internal_text)
            if early_match:
                result['internal_early_count'] = int(early_match.group(1))
            if late_match:
                result['internal_late_count'] = int(late_match.group(1))

            # Extract statistics
            u_match = re.search(r'Mann-Whitney U statistic: ([\d.]+)', internal_text)
            p_match = re.search(r'p-value: ([\d.e-]+)', internal_text)
            effect_match = re.search(r'Effect size \(Cohen\'s d\): ([\d.]+)', internal_text)

            if u_match:
                result['internal_u_stat'] = float(u_match.group(1))
            if p_match:
                result['internal_pval'] = float(p_match.group(1))
            if effect_match:
                result['internal_effect_size'] = float(effect_match.group(1))

            # Extract significance flags
            if re.search(r'Significant.*?: True', internal_text):
                result['internal_significant'] = True
            if re.search(r'Strong evidence.*?: True', internal_text):
                result['internal_strong_evidence'] = True

        # Extract external branch results
        external_section = re.search(r'External branches \(.*?\):(.*?)(?=\n\n|$)', content, re.DOTALL)
        if external_section:
            external_text = external_section.group(1)

            # Extract T value
            t_match = re.search(r'T = ([\d.]+)', external_text)
            if t_match:
                result['external_T'] = float(t_match.group(1))

            # Extract interval counts
            early_match = re.search(r'Early interval.*?(\d+) branches', external_text)
            late_match = re.search(r'Late interval.*?(\d+) branches', external_text)
            if early_match:
                result['external_early_count'] = int(early_match.group(1))
            if late_match:
                result['external_late_count'] = int(late_match.group(1))

            # Extract statistics
            u_match = re.search(r'Mann-Whitney U statistic: ([\d.]+)', external_text)
            p_match = re.search(r'p-value: ([\d.e-]+)', external_text)
            effect_match = re.search(r'Effect size \(Cohen\'s d\): ([\d.]+)', external_text)

            if u_match:
                result['external_u_stat'] = float(u_match.group(1))
            if p_match:
                result['external_pval'] = float(p_match.group(1))
            if effect_match:
                result['external_effect_size'] = float(effect_match.group(1))

            # Extract significance flags
            if re.search(r'Significant.*?: True', external_text):
                result['external_significant'] = True
            if re.search(r'Strong evidence.*?: True', external_text):
                result['external_strong_evidence'] = True

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Combine balanced BDSKY test results")
    parser.add_argument('--logs', nargs='+', required=True, help="Log files to process")
    parser.add_argument('--tab', required=True, help="Output table file")
    parser.add_argument('--parameter-sets', nargs='+', required=True, help="Parameter set names")

    args = parser.parse_args()

    results = []

    print(f"Processing {len(args.logs)} log files...")

    for log_file in args.logs:
        log_path = Path(log_file)

        # Debug output
        print(f"Processing: {log_file}")

        # Determine model type from path
        log_path_str = str(log_path)
        if 'bdsky' in log_path_str.lower() or 'BDSKY' in log_path_str:
            model_type = 'BDSKY'
        elif 'bdct' in log_path_str.lower() or 'BDCT' in log_path_str:
            model_type = 'BDCT'
        else:
            # Try to determine from filename patterns
            if 'final_tree' in log_path.name:
                model_type = 'BDSKY'  # final_tree usually indicates BDSKY
            elif 'tree.' in log_path.name:
                model_type = 'BDCT'  # tree.X usually indicates BDCT
            else:
                model_type = 'UNKNOWN'
                print(f"Warning: Could not determine model type for {log_file}")

        print(f"  Detected model type: {model_type}")

        # Extract tree ID from filename
        tree_id_match = re.search(r'(?:tree|final_tree)\.(\d+)\.', log_path.name)
        tree_id = tree_id_match.group(1) if tree_id_match else 'unknown'

        # Parse the log file
        result = parse_balanced_log(log_file)
        result['model'] = model_type
        result['tree_id'] = tree_id
        result['log_file'] = str(log_file)

        # Determine outcome
        if model_type == 'BDSKY':
            result['outcome'] = 'TP' if result['evidence_found'] else 'FN'
        elif model_type == 'BDCT':
            result['outcome'] = 'FP' if result['evidence_found'] else 'TN'
        else:
            result['outcome'] = 'UNKNOWN'

        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        'model', 'tree_id', 'parameter_set', 'evidence_found', 'bonferroni_evidence', 'outcome',
        'internal_T', 'external_T',
        'internal_early_count', 'internal_late_count', 'external_early_count', 'external_late_count',
        'internal_u_stat', 'external_u_stat', 'internal_pval', 'external_pval',
        'internal_effect_size', 'external_effect_size',
        'internal_significant', 'external_significant',
        'internal_strong_evidence', 'external_strong_evidence',
        'num_tips', 'log_file'
    ]

    # Reorder and save
    df = df.reindex(columns=[col for col in column_order if col in df.columns])
    df.to_csv(args.tab, sep='\t', index=False, na_rep='NA')

    print(f"Combined {len(results)} results into {args.tab}")
    print(f"Parameter sets: {sorted(df['parameter_set'].unique())}")
    print(f"Model types: {df['model'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()