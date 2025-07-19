#!/usr/bin/env python3
"""
Re-parse the balanced BDSKY log files with correct parsing patterns.
"""

import pandas as pd
import sys
from pathlib import Path
import re


def parse_balanced_log_fixed(log_file):
    """Parse a balanced skyline test log file with correct patterns."""

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

        print(f"Parsing: {log_file}")

        # Extract basic results - FIXED patterns
        if 'Evidence of skyline model (Bonferroni): Yes' in content:
            result['evidence_found'] = True
            result['bonferroni_evidence'] = True
            print(f"  Found Bonferroni evidence")
        elif re.search(r'Evidence of skyline model \(α=[\d.]+\): Yes', content):
            result['evidence_found'] = True
            result['bonferroni_evidence'] = False
            print(f"  Found regular evidence")
        elif 'Evidence of BD-Skyline model detected (Bonferroni corrected)' in content:
            result['evidence_found'] = True
            result['bonferroni_evidence'] = True
            print(f"  Found old-style Bonferroni evidence")
        elif 'Evidence of BD-Skyline model detected' in content:
            result['evidence_found'] = True
            result['bonferroni_evidence'] = False
            print(f"  Found old-style evidence")
        else:
            print(f"  No evidence found")

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
        internal_section = re.search(r'Internal branches:(.*?)(?=External branches|$)', content, re.DOTALL)
        if internal_section:
            internal_text = internal_section.group(1)

            # Extract T value
            t_match = re.search(r'T = ([\d.]+)', internal_text)
            if t_match:
                result['internal_T'] = float(t_match.group(1))

            # Extract interval counts - look for pattern like "(25 branches)"
            early_match = re.search(r'Early interval.*?\((\d+) branches\)', internal_text)
            late_match = re.search(r'Late interval.*?\((\d+) branches\)', internal_text)
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
        external_section = re.search(r'External branches:(.*?)(?=\n\n|$)', content, re.DOTALL)
        if external_section:
            external_text = external_section.group(1)

            # Extract T value
            t_match = re.search(r'T = ([\d.]+)', external_text)
            if t_match:
                result['external_T'] = float(t_match.group(1))

            # Extract interval counts
            early_match = re.search(r'Early interval.*?\((\d+) branches\)', external_text)
            late_match = re.search(r'Late interval.*?\((\d+) branches\)', external_text)
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


def reparse_all_logs(results_file, output_file):
    """Re-parse all log files and update the results."""

    # Load existing results to get file paths
    df = pd.read_csv(results_file, sep='\t')

    print(f"Re-parsing {len(df)} log files...")

    new_results = []

    for _, row in df.iterrows():
        log_file = row['log_file']

        if not Path(log_file).exists():
            print(f"Warning: Log file not found: {log_file}")
            continue

        # Re-parse the log file
        parsed = parse_balanced_log_fixed(log_file)

        # Keep original metadata
        result = {
            'model': row['model'],
            'tree_id': row['tree_id'],
            'log_file': log_file
        }

        # Add parsed data
        result.update(parsed)

        # Determine outcome
        if result['model'] == 'BDSKY':
            result['outcome'] = 'TP' if result['evidence_found'] else 'FN'
        elif result['model'] == 'BDCT':
            result['outcome'] = 'FP' if result['evidence_found'] else 'TN'
        else:
            result['outcome'] = 'UNKNOWN'

        new_results.append(result)

    # Create new DataFrame
    new_df = pd.DataFrame(new_results)

    # Reorder columns
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

    new_df = new_df.reindex(columns=[col for col in column_order if col in new_df.columns])

    # Save results
    new_df.to_csv(output_file, sep='\t', index=False, na_rep='NA')

    # Print summary
    print(f"\nRe-parsing complete!")
    print(f"Evidence found distribution: {new_df['evidence_found'].value_counts().to_dict()}")
    print(f"Outcome distribution: {new_df['outcome'].value_counts().to_dict()}")
    print(f"Results saved to: {output_file}")

    return new_df


def main():
    if len(sys.argv) != 3:
        print("Usage: python reparse_balanced_logs.py <input_results.tab> <output_results.tab>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    reparse_all_logs(input_file, output_file)


if __name__ == '__main__':
    main()