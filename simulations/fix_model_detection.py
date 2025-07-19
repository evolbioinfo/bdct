#!/usr/bin/env python3
"""
Fix model detection in the balanced BDSKY results.
"""

import pandas as pd
import sys
from pathlib import Path
import re


def fix_model_detection(input_file, output_file):
    """Fix the model detection based on file paths."""

    # Load the data
    df = pd.read_csv(input_file, sep='\t')

    print(f"Original data: {len(df)} rows")
    print(f"Original model distribution: {df['model'].value_counts().to_dict()}")

    # Fix model detection
    def detect_model_from_path(log_file_path):
        log_path = Path(log_file_path)

        # Method 1: Check directory structure
        path_str = str(log_path).lower()
        if 'simulations_bdsky' in path_str or '/bdsky/' in path_str:
            return 'BDSKY'
        elif 'bdct' in path_str or '/bdct0/' in path_str:
            return 'BDCT'

        # Method 2: Check filename patterns
        filename = log_path.name
        if 'final_tree' in filename:
            return 'BDSKY'  # Usually BDSKY uses final_tree
        elif filename.startswith('tree.') and '.balanced_bdsky_test.' in filename:
            return 'BDCT'  # Usually BDCT uses tree.X

        # Method 3: Check parent directories
        parts = log_path.parts
        for part in parts:
            if 'bdsky' in part.lower():
                return 'BDSKY'
            elif 'bdct' in part.lower():
                return 'BDCT'

        return 'UNKNOWN'

    # Apply the fixed detection
    df['model_fixed'] = df['log_file'].apply(detect_model_from_path)

    print(f"Fixed model distribution: {df['model_fixed'].value_counts().to_dict()}")

    # Show some examples
    print("\nSample file paths and detected models:")
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        print(f"  {row['log_file']} → {row['model_fixed']}")

    # Update the model column
    df['model'] = df['model_fixed']
    df = df.drop('model_fixed', axis=1)

    # Fix outcomes based on corrected model
    def fix_outcome(row):
        if row['model'] == 'BDSKY':
            return 'TP' if row['evidence_found'] else 'FN'
        elif row['model'] == 'BDCT':
            return 'FP' if row['evidence_found'] else 'TN'
        else:
            return 'UNKNOWN'

    df['outcome'] = df.apply(fix_outcome, axis=1)

    # Save the corrected data
    df.to_csv(output_file, sep='\t', index=False)

    print(f"\nCorrected data saved to: {output_file}")
    print(f"Final model distribution: {df['model'].value_counts().to_dict()}")
    print(f"Final outcome distribution: {df['outcome'].value_counts().to_dict()}")

    return df


def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_model_detection.py <input_file.tab> <output_file.tab>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    fix_model_detection(input_file, output_file)


if __name__ == '__main__':
    main()