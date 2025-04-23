import logging
import re
import os
import sys
import pandas as pd
import numpy as np
import glob


def parse_tree_log(filename):
    """Parse parameters from a tree log file."""
    try:
        tree_num = int(re.findall(r'[0-9]+', filename)[-1])

        with open(filename, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            logging.warning(f"Not enough lines in {filename}, skipping")
            return None, None

        # Parse header
        header = [col.strip() for col in lines[0].strip().split(',')]

        # Parse all rows
        models_data = {}
        for row_idx, line in enumerate(lines[1:], 1):
            if not line.strip():
                continue

            values = [val.strip() for val in line.strip().split(',')]
            if len(values) < len(header):
                continue

            row_data = {}
            for col_idx, col in enumerate(header):
                try:
                    if col_idx < len(values) and values[col_idx]:
                        row_data[col] = float(values[col_idx]) if values[col_idx].replace('.', '', 1).isdigit() else \
                            values[col_idx]
                except (ValueError, IndexError):
                    row_data[col] = values[col_idx] if col_idx < len(values) else ""

            # Store data for this model
            models_data[row_idx] = row_data

        # Map real parameters to BDSKY format
        real_params = {
            'tree': tree_num,
            'model': 'real'
        }

        # Add parameters for model 1
        if models_data and 1 in models_data:
            model1 = models_data[1]
            if 'la_ii' in model1:
                real_params['lambda_1'] = model1.get('la_ii', None)
                real_params['psi_1'] = model1.get('psi_i', None)
                real_params['rho_1'] = model1.get('p_i', None)
                real_params['R0_1'] = model1.get('R', None)
                real_params['infectious_time_1'] = model1.get('infectious time', None)
                real_params['tips'] = model1.get('tips', None)
                real_params['t_1'] = model1.get('end_time', None)

        # Add parameters for model 2 if it exists
        if len(models_data) > 1 and 2 in models_data:
            model2 = models_data[2]
            if 'la_ii' in model2:
                real_params['lambda_2'] = model2.get('la_ii', None)
                real_params['psi_2'] = model2.get('psi_i', None)
                real_params['rho_2'] = model2.get('p_i', None)
                real_params['R0_2'] = model2.get('R', None)
                real_params['infectious_time_2'] = model2.get('infectious time', None)
                # Update tips if not set in model 1
                if not real_params.get('tips'):
                    real_params['tips'] = model2.get('tips', None)

        logging.info(f"Parsed tree log {filename}: {real_params}")

        return tree_num, real_params
    except Exception as e:
        logging.error(f"Error parsing tree log {filename}: {e}")
        return None, None


def parse_param_estimation(filename):
    """Parse parameters from a parameter estimation CSV file."""
    try:
        tree_num = int(re.findall(r'[0-9]+', filename)[-1])

        with open(filename, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            logging.warning(f"Not enough lines in {filename}, skipping")
            return None, None

        # First line is header, second line is values
        header = [col.strip() for col in lines[0].strip().split(',')]
        values = [val.strip() for val in lines[1].strip().split(',')]

        est_params = {
            'tree': tree_num,
            'model': 'est'
        }

        # Map parameters
        for i, col in enumerate(header):
            if i < len(values) and values[i]:
                try:
                    est_params[col] = float(values[i])
                except ValueError:
                    est_params[col] = values[i]

        logging.info(f"Parsed parameter estimation {filename}: {est_params}")

        return tree_num, est_params
    except Exception as e:
        logging.error(f"Error parsing parameter estimation {filename}: {e}")
        return None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create comparison table for BDSKY estimations.")
    parser.add_argument('--logs', nargs='*', default=[], type=str,
                        help="Tree log files with real parameters (final_tree.*.log files)")
    parser.add_argument('--estimates', nargs='*', default=[], type=str,
                        help="BDSKY estimated parameters (param_estimation.*.csv files)")
    parser.add_argument('--output', type=str, help="Output comparison table")

    # Add options to specify directory patterns instead of file lists
    parser.add_argument('--logs_pattern', type=str, help="Pattern for tree log files (e.g., ./trees/final_tree.*.log)")
    parser.add_argument('--estimates_pattern', type=str,
                        help="Pattern for estimate files (e.g., ./results/param_estimation.*.csv)")

    params = parser.parse_args()

    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # Use file patterns if provided and lists are empty
    if not params.logs and params.logs_pattern:
        params.logs = glob.glob(params.logs_pattern)
        logging.info(f"Found {len(params.logs)} log files using pattern: {params.logs_pattern}")

    if not params.estimates and params.estimates_pattern:
        params.estimates = glob.glob(params.estimates_pattern)
        logging.info(f"Found {len(params.estimates)} estimate files using pattern: {params.estimates_pattern}")

    # Check if we have input files
    if not params.logs:
        logging.error("No log files provided. Use --logs or --logs_pattern to specify tree log files.")
        sys.exit(1)

    if not params.estimates:
        logging.error(
            "No estimate files provided. Use --estimates or --estimates_pattern to specify BDSKY estimate files.")
        sys.exit(1)

    if not params.output:
        logging.error("No output file specified. Use --output to specify where to save the comparison.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(params.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    # Print summary of files being processed
    logging.info(f"Processing {len(params.logs)} log files and {len(params.estimates)} estimate files")
    logging.info(f"Output will be saved to: {params.output}")

    # Dictionary to store real parameters
    real_params = {}

    # Parse tree log files
    for log_file in params.logs:
        tree_num, tree_data = parse_tree_log(log_file)
        if tree_num is not None and tree_data is not None:
            real_params[tree_num] = tree_data

    # Dictionary to store estimate parameters
    est_params = {}

    # Parse parameter estimation files
    for est_file in params.estimates:
        tree_num, est_data = parse_param_estimation(est_file)
        if tree_num is not None and est_data is not None:
            est_params[tree_num] = est_data

    # Create comparison data
    comparison_data = []

    # Parameters to compare
    param_pairs = [
        ('lambda_1', 'Lambda 1'),
        ('psi_1', 'Psi 1'),
        ('rho_1', 'Rho 1'),
        ('R0_1', 'R0 1'),
        ('infectious_time_1', 'Infectious Time 1'),
        ('lambda_2', 'Lambda 2'),
        ('psi_2', 'Psi 2'),
        ('rho_2', 'Rho 2'),
        ('R0_2', 'R0 2'),
        ('infectious_time_2', 'Infectious Time 2'),
        ('t_1', 'Change Time')
    ]

    # Define error transformation function similar to plot_error.py
    error_or_1 = lambda x: max(min(x, 1), -1)

    # For each tree, compare parameters
    for tree_num in sorted(set(real_params.keys()) & set(est_params.keys())):
        real = real_params[tree_num]
        est = est_params[tree_num]

        row = {'tree': tree_num}

        # For each parameter
        for param, label in param_pairs:
            real_val = real.get(param)
            est_val = est.get(param)

            # Add values
            row[f'real_{param}'] = real_val
            row[f'est_{param}'] = est_val

            # Calculate differences
            if real_val is not None and est_val is not None:
                try:
                    real_val = float(real_val)
                    est_val = float(est_val)

                    # Calculate standard absolute difference
                    abs_diff = est_val - real_val

                    # Calculate relative difference with consistent error calculations
                    if param.startswith('rho_'):
                        # For probability parameters, use absolute difference
                        rel_diff = abs_diff
                    else:
                        # For rate parameters and their inverses, use relative difference
                        rel_diff = abs_diff / real_val
                        # Apply error transformation (cap at ±1 or ±100%)
                        rel_diff = error_or_1(rel_diff)

                    row[f'abs_diff_{param}'] = abs_diff
                    row[f'rel_diff_{param}'] = rel_diff * 100  # Convert to percentage
                except (ValueError, TypeError):
                    pass

        comparison_data.append(row)

    # Convert to DataFrame
    comp_df = pd.DataFrame(comparison_data)

    # Check if we have comparison data
    if comp_df.empty:
        logging.error("No comparison data could be generated. Check that file formats are correct.")
        # Save empty files
        comp_df.to_csv(params.output, index=False)
        summary_df = pd.DataFrame()
        summary_path = os.path.splitext(params.output)[0] + '_summary.csv'
        summary_df.to_csv(summary_path)
        mae_bias_path = os.path.splitext(params.output)[0] + '_mae_bias.csv'
        mae_bias_summary = pd.DataFrame({'Parameter': [label for param, label in param_pairs]})
        mae_bias_summary.to_csv(mae_bias_path, index=False)
        sys.exit(0)

    # Calculate MAE and bias with the same capping as in plot_error.py
    mae_results = {}
    bias_results = {}

    for param, label in param_pairs:
        real_col = f'real_{param}'
        est_col = f'est_{param}'

        # Check if columns exist
        if real_col in comp_df.columns and est_col in comp_df.columns:
            # Get valid rows
            valid_rows = comp_df.dropna(subset=[real_col, est_col])

            if len(valid_rows) > 0:
                # Convert to float if needed
                valid_rows[real_col] = valid_rows[real_col].astype(float)
                valid_rows[est_col] = valid_rows[est_col].astype(float)

                # Calculate using approach from plot_error.py
                mae_sum = 0.0
                bias_sum = 0.0
                count = 0

                for _, row in valid_rows.iterrows():
                    real_value = row[real_col]
                    est_value = row[est_col]

                    if param.startswith('rho_'):
                        # For probability parameters, use absolute difference
                        rel_diff = est_value - real_value
                    else:
                        # For other parameters, use relative difference
                        rel_diff = (est_value - real_value) / real_value

                    # Apply error transformation (cap at ±1 or ±100%)
                    rel_diff = error_or_1(rel_diff)

                    mae_sum += abs(rel_diff)
                    bias_sum += rel_diff
                    count += 1

                # Calculate final values by dividing by count
                if count > 0:
                    mae = mae_sum / count
                    bias = bias_sum / count

                    mae_results[param] = mae
                    bias_results[param] = bias

    # Create summary dataframe
    summary_df = pd.DataFrame()

    # Add MAE and bias
    summary_df['MAE'] = pd.Series(mae_results)
    summary_df['Bias'] = pd.Series(bias_results)

    # Calculate other statistics
    diff_cols = [col for col in comp_df.columns if col.startswith('abs_diff_') or col.startswith('rel_diff_')]

    if diff_cols:
        for stat in ['mean', 'median', 'std', 'min', 'max']:
            if stat == 'mean':
                summary_df[stat] = comp_df[diff_cols].mean()
            elif stat == 'median':
                summary_df[stat] = comp_df[diff_cols].median()
            elif stat == 'std':
                summary_df[stat] = comp_df[diff_cols].std()
            elif stat == 'min':
                summary_df[stat] = comp_df[diff_cols].min()
            elif stat == 'max':
                summary_df[stat] = comp_df[diff_cols].max()

    # Save results
    comp_df.to_csv(params.output, index=False)
    logging.info(f"Saved comparison table to {params.output}")

    summary_path = os.path.splitext(params.output)[0] + '_summary.csv'
    summary_df.to_csv(summary_path)
    logging.info(f"Saved summary statistics to {summary_path}")

    # Create MAE/Bias summary
    mae_bias_summary = pd.DataFrame({
        'Parameter': [label for param, label in param_pairs],
        'MAE': [mae_results.get(param, np.nan) for param, label in param_pairs],
        'Bias': [bias_results.get(param, np.nan) for param, label in param_pairs]
    })

    mae_bias_path = os.path.splitext(params.output)[0] + '_mae_bias.csv'
    mae_bias_summary.to_csv(mae_bias_path, index=False)
    logging.info(f"Saved MAE/Bias summary to {mae_bias_path}")