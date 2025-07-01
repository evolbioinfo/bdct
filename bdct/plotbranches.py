import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ete3 import Tree
import numpy as np
import argparse


def annotate_tree_with_time(tree):
    """
    Annotates tree nodes with their time from the root.
    """
    tree.add_features(time=0.0)
    for node in tree.traverse("preorder"):
        if not node.is_root():
            node.add_features(time=node.up.time + node.dist)


def get_branch_data_with_intervals(tree_file_path, internal_T=None, external_T=None):
    """
    Reads a Newick tree file and extracts branch data with interval classifications.
    """
    try:
        tree = Tree(tree_file_path, format=1)
    except Exception as e:
        print(f"Error reading tree file {tree_file_path}: {e}")
        return [], [], 0

    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, 'time', 0) for node in tree.traverse())

    internal_branches_data = []
    external_branches_data = []

    for node in tree.traverse("postorder"):
        if node.is_root():
            continue

        branch_length = node.dist
        branch_end_time = getattr(node, 'time', None)
        branch_start_time = branch_end_time - branch_length

        if branch_start_time is None or branch_end_time is None:
            continue

        branch_info = {
            'length': branch_length,
            'start_time': branch_start_time,
            'end_time': branch_end_time,
            'interval_type': 'outside'
        }

        if node.is_leaf():
            branch_info['type'] = 'external'

            if external_T is not None:
                # Early interval: [0, external_T]
                early_qualify = branch_start_time >= 0 and branch_end_time <= external_T

                # Late interval: [tree_height - external_T, tree_height]
                late_start = tree_height - external_T
                late_qualify = branch_start_time >= late_start and branch_end_time <= tree_height

                # Check for overlap
                if late_start < external_T:  # Intervals overlap
                    overlap_start = late_start
                    overlap_end = external_T
                    in_overlap = branch_start_time >= overlap_start and branch_end_time <= overlap_end

                    if in_overlap:
                        branch_info['interval_type'] = 'overlap'
                    elif early_qualify:
                        branch_info['interval_type'] = 'early'
                    elif late_qualify:
                        branch_info['interval_type'] = 'late'
                else:
                    if early_qualify:
                        branch_info['interval_type'] = 'early'
                    elif late_qualify:
                        branch_info['interval_type'] = 'late'

            external_branches_data.append(branch_info)
        else:
            branch_info['type'] = 'internal'

            if internal_T is not None:
                # Early interval: [0, internal_T]
                early_qualify = branch_start_time >= 0 and branch_end_time <= internal_T

                # Late interval: [tree_height - internal_T, tree_height]
                late_start = tree_height - internal_T
                late_qualify = branch_start_time >= late_start and branch_end_time <= tree_height

                # Check for overlap
                if late_start < internal_T:  # Intervals overlap
                    overlap_start = late_start
                    overlap_end = internal_T
                    in_overlap = branch_start_time >= overlap_start and branch_end_time <= overlap_end

                    if in_overlap:
                        branch_info['interval_type'] = 'overlap'
                    elif early_qualify:
                        branch_info['interval_type'] = 'early'
                    elif late_qualify:
                        branch_info['interval_type'] = 'late'
                else:
                    if early_qualify:
                        branch_info['interval_type'] = 'early'
                    elif late_qualify:
                        branch_info['interval_type'] = 'late'

            internal_branches_data.append(branch_info)

    return internal_branches_data, external_branches_data, tree_height


def plot_bdsky_intervals(tree_file_path, internal_T=None, external_T=None, output_file=None):
    """
    Generates a scatter plot showing BD-Skyline test intervals.
    """
    internal_data, external_data, tree_height = get_branch_data_with_intervals(
        tree_file_path, internal_T, external_T)

    if not internal_data and not external_data:
        print(f"No branch data found in {tree_file_path}. Plotting skipped.")
        return

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Colors for different interval types
    colors = {
        'early': '#FF6B6B',  # Red for early interval only
        'late': '#4ECDC4',  # Teal for late interval only
        'overlap': '#9B59B6',  # Purple for overlap (counted in both)
        'outside': '#95A5A6'  # Gray for outside intervals
    }

    # Plot internal branches
    if internal_data:
        df_internal = pd.DataFrame(internal_data)

        for interval_type, color in colors.items():
            subset = df_internal[df_internal['interval_type'] == interval_type]
            if not subset.empty:
                ax1.scatter(subset['end_time'], subset['length'],
                            c=color, alpha=0.7, s=20, label=f'{interval_type.capitalize()}')

        ax1.set_title('Internal Branches - BD-Skyline Test Intervals')
        ax1.set_xlabel('Branch End Time')
        ax1.set_ylabel('Branch Length')
        ax1.grid(True, alpha=0.3)

        # Add vertical lines and shaded regions for internal branches
        if internal_T is not None:
            # Early interval
            ax1.axvspan(0, internal_T, alpha=0.2, color=colors['early'])
            ax1.axvline(x=internal_T, color=colors['early'], linestyle='--', linewidth=2)

            # Late interval
            late_start = tree_height - internal_T
            ax1.axvspan(late_start, tree_height, alpha=0.2, color=colors['late'])
            ax1.axvline(x=late_start, color=colors['late'], linestyle='--', linewidth=2)

            # Count branches (overlap counts in both)
            early_count = sum(df_internal['interval_type'] == 'early') + sum(df_internal['interval_type'] == 'overlap')
            late_count = sum(df_internal['interval_type'] == 'late') + sum(df_internal['interval_type'] == 'overlap')
            overlap_count = sum(df_internal['interval_type'] == 'overlap')

            # Add text annotations
            ax1.text(internal_T / 2, ax1.get_ylim()[1] * 0.9,
                     f'Early\n{early_count} branches\n(incl. {overlap_count} overlap)',
                     ha='center', va='top', bbox=dict(boxstyle='round', facecolor=colors['early'], alpha=0.7))
            ax1.text((late_start + tree_height) / 2, ax1.get_ylim()[1] * 0.9,
                     f'Late\n{late_count} branches\n(incl. {overlap_count} overlap)',
                     ha='center', va='top', bbox=dict(boxstyle='round', facecolor=colors['late'], alpha=0.7))

        ax1.legend()

    # Plot external branches
    if external_data:
        df_external = pd.DataFrame(external_data)

        for interval_type, color in colors.items():
            subset = df_external[df_external['interval_type'] == interval_type]
            if not subset.empty:
                ax2.scatter(subset['end_time'], subset['length'],
                            c=color, alpha=0.7, s=20, label=f'{interval_type.capitalize()}')

        ax2.set_title('External Branches - BD-Skyline Test Intervals')
        ax2.set_xlabel('Branch End Time')
        ax2.set_ylabel('Branch Length')
        ax2.grid(True, alpha=0.3)

        # Add vertical lines and shaded regions for external branches
        if external_T is not None:
            # Early interval
            ax2.axvspan(0, external_T, alpha=0.2, color=colors['early'])
            ax2.axvline(x=external_T, color=colors['early'], linestyle='--', linewidth=2)

            # Late interval
            late_start = tree_height - external_T
            ax2.axvspan(late_start, tree_height, alpha=0.2, color=colors['late'])
            ax2.axvline(x=late_start, color=colors['late'], linestyle='--', linewidth=2)

            # Count branches (overlap counts in both)
            early_count = sum(df_external['interval_type'] == 'early') + sum(df_external['interval_type'] == 'overlap')
            late_count = sum(df_external['interval_type'] == 'late') + sum(df_external['interval_type'] == 'overlap')
            overlap_count = sum(df_external['interval_type'] == 'overlap')

            # Add text annotations
            ax2.text(external_T / 2, ax2.get_ylim()[1] * 0.9,
                     f'Early\n{early_count} branches\n(incl. {overlap_count} overlap)',
                     ha='center', va='top', bbox=dict(boxstyle='round', facecolor=colors['early'], alpha=0.7))
            ax2.text((late_start + tree_height) / 2, ax2.get_ylim()[1] * 0.9,
                     f'Late\n{late_count} branches\n(incl. {overlap_count} overlap)',
                     ha='center', va='top', bbox=dict(boxstyle='round', facecolor=colors['late'], alpha=0.7))

        ax2.legend()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("BD-SKYLINE INTERVAL ANALYSIS")
    print("=" * 60)
    print(f"Tree height: {tree_height:.6f}")

    if internal_T is not None and internal_data:
        df_internal = pd.DataFrame(internal_data)
        early_only = sum(df_internal['interval_type'] == 'early')
        late_only = sum(df_internal['interval_type'] == 'late')
        overlap = sum(df_internal['interval_type'] == 'overlap')
        outside = sum(df_internal['interval_type'] == 'outside')

        print(f"\nINTERNAL BRANCHES (T = {internal_T:.6f}):")
        print(f"  Early total: {early_only + overlap} branches (early only: {early_only}, overlap: {overlap})")
        print(f"  Late total: {late_only + overlap} branches (late only: {late_only}, overlap: {overlap})")
        print(f"  Outside intervals: {outside} branches")
        print(f"  Total internal branches: {len(internal_data)}")

    if external_T is not None and external_data:
        df_external = pd.DataFrame(external_data)
        early_only = sum(df_external['interval_type'] == 'early')
        late_only = sum(df_external['interval_type'] == 'late')
        overlap = sum(df_external['interval_type'] == 'overlap')
        outside = sum(df_external['interval_type'] == 'outside')

        print(f"\nEXTERNAL BRANCHES (T = {external_T:.6f}):")
        print(f"  Early total: {early_only + overlap} branches (early only: {early_only}, overlap: {overlap})")
        print(f"  Late total: {late_only + overlap} branches (late only: {late_only}, overlap: {overlap})")
        print(f"  Outside intervals: {outside} branches")
        print(f"  Total external branches: {len(external_data)}")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plots BD-Skyline test intervals showing branch distributions."
    )
    parser.add_argument('--nwk', required=True, type=str,
                        help="Path to the input Newick tree file.")
    parser.add_argument('--internal_T', type=float, default=None,
                        help="Split time for internal branches (T_internal)")
    parser.add_argument('--external_T', type=float, default=None,
                        help="Split time for external branches (T_external)")
    parser.add_argument('--output', type=str, default=None,
                        help="Optional path to save the plot.")

    args = parser.parse_args()

    plot_bdsky_intervals(
        tree_file_path=args.nwk,
        internal_T=args.internal_T,
        external_T=args.external_T,
        output_file=args.output
    )