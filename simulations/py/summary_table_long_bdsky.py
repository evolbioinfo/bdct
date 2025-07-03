import logging
import numpy as np
import scipy.stats
from ete3 import Tree
from typing import List, Tuple, Optional
import argparse
import matplotlib.pyplot as plt

DEFAULT_MIN_BRANCHES = 10
TIME = 'time'


def annotate_tree_with_time(tree):
    """
    Annotates tree nodes with their time from the root.
    """
    tree.add_features(time=0.0)
    for node in tree.traverse("preorder"):
        if not node.is_root():
            node.add_features(time=node.up.time + node.dist)


def get_tree_diagnostics(tree):
    """
    Get diagnostic information about the tree structure.
    """
    annotate_tree_with_time(tree)

    internal_branches = []
    external_branches = []
    all_branch_times = []

    for node in tree.traverse():
        if node.is_root() or node.dist is None:
            continue

        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist

        if node.is_leaf():
            external_branches.append({
                'length': node.dist,
                'start_time': branch_start_time,
                'end_time': branch_end_time
            })
        else:
            internal_branches.append({
                'length': node.dist,
                'start_time': branch_start_time,
                'end_time': branch_end_time
            })

        all_branch_times.append(branch_end_time)

    tree_height = max(all_branch_times)

    print(f"Tree diagnostics:")
    print(f"  Tree height: {tree_height:.4f}")
    print(f"  Total internal branches: {len(internal_branches)}")
    print(f"  Total external branches: {len(external_branches)}")
    print(f"  Total tips: {len(tree.get_leaves())}")

    return {
        'tree_height': tree_height,
        'internal_branches': internal_branches,
        'external_branches': external_branches,
        'n_internal': len(internal_branches),
        'n_external': len(external_branches)
    }


def plot_branch_distribution_over_time(tree, diagnostics, outfile=None):
    """
    Plot branch distributions over time to visualize potential issues.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Plot branch end times
    internal_end_times = [b['end_time'] for b in diagnostics['internal_branches']]
    external_end_times = [b['end_time'] for b in diagnostics['external_branches']]

    ax1.hist(internal_end_times, bins=20, alpha=0.7, label='Internal', color='blue')
    ax1.hist(external_end_times, bins=20, alpha=0.7, label='External', color='red')
    ax1.set_xlabel('Branch End Time')
    ax1.set_ylabel('Count')
    ax1.set_title('Branch End Time Distribution')
    ax1.legend()

    # Plot branch lengths over time
    internal_lengths = [b['length'] for b in diagnostics['internal_branches']]
    external_lengths = [b['length'] for b in diagnostics['external_branches']]

    ax2.scatter(internal_end_times, internal_lengths, alpha=0.6, label='Internal', color='blue', s=20)
    ax2.scatter(external_end_times, external_lengths, alpha=0.6, label='External', color='red', s=20)
    ax2.set_xlabel('Branch End Time')
    ax2.set_ylabel('Branch Length')
    ax2.set_title('Branch Length vs End Time')
    ax2.legend()

    # Box plots of branch lengths
    ax3.boxplot([internal_lengths, external_lengths], labels=['Internal', 'External'])
    ax3.set_ylabel('Branch Length')
    ax3.set_title('Branch Length Distributions')

    # Cumulative branch counts
    internal_times_sorted = sorted(internal_end_times)
    external_times_sorted = sorted(external_end_times)

    ax4.plot(internal_times_sorted, range(1, len(internal_times_sorted) + 1), label='Internal', color='blue')
    ax4.plot(external_times_sorted, range(1, len(external_times_sorted) + 1), label='External', color='red')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative Branch Count')
    ax4.set_title('Cumulative Branch Counts Over Time')
    ax4.legend()

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plot saved to {outfile}")
    else:
        plt.show()


def find_time_for_n_branches_improved(tree, n_branches, branch_type='internal'):
    """
    Improved version that provides more diagnostics.
    """
    branches_with_times = []

    for node in tree.traverse():
        if node.is_root() or node.dist is None:
            continue

        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist

        if branch_type == 'internal' and not node.is_leaf():
            branches_with_times.append((branch_end_time, branch_start_time, node.dist))
        elif branch_type == 'external' and node.is_leaf():
            branches_with_times.append((branch_end_time, branch_start_time, node.dist))

    if len(branches_with_times) < n_branches:
        print(f"Warning: Only {len(branches_with_times)} {branch_type} branches available, need {n_branches}")
        return None, branches_with_times

    branches_with_times.sort(key=lambda x: x[0])  # Sort by end time

    # Find the nth branch end time
    nth_branch_end_time = branches_with_times[n_branches - 1][0]

    print(f"Time for {n_branches} {branch_type} branches: {nth_branch_end_time:.4f}")

    return nth_branch_end_time, branches_with_times


def extract_branches_in_interval_improved(tree, t_start, t_end, verbose=False):
    """
    Improved version with better diagnostics.
    """
    internal_branches = []
    external_branches = []

    for node in tree.traverse():
        if node.is_root() or node.dist is None:
            continue

        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist

        # Check if branch falls completely within interval
        if branch_start_time >= t_start and branch_end_time <= t_end:
            if node.is_leaf():
                external_branches.append(node.dist)
            else:
                internal_branches.append(node.dist)

    if verbose:
        print(f"  Interval [{t_start:.4f}, {t_end:.4f}]:")
        print(f"    Internal branches: {len(internal_branches)}")
        print(f"    External branches: {len(external_branches)}")
        if len(internal_branches) > 0:
            print(f"    Internal lengths: mean={np.mean(internal_branches):.4f}, std={np.std(internal_branches):.4f}")
        if len(external_branches) > 0:
            print(f"    External lengths: mean={np.mean(external_branches):.4f}, std={np.std(external_branches):.4f}")

    return internal_branches, external_branches


def alternative_sky_test(tree, n_branches=DEFAULT_MIN_BRANCHES):
    """
    Alternative implementation with better interval selection.
    Instead of using symmetric intervals, use non-overlapping intervals.
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    print(f"\nAlternative SKY test - Non-overlapping intervals")
    print(f"Tree height: {tree_height:.4f}")

    results = {}

    for branch_type in ['internal', 'external']:
        print(f"\nTesting {branch_type} branches...")

        # Get all branches of this type with their times
        all_branches = []
        for node in tree.traverse():
            if node.is_root() or node.dist is None:
                continue

            branch_end_time = getattr(node, TIME)
            branch_start_time = branch_end_time - node.dist

            if (branch_type == 'internal' and not node.is_leaf()) or \
                    (branch_type == 'external' and node.is_leaf()):
                all_branches.append({
                    'length': node.dist,
                    'start_time': branch_start_time,
                    'end_time': branch_end_time,
                    'mid_time': (branch_start_time + branch_end_time) / 2
                })

        if len(all_branches) < 2 * n_branches:
            print(f"  Not enough {branch_type} branches ({len(all_branches)} < {2 * n_branches})")
            results[branch_type] = None
            continue

        # Sort branches by their midpoint time
        all_branches.sort(key=lambda x: x['mid_time'])

        # Take first n_branches and last n_branches based on midpoint time
        early_branches = [b['length'] for b in all_branches[:n_branches]]
        late_branches = [b['length'] for b in all_branches[-n_branches:]]

        # Perform statistical test
        u_result = scipy.stats.mannwhitneyu(early_branches, late_branches, alternative='two-sided')

        # Also perform Kolmogorov-Smirnov test
        ks_result = scipy.stats.ks_2samp(early_branches, late_branches)

        results[branch_type] = {
            'early_branches': early_branches,
            'late_branches': late_branches,
            'early_midtime_range': (all_branches[0]['mid_time'], all_branches[n_branches - 1]['mid_time']),
            'late_midtime_range': (all_branches[-n_branches]['mid_time'], all_branches[-1]['mid_time']),
            'mann_whitney_u': u_result.statistic,
            'mann_whitney_p': u_result.pvalue,
            'ks_statistic': ks_result.statistic,
            'ks_p_value': ks_result.pvalue
        }

        print(
            f"  Early {n_branches} branches (midtime {results[branch_type]['early_midtime_range'][0]:.3f}-{results[branch_type]['early_midtime_range'][1]:.3f})")
        print(f"    Mean length: {np.mean(early_branches):.4f} ± {np.std(early_branches):.4f}")
        print(
            f"  Late {n_branches} branches (midtime {results[branch_type]['late_midtime_range'][0]:.3f}-{results[branch_type]['late_midtime_range'][1]:.3f})")
        print(f"    Mean length: {np.mean(late_branches):.4f} ± {np.std(late_branches):.4f}")
        print(f"  Mann-Whitney U: {u_result.statistic:.4f}, p = {u_result.pvalue:.6f}")
        print(f"  Kolmogorov-Smirnov: {ks_result.statistic:.4f}, p = {ks_result.pvalue:.6f}")

    return results


def comprehensive_sky_test(tree, n_branches=DEFAULT_MIN_BRANCHES):
    """
    Run both original and alternative tests with comprehensive diagnostics.
    """
    print("=" * 60)
    print("COMPREHENSIVE SKY TEST ANALYSIS")
    print("=" * 60)

    # Get tree diagnostics
    diagnostics = get_tree_diagnostics(tree)

    # Run original test with verbose output
    print("\n" + "=" * 40)
    print("ORIGINAL SKY TEST")
    print("=" * 40)

    annotate_tree_with_time(tree)
    tree_height = diagnostics['tree_height']

    original_results = {}

    for branch_type in ['internal', 'external']:
        print(f"\n--- {branch_type.upper()} BRANCHES ---")

        T, all_branches = find_time_for_n_branches_improved(tree, n_branches, branch_type)

        if T is None:
            original_results[branch_type] = None
            continue

        if T >= tree_height:
            print(f"T ({T:.4f}) >= tree_height ({tree_height:.4f}), skipping")
            original_results[branch_type] = None
            continue

        # Extract branches from intervals
        print(f"Early interval: [0, {T:.4f}]")
        early_branches, _ = extract_branches_in_interval_improved(tree, 0, T, verbose=True)

        late_start = tree_height - T
        if late_start < 0:
            late_start = 0
        print(f"Late interval: [{late_start:.4f}, {tree_height:.4f}]")

        if branch_type == 'internal':
            late_branches, _ = extract_branches_in_interval_improved(tree, late_start, tree_height, verbose=True)
        else:
            _, late_branches = extract_branches_in_interval_improved(tree, late_start, tree_height, verbose=True)

        if len(early_branches) < n_branches or len(late_branches) < n_branches:
            print(f"Insufficient branches: early={len(early_branches)}, late={len(late_branches)}")
            original_results[branch_type] = None
            continue

        # Statistical tests
        u_result = scipy.stats.mannwhitneyu(early_branches, late_branches, alternative='two-sided')

        original_results[branch_type] = {
            'T': T,
            'early_branches': early_branches,
            'late_branches': late_branches,
            'u_statistic': u_result.statistic,
            'p_value': u_result.pvalue
        }

        print(f"Mann-Whitney U test: statistic={u_result.statistic:.4f}, p={u_result.pvalue:.6f}")

    # Run alternative test
    print("\n" + "=" * 40)
    print("ALTERNATIVE SKY TEST")
    print("=" * 40)

    alternative_results = alternative_sky_test(tree, n_branches)

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)

    alpha = 0.05
    bonferroni_alpha = alpha / 2

    print(f"Significance levels: α = {alpha}, Bonferroni α = {bonferroni_alpha:.3f}")

    for branch_type in ['internal', 'external']:
        print(f"\n{branch_type.upper()} BRANCHES:")

        # Original test results
        if original_results[branch_type] is not None:
            p_val = original_results[branch_type]['p_value']
            significant = "YES" if p_val < alpha else "NO"
            bonf_significant = "YES" if p_val < bonferroni_alpha else "NO"
            print(f"  Original test: p = {p_val:.6f}, significant = {significant}, Bonferroni = {bonf_significant}")
        else:
            print(f"  Original test: FAILED")

        # Alternative test results
        if alternative_results[branch_type] is not None:
            mw_p = alternative_results[branch_type]['mann_whitney_p']
            ks_p = alternative_results[branch_type]['ks_p_value']
            mw_sig = "YES" if mw_p < alpha else "NO"
            ks_sig = "YES" if ks_p < alpha else "NO"
            print(f"  Alternative test (MW): p = {mw_p:.6f}, significant = {mw_sig}")
            print(f"  Alternative test (KS): p = {ks_p:.6f}, significant = {ks_sig}")
        else:
            print(f"  Alternative test: FAILED")

    return original_results, alternative_results, diagnostics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive SKY test diagnostics")
    parser.add_argument('--nwk', required=True, help="Input tree file in Newick format")
    parser.add_argument('--plot', help="Output diagnostic plot file")
    parser.add_argument('--min-branches', type=int, default=DEFAULT_MIN_BRANCHES,
                        help=f"Minimum branches (default: {DEFAULT_MIN_BRANCHES})")

    args = parser.parse_args()

    try:
        tree = Tree(args.nwk, format=1)
        original_results, alternative_results, diagnostics = comprehensive_sky_test(tree, args.min_branches)

        if args.plot:
            plot_branch_distribution_over_time(tree, diagnostics, args.plot)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)