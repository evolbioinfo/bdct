import logging
import numpy as np
import scipy.stats
from ete3 import Tree
from typing import List, Tuple, Optional
import argparse

DEFAULT_MIN_BRANCHES = 20
TIME = 'time'


def annotate_tree_with_time(tree):
    """
    Annotates tree nodes with their time from the root.

    :param tree: ete3.Tree, the tree to annotate
    """
    # Set root time to 0
    tree.add_features(time=0.0)

    # Calculate times for all nodes
    for node in tree.traverse("preorder"):
        if not node.is_root():
            node.add_features(time=node.up.time + node.dist)


def extract_branches_in_interval(tree, t_start, t_end):
    """
    Extract branch lengths that fall completely within a time interval.

    :param tree: ete3.Tree, the tree of interest
    :param t_start: float, start time of interval
    :param t_end: float, end time of interval
    :return: tuple of (internal_branches, external_branches)
    """
    internal_branches = []
    external_branches = []

    for node in tree.traverse():
        if node.is_root() or node.dist is None:
            continue

        # Calculate branch start and end times
        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist

        # Check if branch falls completely within interval
        if branch_start_time >= t_start and branch_end_time <= t_end:
            if node.is_leaf():
                external_branches.append(node.dist)
            else:
                internal_branches.append(node.dist)

    return internal_branches, external_branches


def find_largest_subtree_in_interval(tree, t_start, t_end, branch_type='internal'):
    """
    Find the largest subtree that falls completely within a time interval.

    :param tree: ete3.Tree, the tree of interest
    :param t_start: float, start time of interval
    :param t_end: float, end time of interval
    :param branch_type: str, either 'internal' or 'external'
    :return: tuple of (internal_branches, external_branches) from largest subtree
    """

    def subtree_falls_in_interval(node, t_start, t_end):
        """Check if entire subtree rooted at node falls within interval"""
        # Check if node itself is within interval
        node_time = getattr(node, TIME)
        if node_time < t_start or node_time > t_end:
            return False

        # Check all descendants
        for descendant in node.traverse():
            if descendant == node:
                continue
            desc_time = getattr(descendant, TIME)
            desc_start_time = desc_time - descendant.dist if descendant.dist else desc_time

            if desc_start_time < t_start or desc_time > t_end:
                return False

        return True

    def count_branches_in_subtree(node, branch_type):
        """Count branches of specified type in subtree"""
        count = 0
        for descendant in node.traverse():
            if descendant == node:
                continue  # Skip the root node of subtree
            if branch_type == 'internal' and not descendant.is_leaf():
                count += 1
            elif branch_type == 'external' and descendant.is_leaf():
                count += 1
        return count

    def extract_branches_from_subtree(node):
        """Extract all branches from subtree"""
        internal_branches = []
        external_branches = []

        for descendant in node.traverse():
            if descendant == node:
                continue  # Skip the root node of subtree
            if descendant.is_leaf():
                external_branches.append(descendant.dist)
            else:
                internal_branches.append(descendant.dist)

        return internal_branches, external_branches

    # Find all potential subtree roots (nodes within the interval)
    candidate_roots = []
    for node in tree.traverse():
        if node.is_root():
            continue
        node_time = getattr(node, TIME)
        if t_start <= node_time <= t_end:
            if subtree_falls_in_interval(node, t_start, t_end):
                branch_count = count_branches_in_subtree(node, branch_type)
                if branch_count > 0:  # Only consider subtrees with relevant branches
                    candidate_roots.append((node, branch_count))

    if not candidate_roots:
        logging.warning(f"No valid subtrees found in interval [{t_start:.4f}, {t_end:.4f}]")
        return [], []

    # Find the subtree with the most branches of the specified type
    best_root = max(candidate_roots, key=lambda x: x[1])[0]

    logging.info(
        f"Selected subtree rooted at time {getattr(best_root, TIME):.4f} with {max(candidate_roots, key=lambda x: x[1])[1]} {branch_type} branches")

    return extract_branches_from_subtree(best_root)


def find_time_for_n_branches(tree, n_branches, branch_type='internal'):
    """
    Find the time T from root needed to accumulate n complete branches of specified type.

    :param tree: ete3.Tree, the tree of interest
    :param n_branches: int, number of branches to accumulate
    :param branch_type: str, either 'internal' or 'external'
    :return: float, time T from root, or None if not enough branches found
    """
    # Collect all branches with their end times
    branches_with_times = []

    for node in tree.traverse():
        if node.is_root() or node.dist is None:
            continue

        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist

        # Check branch type
        if branch_type == 'internal' and not node.is_leaf():
            branches_with_times.append((branch_end_time, branch_start_time, node.dist))
        elif branch_type == 'external' and node.is_leaf():
            branches_with_times.append((branch_end_time, branch_start_time, node.dist))

    if len(branches_with_times) < n_branches:
        return None

    # Sort by end time to process branches chronologically
    branches_with_times.sort(key=lambda x: x[0])

    # Find the time when we have n complete branches
    complete_branches = 0
    current_time = 0.0

    for branch_end_time, branch_start_time, branch_length in branches_with_times:
        # Check if this branch would be complete by its end time
        if branch_start_time >= 0:  # Branch starts at or after root
            complete_branches += 1
            if complete_branches >= n_branches:
                return branch_end_time

    return None


def sky_test_early_vs_late_balanced(tree, n_branches=DEFAULT_MIN_BRANCHES):
    """
    Balanced BD-Skyline test that compares early interval vs largest subtree from late interval.

    :param tree: ete3.Tree, the tree of interest
    :param n_branches: int, minimum number of branches to define interval size
    :return: tuple of (evidence_found, test_results, bonferroni_evidence, split_times)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    logging.info(f'Testing tree with height {tree_height:.4f}')

    if tree_height == 0:
        logging.warning("Tree height is zero, cannot perform SKY test.")
        return False, None, False, {}

    results = {}
    split_times = {}

    # Test internal branches
    logging.info("Testing internal branches...")
    T_internal = find_time_for_n_branches(tree, n_branches, 'internal')
    split_times['internal'] = T_internal

    if T_internal is None:
        logging.warning(f"Not enough internal branches found (need {n_branches})")
        results['internal'] = None
    elif T_internal >= tree_height:
        logging.warning(
            f"Time for {n_branches} internal branches ({T_internal:.4f}) exceeds tree height ({tree_height:.4f})")
        results['internal'] = None
    else:
        # Extract branches from early interval [0, T_internal] (standard method)
        early_internal, _ = extract_branches_in_interval(tree, 0, T_internal)

        # Extract branches from largest subtree in late interval [tree_height - T_internal, tree_height]
        late_start = tree_height - T_internal
        if late_start < 0:
            late_start = 0

        logging.info(f"Extracting largest subtree from late interval [{late_start:.4f}, {tree_height:.4f}]")
        late_internal, _ = find_largest_subtree_in_interval(tree, late_start, tree_height, 'internal')

        if len(early_internal) >= n_branches and len(late_internal) >= n_branches:
            # Perform Mann-Whitney U test
            u_result = scipy.stats.mannwhitneyu(early_internal, late_internal, alternative='two-sided')

            results['internal'] = {
                'T': T_internal,
                'early_interval': (0, T_internal),
                'late_interval': (late_start, tree_height),
                'early_branches': early_internal,
                'late_branches': late_internal,
                'early_count': len(early_internal),
                'late_count': len(late_internal),
                'u_statistic': u_result.statistic,
                'p_value': u_result.pvalue,
                'method': 'balanced_subtree'
            }

            logging.info(f"Internal branches - T={T_internal:.4f}")
            logging.info(f"  Early interval [0, {T_internal:.4f}]: {len(early_internal)} branches (standard)")
            logging.info(
                f"  Late interval [{late_start:.4f}, {tree_height:.4f}]: {len(late_internal)} branches (largest subtree)")
            logging.info(f"  Mann-Whitney U statistic: {u_result.statistic:.4f}, p-value: {u_result.pvalue:.6f}")
        else:
            logging.warning(
                f"Insufficient internal branches in intervals (early: {len(early_internal)}, late: {len(late_internal)})")
            results['internal'] = None

    # Test external branches
    logging.info("Testing external branches...")
    T_external = find_time_for_n_branches(tree, n_branches, 'external')
    split_times['external'] = T_external

    if T_external is None:
        logging.warning(f"Not enough external branches found (need {n_branches})")
        results['external'] = None
    elif T_external >= tree_height:
        logging.warning(
            f"Time for {n_branches} external branches ({T_external:.4f}) exceeds tree height ({tree_height:.4f})")
        results['external'] = None
    else:
        # Extract branches from early interval [0, T_external] (standard method)
        _, early_external = extract_branches_in_interval(tree, 0, T_external)

        # Extract branches from largest subtree in late interval [tree_height - T_external, tree_height]
        late_start = tree_height - T_external
        if late_start < 0:
            late_start = 0

        logging.info(f"Extracting largest subtree from late interval [{late_start:.4f}, {tree_height:.4f}]")
        _, late_external = find_largest_subtree_in_interval(tree, late_start, tree_height, 'external')

        if len(early_external) >= n_branches and len(late_external) >= n_branches:
            # Perform Mann-Whitney U test
            u_result = scipy.stats.mannwhitneyu(early_external, late_external, alternative='two-sided')

            results['external'] = {
                'T': T_external,
                'early_interval': (0, T_external),
                'late_interval': (late_start, tree_height),
                'early_branches': early_external,
                'late_branches': late_external,
                'early_count': len(early_external),
                'late_count': len(late_external),
                'u_statistic': u_result.statistic,
                'p_value': u_result.pvalue,
                'method': 'balanced_subtree'
            }

            logging.info(f"External branches - T={T_external:.4f}")
            logging.info(f"  Early interval [0, {T_external:.4f}]: {len(early_external)} branches (standard)")
            logging.info(
                f"  Late interval [{late_start:.4f}, {tree_height:.4f}]: {len(late_external)} branches (largest subtree)")
            logging.info(f"  Mann-Whitney U statistic: {u_result.statistic:.4f}, p-value: {u_result.pvalue:.6f}")
        else:
            logging.warning(
                f"Insufficient external branches in intervals (early: {len(early_external)}, late: {len(late_external)})")
            results['external'] = None

    # Determine if evidence of skyline model is found
    alpha = 0.05
    evidence_found = False
    significant_tests = []

    for branch_type in ['internal', 'external']:
        if results[branch_type] is not None:
            if results[branch_type]['p_value'] < alpha:
                evidence_found = True
                significant_tests.append(branch_type)

    # Apply Bonferroni correction for multiple testing (we do 2 tests)
    bonferroni_alpha = alpha / 2
    bonferroni_evidence = False
    bonferroni_significant = []

    for branch_type in ['internal', 'external']:
        if results[branch_type] is not None:
            if results[branch_type]['p_value'] < bonferroni_alpha:
                bonferroni_evidence = True
                bonferroni_significant.append(branch_type)

    logging.info(f'Evidence found (α={alpha}): {evidence_found} ({significant_tests})')
    logging.info(
        f'Evidence found (Bonferroni α={bonferroni_alpha:.3f}): {bonferroni_evidence} ({bonferroni_significant})')

    return evidence_found, results, bonferroni_evidence, split_times


def plot_early_vs_late_results(tree, results, outfile=None):
    """
    Plot branch length distributions for early vs late intervals.
    Requires matplotlib and seaborn installed.

    :param tree: ete3.Tree, the tree of interest
    :param results: dict, results from sky_test_early_vs_late_balanced
    :param outfile: str, optional output file for plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.error(
            "matplotlib and seaborn required for plotting. Please install them using 'pip install matplotlib seaborn'.")
        return

    # Count valid results
    valid_results = [k for k, v in results.items() if v is not None]
    if not valid_results:
        logging.warning("No valid results to plot.")
        return

    n_plots = len(valid_results)
    fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 8))

    if n_plots == 1:
        axes = axes.reshape(2, 1)

    colors = ['skyblue', 'lightcoral']

    for i, branch_type in enumerate(valid_results):
        result = results[branch_type]

        # Plot early interval
        axes[0, i].hist(result['early_branches'], bins=15, alpha=0.7,
                        color=colors[0], edgecolor='black', label='Early (Standard)')
        axes[0, i].set_title(f'{branch_type.capitalize()} Branches - Early Interval\n'
                             f'[{result["early_interval"][0]:.2f}, {result["early_interval"][1]:.2f}]\n'
                             f'({result["early_count"]} branches)')
        axes[0, i].set_xlabel('Branch Length')
        axes[0, i].set_ylabel('Frequency')

        # Plot late interval
        axes[1, i].hist(result['late_branches'], bins=15, alpha=0.7,
                        color=colors[1], edgecolor='black', label='Late (Largest Subtree)')
        axes[1, i].set_title(f'{branch_type.capitalize()} Branches - Late Interval\n'
                             f'[{result["late_interval"][0]:.2f}, {result["late_interval"][1]:.2f}]\n'
                             f'({result["late_count"]} branches from largest subtree)')
        axes[1, i].set_xlabel('Branch Length')
        axes[1, i].set_ylabel('Frequency')

        # Add Mann-Whitney U test results as text
        axes[1, i].text(0.05, 0.95, f'U stat: {result["u_statistic"]:.4f}\np-value: {result["p_value"]:.6f}',
                        transform=axes[1, i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if outfile:
        try:
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to {outfile}")
        except Exception as e:
            logging.error(f"Error saving plot to {outfile}: {e}")
    else:
        plt.show()


def main():
    """
    Entry point for balanced BD-Skyline test with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="""
Balanced BD-Skyline test for Birth-Death Skyline models.

Tests if the input tree was generated under a Birth-Death Skyline model
by comparing early interval (standard method) vs largest subtree from late interval.
This addresses the phylogenetic imbalance where early intervals contain coherent 
subtrees while late intervals contain scattered branches from different lineages.
""")

    parser.add_argument('--nwk', required=True, type=str,
                        help="Input tree file in Newick format")
    parser.add_argument('--log', type=str, help="Output log file")
    parser.add_argument('--plot', type=str, help="Output plot file")
    parser.add_argument('--min-branches', type=int, default=DEFAULT_MIN_BRANCHES,
                        help=f"Minimum branches to define interval (default: {DEFAULT_MIN_BRANCHES})")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")

    args = parser.parse_args()

    # Set up logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    try:
        # Read tree
        tree = Tree(args.nwk, format=1)
        total_tips = len(tree.get_leaves())

        # Log total tips
        print(f"Total tips in tree: {total_tips}")

        # Run balanced test
        evidence_found, results, bonferroni_evidence, split_times = sky_test_early_vs_late_balanced(tree,
                                                                                                    args.min_branches)

        # Print split times prominently
        print("\n" + "=" * 50)
        print("BALANCED BD-SKYLINE TEST RESULTS")
        print("=" * 50)

        tree_height = max(getattr(node, TIME) for node in tree.traverse() if hasattr(node, TIME))
        print(f"Tree height: {tree_height:.6f}")

        for branch_type in ['internal', 'external']:
            if split_times[branch_type] is not None:
                split_time = split_times[branch_type]
                percentage = (split_time / tree_height) * 100 if tree_height > 0 else 0
                print(f"Split time for {branch_type} branches: {split_time:.6f} ({percentage:.1f}% of tree height)")
            else:
                print(f"Split time for {branch_type} branches: Not available (insufficient branches)")

        print("=" * 50)

        # Results
        if bonferroni_evidence:
            print("\nBalanced SKY test: Evidence of BD-Skyline model detected (Bonferroni corrected)")
        elif evidence_found:
            print("\nBalanced SKY test: Evidence of BD-Skyline model detected (uncorrected)")
        else:
            print("\nBalanced SKY test: No evidence of BD-Skyline model (consistent with simple BD)")

        # Print detailed results
        for branch_type in ['internal', 'external']:
            if results[branch_type] is not None:
                result = results[branch_type]
                print(f"\n{branch_type.capitalize()} branches (balanced comparison):")
                print(f"  T = {result['T']:.4f}")
                print(
                    f"  Early interval [{result['early_interval'][0]:.4f}, {result['early_interval'][1]:.4f}]: {result['early_count']} branches (standard method)")
                print(
                    f"  Late interval [{result['late_interval'][0]:.4f}, {result['late_interval'][1]:.4f}]: {result['late_count']} branches (largest subtree)")
                print(f"  Mann-Whitney U statistic: {result['u_statistic']:.4f}")
                print(f"  p-value: {result['p_value']:.6f}")

        # Generate plot if requested
        if args.plot:
            plot_early_vs_late_results(tree, results, args.plot)

        # Write log if requested
        if args.log:
            with open(args.log, 'w') as f:
                f.write('Balanced BD-Skyline test results - Early vs Largest Subtree comparison\n')
                f.write('========================================================================\n')
                f.write(f'Total tips in tree: {total_tips}\n')
                f.write(f'Tree height: {tree_height:.6f}\n')
                f.write('\nSPLIT TIMES:\n')
                f.write('-----------\n')
                for branch_type in ['internal', 'external']:
                    if split_times[branch_type] is not None:
                        split_time = split_times[branch_type]
                        percentage = (split_time / tree_height) * 100 if tree_height > 0 else 0
                        f.write(
                            f'Split time for {branch_type} branches: {split_time:.6f} ({percentage:.1f}% of tree height)\n')
                    else:
                        f.write(f'Split time for {branch_type} branches: Not available (insufficient branches)\n')

                f.write(f'\nEvidence of skyline model (uncorrected): {"Yes" if evidence_found else "No"}\n')
                f.write(f'Evidence of skyline model (Bonferroni): {"Yes" if bonferroni_evidence else "No"}\n')

                for branch_type in ['internal', 'external']:
                    if results[branch_type] is not None:
                        result = results[branch_type]
                        f.write(f'\n{branch_type.capitalize()} branches (balanced):\n')
                        f.write(f'  T = {result["T"]:.6f}\n')
                        f.write(
                            f'  Early interval: [{result["early_interval"][0]:.6f}, {result["early_interval"][1]:.6f}] ({result["early_count"]} branches, standard)\n')
                        f.write(
                            f'  Late interval: [{result["late_interval"][0]:.6f}, {result["late_interval"][1]:.6f}] ({result["late_count"]} branches, largest subtree)\n')
                        f.write(f'  Mann-Whitney U statistic: {result["u_statistic"]:.6f}\n')
                        f.write(f'  p-value: {result["p_value"]:.6f}\n')

    except Exception as e:
        logging.error(f"Error running balanced BD-Skyline test: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())