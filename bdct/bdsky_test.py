import logging
import numpy as np
import scipy.stats
from ete3 import Tree
from typing import List, Tuple, Optional
import argparse

DEFAULT_MIN_BRANCHES = 10
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


def prune_tips_after_time(tree, cutoff_time):
    """
    Prune all tips that were sampled after the given cutoff time.

    :param tree: ete3.Tree, the tree to prune
    :param cutoff_time: float, time after which to prune tips
    :return: ete3.Tree, pruned tree copy
    """
    tree_copy = tree.copy()

    # Find tips to remove
    tips_to_remove = []
    for leaf in tree_copy:
        if getattr(leaf, TIME) > cutoff_time:
            tips_to_remove.append(leaf)

    # Remove tips
    for tip in tips_to_remove:
        tip.delete()

    return tree_copy


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


def test_interval_pair(tree, t1, t2, delta_t, T, min_branches=DEFAULT_MIN_BRANCHES):
    """
    Test a single pair of intervals for skyline evidence.

    :param tree: ete3.Tree, the tree of interest
    :param t1: float, start time of first interval
    :param t2: float, start time of second interval
    :param delta_t: float, duration of each interval
    :param T: float, total tree time
    :param min_branches: int, minimum number of branches required per interval
    :return: dict with test results or None if insufficient data
    """
    # Validate input parameters
    if not (t1 < t2 < T and t1 + delta_t < t2 and t2 + delta_t < T):
        return None

    # Process first interval [t1, t1 + Δt]
    tree1_pruned = prune_tips_after_time(tree, t1 + delta_t)
    annotate_tree_with_time(tree1_pruned)
    internal1, external1 = extract_branches_in_interval(tree1_pruned, t1, t1 + delta_t)

    # Process second interval [t2, t2 + Δt]
    tree2_pruned = prune_tips_after_time(tree, t2 + delta_t)
    annotate_tree_with_time(tree2_pruned)
    internal2, external2 = extract_branches_in_interval(tree2_pruned, t2, t2 + delta_t)

    # Check minimum branch requirements
    if (len(internal1) < min_branches or len(external1) < min_branches or
            len(internal2) < min_branches or len(external2) < min_branches):
        return None

    # Kolmogorov-Smirnov tests
    ks_internal = scipy.stats.ks_2samp(internal1, internal2)
    ks_external = scipy.stats.ks_2samp(external1, external2)

    return {
        'intervals': (t1, t2, delta_t),
        'branch_counts': (len(internal1), len(external1), len(internal2), len(external2)),
        'ks_internal': ks_internal.statistic,
        'ks_external': ks_external.statistic,
        'pval_internal': ks_internal.pvalue,
        'pval_external': ks_external.pvalue
    }


def sky_test(tree, min_branches=DEFAULT_MIN_BRANCHES):
    """
    Tests if the input tree was generated under a BD-Skyline model.

    The test automatically explores different interval combinations to detect
    evidence of time-varying birth/death rates using Kolmogorov-Smirnov tests.

    :param tree: ete3.Tree, the tree of interest
    :param min_branches: int, minimum number of branches required per interval
    :return: tuple of (evidence_found, num_tests, best_result)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    logging.info(f'Testing tree with height {tree_height:.4f}')

    # Try different interval configurations
    valid_tests = []

    # Strategy 1: Divide tree into equal parts and test adjacent intervals
    for n_divisions in [4, 6, 8]:
        interval_size = tree_height / n_divisions

        for i in range(n_divisions - 1):
            t1 = i * interval_size
            t2 = (i + 1) * interval_size
            delta_t = interval_size * 0.8  # Use 80% of interval size

            result = test_interval_pair(tree, t1, t2, delta_t, tree_height, min_branches)
            if result:
                valid_tests.append(result)

    # Strategy 2: Test early vs late intervals
    quarter = tree_height / 4
    for delta_t_factor in [0.6, 0.8, 1.0]:
        delta_t = quarter * delta_t_factor

        # Early vs middle
        result = test_interval_pair(tree, 0, quarter, delta_t, tree_height, min_branches)
        if result:
            valid_tests.append(result)

        # Early vs late
        result = test_interval_pair(tree, 0, 2 * quarter, delta_t, tree_height, min_branches)
        if result:
            valid_tests.append(result)

        # Middle vs late
        result = test_interval_pair(tree, quarter, 2 * quarter, delta_t, tree_height, min_branches)
        if result:
            valid_tests.append(result)

    num_tests = len(valid_tests)
    logging.info(f'Performed {num_tests} valid interval comparisons')

    if num_tests == 0:
        return False, 0, None

    # Find the test with strongest evidence (minimum p-value)
    min_pval = float('inf')
    best_result = None

    for result in valid_tests:
        min_test_pval = min(result['pval_internal'], result['pval_external'])
        if min_test_pval < min_pval:
            min_pval = min_test_pval
            best_result = result

    # Apply Bonferroni correction for multiple testing
    corrected_pval = min(min_pval * num_tests, 1.0)

    # Evidence of skyline model if corrected p-value < 0.05
    evidence_found = corrected_pval < 0.05

    logging.info(
        f'Best test: intervals [{best_result["intervals"][0]:.3f}, {best_result["intervals"][0] + best_result["intervals"][2]:.3f}] vs '
        f'[{best_result["intervals"][1]:.3f}, {best_result["intervals"][1] + best_result["intervals"][2]:.3f}]')
    logging.info(f'Min p-value: {min_pval:.6f}, Corrected: {corrected_pval:.6f}')

    return evidence_found, num_tests, best_result


def plot_sky_test_results(tree, outfile=None):
    """
    Plot branch length distributions across time for visualization.
    Requires matplotlib and seaborn installed.

    :param tree: ete3.Tree, the tree of interest
    :param outfile: str, optional output file for plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.error("matplotlib and seaborn required for plotting")
        return

    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    # Divide tree into 4 quarters for visualization
    quarter = tree_height / 4
    intervals = [(i * quarter, (i + 1) * quarter) for i in range(4)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = sns.color_palette("husl", 4)

    for i, (t_start, t_end) in enumerate(intervals):
        internal, external = extract_branches_in_interval(tree, t_start, t_end)

        # Plot internal branches
        if internal:
            axes[0, i].hist(internal, bins=15, alpha=0.7, color=colors[i])
        axes[0, i].set_title(f'Internal [{t_start:.2f}, {t_end:.2f}]\n({len(internal)} branches)')
        axes[0, i].set_xlabel('Branch Length')
        axes[0, i].set_ylabel('Frequency')

        # Plot external branches
        if external:
            axes[1, i].hist(external, bins=15, alpha=0.7, color=colors[i])
        axes[1, i].set_title(f'External [{t_start:.2f}, {t_end:.2f}]\n({len(external)} branches)')
        axes[1, i].set_xlabel('Branch Length')
        axes[1, i].set_ylabel('Frequency')

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def main():
    """
    Entry point for SKY test with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="""
SKY test for Birth-Death Skyline models.

Tests if the input tree was generated under a Birth-Death Skyline model 
by automatically searching for time intervals with different branch length 
distributions. Uses Kolmogorov-Smirnov tests to compare distributions.
""")

    parser.add_argument('--nwk', required=True, type=str,
                        help="Input tree file in Newick format")
    parser.add_argument('--log', type=str, help="Output log file")
    parser.add_argument('--plot', type=str, help="Output plot file")
    parser.add_argument('--min-branches', type=int, default=DEFAULT_MIN_BRANCHES,
                        help=f"Minimum branches per interval (default: {DEFAULT_MIN_BRANCHES})")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")

    args = parser.parse_args()

    # Set up logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    try:
        # Read tree
        tree = Tree(args.nwk, format=1)

        # Run test
        evidence_found, num_tests, best_result = sky_test(tree, args.min_branches)

        # Results
        if evidence_found:
            print("SKY test: Evidence of BD-Skyline model detected")
            if best_result:
                print(
                    f"Best evidence: Internal KS={best_result['ks_internal']:.4f}, External KS={best_result['ks_external']:.4f}")
        else:
            print("SKY test: No evidence of BD-Skyline model (consistent with simple BD)")

        print(f"Number of valid interval tests: {num_tests}")

        # Generate plot if requested
        if args.plot:
            plot_sky_test_results(tree, args.plot)

        # Write log if requested
        if args.log:
            with open(args.log, 'w') as f:
                f.write('SKY test results\n')
                f.write('================\n')
                f.write(f'Evidence of skyline model: {"Yes" if evidence_found else "No"}\n')
                f.write(f'Number of tests performed: {num_tests}\n')
                if best_result:
                    f.write(f'Best internal KS statistic: {best_result["ks_internal"]:.6f}\n')
                    f.write(f'Best external KS statistic: {best_result["ks_external"]:.6f}\n')
                    f.write(f'Best internal p-value: {best_result["pval_internal"]:.6f}\n')
                    f.write(f'Best external p-value: {best_result["pval_external"]:.6f}\n')

    except Exception as e:
        logging.error(f"Error running SKY test: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())