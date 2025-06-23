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
        # Ensure the tip exists before attempting to delete it
        if tip in tree_copy.iter_leaves():
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
    # Validate input parameters as per the image description
    # Ensure t1 < t2 < T, t1 + delta_t < t2, and t2 + delta_t < T
    # Added a small epsilon to comparisons to account for floating point inaccuracies
    epsilon = 1e-9
    if not (t1 < t2 - epsilon and t1 + delta_t < t2 - epsilon and t2 + delta_t < T - epsilon):
        return None

    # Process first interval [t1, t1 + Δt]
    tree1_pruned = prune_tips_after_time(tree, t1 + delta_t)
    # Re-annotate time for the pruned tree as branch lengths might change due to pruning
    annotate_tree_with_time(tree1_pruned)
    internal1, external1 = extract_branches_in_interval(tree1_pruned, t1, t1 + delta_t)

    # Process second interval [t2, t2 + Δt]
    tree2_pruned = prune_tips_after_time(tree, t2 + delta_t)
    # Re-annotate time for the pruned tree
    annotate_tree_with_time(tree2_pruned)
    internal2, external2 = extract_branches_in_interval(tree2_pruned, t2, t2 + delta_t)

    # Check minimum branch requirements for both internal and external branches
    if (len(internal1) < min_branches or len(external1) < min_branches or
            len(internal2) < min_branches or len(external2) < min_branches):
        return None

    # Kolmogorov-Smirnov tests
    # Use 2-sample KS test to compare the distributions
    ks_internal = scipy.stats.ks_2samp(internal1, internal2)
    ks_external = scipy.stats.ks_2samp(external1, external2)

    return {
        'intervals': (t1, t2, delta_t),
        'branch_counts': (len(internal1), len(external1), len(internal2), len(external2)),
        'ks_internal_stat': ks_internal.statistic,
        'ks_external_stat': ks_external.statistic,
        'pval_internal': ks_internal.pvalue,
        'pval_external': ks_external.pvalue
    }


def sky_test(tree, min_branches=DEFAULT_MIN_BRANCHES):
    """
    Tests if the input tree was generated under a BD-Skyline model.

    This version uses a more systematic search for interval pairs and applies
    Benjamini-Hochberg (FDR) correction for multiple testing.

    :param tree: ete3.Tree, the tree of interest
    :param min_branches: int, minimum number of branches required per interval
    :return: tuple of (evidence_found, num_tests, best_result)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    logging.info(f'Testing tree with height {tree_height:.4f}')

    if tree_height == 0:
        logging.warning("Tree height is zero, cannot perform SKY test.")
        return False, 0, None

    all_p_values = []
    all_test_results = []

    # Define parameters for systematic interval search
    # These ratios are relative to the tree_height to make them adaptable to different tree sizes.
    # We choose ranges that are likely to contain meaningful intervals for skyline detection.
    # Aumentado el número de puntos para generar más pares de intervalos.
    t1_ratios = np.linspace(0.0, 0.5, 40)  # Start times for the first interval (e.g., 0% to 50% of tree height)
    delta_t_ratios = np.linspace(0.1, 0.3, 7)  # Durations for each interval (e.g., 10% to 30% of tree height)
    t2_gap_ratios = np.linspace(0.05, 0.3, 40)  # Gap between the two intervals (e.g., 5% to 30% of tree height)

    # Minimum absolute delta_t to ensure intervals are not infinitesimally small,
    # which can lead to unstable statistics or empty branch sets.
    min_delta_t_abs = max(0.001 * tree_height, 0.01)

    for t1_ratio in t1_ratios:
        t1 = t1_ratio * tree_height
        for delta_t_ratio in delta_t_ratios:
            delta_t = max(delta_t_ratio * tree_height, min_delta_t_abs)

            # Ensure the first interval fits within the tree height
            if t1 + delta_t >= tree_height:
                continue

            for t2_gap_ratio in t2_gap_ratios:
                t2_gap = t2_gap_ratio * tree_height
                t2 = t1 + delta_t + t2_gap  # Calculate start time of second interval

                # Ensure the second interval fits within the tree height
                if t2 + delta_t >= tree_height:
                    continue

                result = test_interval_pair(tree, t1, t2, delta_t, tree_height, min_branches)
                if result:
                    all_test_results.append(result)
                    # Collect both internal and external p-values for FDR correction
                    all_p_values.append(result['pval_internal'])
                    all_p_values.append(result['pval_external'])

    num_tests_performed = len(all_test_results)  # This is the count of valid interval PAIRS tested
    num_p_values = len(all_p_values)  # This is the total number of p-values for FDR correction

    logging.info(f'Performed {num_tests_performed} valid interval pair comparisons, yielding {num_p_values} p-values.')

    if num_p_values == 0:
        return False, 0, None

    # Benjamini-Hochberg (FDR) correction to control false discoveries
    # Sort all p-values in ascending order
    sorted_p_values = sorted(all_p_values)
    alpha = 0.05  # Standard significance level for FDR

    # Find the largest k such that P_(k) <= (k/m) * alpha
    # Iterate backwards from the largest p-value to find the highest rank 'k' that satisfies the condition.
    evidence_found = False

    # We will store all p-values that are considered significant after BH correction.
    # The 'best_result' will then be the one among these significant results with the lowest original p-value.
    significant_results_candidates = []

    for k in range(num_p_values):
        bh_threshold = (k + 1) / num_p_values * alpha
        # Check if the current p-value (from sorted list) is less than or equal to its BH threshold
        if sorted_p_values[k] <= bh_threshold:
            # Mark that evidence is found, and then collect all results that meet *this* threshold
            # as any p-value less than or equal to sorted_p_values[k] will also be significant.
            # This logic needs to iterate through ALL original test results and check against the final threshold.
            evidence_found = True

    if evidence_found:
        final_bh_threshold = sorted_p_values[k]  # The largest p-value that was significant

        for result in all_test_results:
            if result['pval_internal'] <= final_bh_threshold or result['pval_external'] <= final_bh_threshold:
                significant_results_candidates.append(result)

        if significant_results_candidates:
            # Find the best result among the significant candidates based on the minimum of internal/external p-values
            best_result = min(significant_results_candidates, key=lambda x: min(x['pval_internal'], x['pval_external']))
        else:
            # This case should ideally not happen if evidence_found is True, but as a fallback
            best_result = None
            evidence_found = False  # Reset if no actual significant results were found

    else:
        best_result = None

    logging.info(f'Evidence found (FDR controlled at alpha={alpha}): {evidence_found}')
    if evidence_found and best_result:
        logging.info(
            f'Best result details: intervals [{best_result["intervals"][0]:.4f}-{best_result["intervals"][0] + best_result["intervals"][2]:.4f}] vs '
            f'[{best_result["intervals"][1]:.4f}-{best_result["intervals"][1] + best_result["intervals"][2]:.4f}]')
        logging.info(
            f'  Internal KS-statistic={best_result["ks_internal_stat"]:.4f}, External KS-statistic={best_result["ks_external_stat"]:.4f}')
        logging.info(
            f'  Internal P-value={best_result["pval_internal"]:.6f}, External P-value={best_result["pval_external"]:.6f}')

    return evidence_found, num_tests_performed, best_result


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
        logging.error(
            "matplotlib and seaborn required for plotting. Please install them using 'pip install matplotlib seaborn'.")
        return

    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    if tree_height == 0:
        logging.warning("Tree height is zero, cannot generate plot.")
        return

    # Divide tree into 4 quarters for visualization
    quarter = tree_height / 4
    intervals = [(i * quarter, (i + 1) * quarter) for i in range(4)]

    # Adjust figsize to give more space for titles and labels if needed
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    colors = sns.color_palette("husl", 4)

    for i, (t_start, t_end) in enumerate(intervals):
        internal, external = extract_branches_in_interval(tree, t_start, t_end)

        # Plot internal branches
        # Check if axes[0, i] is a valid matplotlib axes object before plotting.
        if internal and axes[0, i] is not None:
            axes[0, i].hist(internal, bins=15, alpha=0.7, color=colors[i],
                            edgecolor='black')  # Added edgecolor for clarity
        if axes[0, i] is not None:
            axes[0, i].set_title(f'Internal [{t_start:.2f}, {t_end:.2f}]\n({len(internal)} branches)',
                                 fontsize=10)  # Reduced fontsize
            axes[0, i].set_xlabel('Branch Length', fontsize=9)
            axes[0, i].set_ylabel('Frequency', fontsize=9)
            axes[0, i].tick_params(axis='both', which='major', labelsize=8)  # Reduced tick label size

        # Plot external branches
        if external and axes[1, i] is not None:
            axes[1, i].hist(external, bins=15, alpha=0.7, color=colors[i],
                            edgecolor='black')  # Added edgecolor for clarity
        if axes[1, i] is not None:
            axes[1, i].set_title(f'External [{t_start:.2f}, {t_end:.2f}]\n({len(external)} branches)',
                                 fontsize=10)  # Reduced fontsize
            axes[1, i].set_xlabel('Branch Length', fontsize=9)
            axes[1, i].set_ylabel('Frequency', fontsize=9)
            axes[1, i].tick_params(axis='both', which='major', labelsize=8)  # Reduced tick label size

    plt.tight_layout()  # Ensures elements don't overlap

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
        total_tips = len(tree.get_leaves())  # Get total number of tips

        # Log total tips
        print(f"Total tips in tree: {total_tips}")  # Added this line for easier parsing

        # Run test
        evidence_found, num_tests, best_result = sky_test(tree, args.min_branches)

        # Results
        if evidence_found:
            print("SKY test: Evidence of BD-Skyline model detected")
            if best_result:
                print(
                    f"Best evidence from interval pair: "
                    f"[{best_result['intervals'][0]:.4f}-{best_result['intervals'][0] + best_result['intervals'][2]:.4f}] "
                    f"vs [{best_result['intervals'][1]:.4f}-{best_result['intervals'][1] + best_result['intervals'][2]:.4f}]")
                print(
                    f"  Internal KS-statistic={best_result['ks_internal_stat']:.4f}, External KS-statistic={best_result['ks_external_stat']:.4f}")
                print(
                    f"  Internal P-value={best_result['pval_internal']:.6f}, External P-value={best_result['pval_external']:.6f}")
        else:
            print("SKY test: No evidence of BD-Skyline model (consistent with simple BD)")

        print(f"Number of valid interval pairs tested: {num_tests}")

        # Generate plot if requested
        if args.plot:
            plot_sky_test_results(tree, args.plot)

        # Write log if requested
        if args.log:
            with open(args.log, 'w') as f:
                f.write('SKY test results\n')
                f.write('================\n')
                f.write(f'Total tips in tree: {total_tips}\n')  # Also write to log file
                f.write(f'Evidence of skyline model: {"Yes" if evidence_found else "No"}\n')
                f.write(f'Number of interval pairs tested: {num_tests}\n')
                if best_result:
                    f.write(
                        f'Best evidence interval pair: [{best_result["intervals"][0]:.4f}-{best_result["intervals"][0] + best_result["intervals"][2]:.4f}] vs [{best_result["intervals"][1]:.4f}-{best_result["intervals"][1] + best_result["intervals"][2]:.4f}]\n')
                    f.write(f'Best internal KS statistic: {best_result["ks_internal_stat"]:.6f}\n')
                    f.write(f'Best external KS statistic: {best_result["ks_external_stat"]:.6f}\n')
                    f.write(f'Best internal p-value: {best_result["pval_internal"]:.6f}\n')
                    f.write(f'Best external p-value: {best_result["pval_external"]:.6f}\n')

    except Exception as e:
        logging.error(f"Error running SKY test: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())