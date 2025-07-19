import logging
import numpy as np
import scipy.stats
from ete3 import Tree
from typing import List, Tuple, Optional
import argparse
import sys


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


DEFAULT_MANN_WHITNEY_MIN_SAMPLES = 20
TIME = 'time'


def annotate_tree_with_time(tree):
    """
    Annotates tree nodes with their time from the root.

    :param tree: ete3.Tree, the tree to annotate
    """
    tree.add_features(time=0.0)

    for node in tree.traverse("preorder"):
        if not node.is_root():
            node.add_features(time=node.up.time + node.dist)


def remove_certain_leaves(tr: Tree, to_remove=lambda node: False) -> Optional[Tree]:
    """
    Removes all the branches leading to leaves identified positively by to_remove function.
    :param tr: the tree of interest (ete3 Tree)
    :param to_remove: a method to check is a leaf should be removed.
    :return: ete3.Tree: the pruned tree, or None if the root is removed.
    """

    tips_to_remove = [tip for tip in tr.get_leaves() if to_remove(tip)]

    for node in tips_to_remove:
        if node.is_root():
            return None

        parent = node.up
        parent.remove_child(node)


        if len(parent.children) == 1 and not parent.is_root():
            child_to_merge = parent.children[0]
            child_to_merge.dist += parent.dist

            grandparent = parent.up
            if grandparent:
                grandparent.remove_child(parent)
                grandparent.add_child(child_to_merge)
            else:
                child_to_merge.up = None
                tr = child_to_merge
        elif not parent.children and not parent.is_root():
            pass

    return tr


def extract_branches_in_interval(tree, t_start, t_end):
    """
    Extract branch lengths that fall completely within a time interval.
    This function might be less central in the new strategy but kept for completeness
    or potential future use.

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


        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist


        if branch_start_time >= t_start and branch_end_time <= t_end:
            if node.is_leaf():
                external_branches.append(node.dist)
            else:
                internal_branches.append(node.dist)

    return internal_branches, external_branches


def find_largest_subtree_in_interval(tree, t_start, t_end):
    """
    Find the largest subtree (by total nodes) that falls completely within
    a time interval and extract ALL its internal and external branch lengths.
    The "root" of this subtree must be an internal node whose incoming branch
    starts *after* the t_start of the given interval.

    :param tree: ete3.Tree, the tree of interest
    :param t_start: float, start time of interval
    :param t_end: float, end time of interval
    :return: tuple of (internal_branches, external_branches) from largest subtree
    """

    def subtree_falls_in_interval(node, t_start_check, t_end_check):
        """Check if entire subtree rooted at node falls within interval."""

        node_time = getattr(node, TIME)
        if node_time < t_start_check or node_time > t_end_check:
            return False


        if not node.is_root() and node.dist is not None:
            branch_start_time = getattr(node.up, TIME)
            if branch_start_time < t_start_check or node_time > t_end_check:
                return False


        for descendant in node.traverse():
            if descendant == node:
                continue

            desc_time = getattr(descendant, TIME)

            if desc_time < t_start_check or desc_time > t_end_check:
                return False


            if descendant.dist is not None:
                desc_branch_start_time = getattr(descendant.up, TIME)
                if desc_branch_start_time < t_start_check or desc_time > t_end_check:
                    return False
        return True

    def count_nodes_in_subtree(node):
        """Count total nodes (internal + leaves) in a subtree."""
        return len(list(node.traverse()))

    def extract_all_branches_from_subtree(node):
        """Extract all internal and external branches from a given subtree."""
        internal = []
        external = []
        for descendant in node.traverse():
            if descendant == node:
                continue
            if descendant.dist is not None:
                if descendant.is_leaf():
                    external.append(descendant.dist)
                else:
                    internal.append(descendant.dist)
        return internal, external


    max_nodes_in_subtree = -1
    best_root = None

    for node in tree.traverse():

        if node.is_root() or node.is_leaf() or node.dist is None:
            continue

        branch_start_time = getattr(node.up, TIME)
        node_time = getattr(node, TIME)


        if branch_start_time >= t_start and node_time <= t_end:
            if subtree_falls_in_interval(node, t_start, t_end):
                current_nodes_in_subtree = count_nodes_in_subtree(node)

                if current_nodes_in_subtree > max_nodes_in_subtree:
                    max_nodes_in_subtree = current_nodes_in_subtree
                    best_root = node

    if best_root is None:
        logging.warning(f"No valid subtrees found fully within interval [{t_start:.4f}, {t_end:.4f}] "
                        f"with root's incoming branch starting after {t_start:.4f}.")
        return [], []
    else:
        logging.info(
            f"Selected largest subtree rooted at time {getattr(best_root, TIME):.4f} "
            f"(incoming branch starts at {getattr(best_root.up, TIME):.4f}) "
            f"with {max_nodes_in_subtree} total nodes in interval [{t_start:.4f}, {t_end:.4f}]")
        return extract_all_branches_from_subtree(best_root)


def find_time_for_n_tips(tree, n_tips_threshold):
    """
    Finds the time T from the root needed to accumulate n_tips_threshold tips.

    :param tree: ete3.Tree, the tree of interest (must have 'time' annotated)
    :param n_tips_threshold: int, the number of tips to reach
    :return: float, time T from root, or None if not enough tips found
    """
    tip_times = []
    for node in tree.traverse():
        if node.is_leaf():
            if hasattr(node, TIME):
                tip_times.append(getattr(node, TIME))
            else:
                logging.warning(
                    f"Tip {node.name} does not have a 'time' attribute. Ensure annotate_tree_with_time was run.")
                return None

    if len(tip_times) < n_tips_threshold:
        logging.warning(f"Not enough tips ({len(tip_times)}) to find time for {n_tips_threshold} tips.")
        return None

    tip_times.sort()

    return tip_times[n_tips_threshold - 1]


def count_tips_in_subtree(node):
    """
    Count the number of tips (leaves) in the subtree rooted at the given node.

    :param node: ete3.Tree node
    :return: int, number of tips in subtree
    """
    return len(node.get_leaves())


def find_nodes_with_min_subtree_tips(tree, min_tips):
    """
    Find all internal nodes that have at least min_tips in their subtree.

    :param tree: ete3.Tree, the tree of interest
    :param min_tips: int, minimum number of tips required in subtree
    :return: list of nodes that qualify
    """
    qualifying_nodes = []

    for node in tree.traverse():
        if not node.is_leaf():
            tips_in_subtree = count_tips_in_subtree(node)
            if tips_in_subtree >= min_tips:
                qualifying_nodes.append(node)

    return qualifying_nodes


def calculate_robust_T(tree, n_tips_fraction_denominator: int = 4):
    """
    Calculate a robust T value using the proposed method:
    T = max(T_top, T_bottom, T_fallback)

    Where:
    - T_top: time to accumulate N/4 tips from root (current method)
    - T_bottom: tree_height - max_time_of_nodes_with_N/4_tips_in_subtree
    - T_fallback: tree_height / 2

    :param tree: ete3.Tree, the tree of interest (must have 'time' annotated)
    :param n_tips_fraction_denominator: int, denominator for fraction of tips (4 for N/4, 2 for N/2)
    :return: tuple of (T, calculation_details)
    """
    tree_height = max(getattr(node, TIME) for node in tree.traverse())
    total_tips = len(tree.get_leaves())
    n_tips_for_calculation = total_tips // n_tips_fraction_denominator

    calculation_details = {
        'tree_height': tree_height,
        'total_tips': total_tips,
        'n_tips_for_calculation': n_tips_for_calculation,
        'n_tips_fraction_denominator': n_tips_fraction_denominator
    }


    T_top = find_time_for_n_tips(tree, n_tips_for_calculation)
    calculation_details['T_top'] = T_top


    T_bottom = None
    if n_tips_for_calculation > 0:
        qualifying_nodes = find_nodes_with_min_subtree_tips(tree, n_tips_for_calculation)
        if qualifying_nodes:

            max_time_qualifying_node = max(getattr(node, TIME) for node in qualifying_nodes)
            T_bottom = tree_height - max_time_qualifying_node
            calculation_details['max_time_qualifying_node'] = max_time_qualifying_node
            calculation_details['qualifying_nodes_count'] = len(qualifying_nodes)
        else:
            logging.warning(f"No internal nodes found with >= {n_tips_for_calculation} tips in subtree.")

    calculation_details['T_bottom'] = T_bottom


    T_fallback = tree_height / 2
    calculation_details['T_fallback'] = T_fallback


    valid_T_values = [T for T in [T_top, T_bottom, T_fallback] if T is not None and T > 0]

    if not valid_T_values:
        logging.error("No valid T values could be calculated.")
        return None, calculation_details

    T_robust = max(valid_T_values)
    calculation_details['T_robust'] = T_robust


    if T_robust == T_top:
        calculation_details['dominant_criterion'] = 'T_top (tip accumulation)'
    elif T_robust == T_bottom:
        calculation_details['dominant_criterion'] = 'T_bottom (bottom structure)'
    elif T_robust == T_fallback:
        calculation_details['dominant_criterion'] = 'T_fallback (midpoint)'
    else:
        calculation_details['dominant_criterion'] = 'unknown'

    logging.info(f"Robust T calculation: T_top={T_top:.4f}, T_bottom={T_bottom}, T_fallback={T_fallback:.4f}")
    logging.info(f"Selected T={T_robust:.4f} (dominated by {calculation_details['dominant_criterion']})")

    return T_robust, calculation_details


def sky_test_new_strategy(tree, min_mann_whitney_samples=DEFAULT_MANN_WHITNEY_MIN_SAMPLES,
                          n_tips_fraction_denominator: int = 4):
    """
    New strategy BD-Skyline test based on tip accumulation and subtree comparison with robust T selection.

    :param tree: ete3.Tree, the tree of interest
    :param min_mann_whitney_samples: int, minimum samples required for Mann-Whitney U test
    :param n_tips_fraction_denominator: int, denominator for fraction of tips to use (e.g., 4 for N/4, 2 for N/2)
    :return: tuple of (evidence_found, test_results, split_times)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())
    total_tips = len(tree.get_leaves())

    logging.info(
        f'Testing tree with height {tree_height:.4f} and {total_tips} tips using robust T selection with N/{n_tips_fraction_denominator} tips criterion.')

    if tree_height == 0:
        logging.warning("Tree height is zero, cannot perform SKY test.")
        return False, None, {}

    results = {}
    split_times = {}

    T, calculation_details = calculate_robust_T(tree, n_tips_fraction_denominator)

    split_times.update(calculation_details)
    split_times['T_from_robust_calculation'] = T

    if T is None or T >= tree_height or T <= 0:
        logging.warning(f"Could not determine a valid robust T. T={T}")
        results['internal'] = results['external'] = None
        return False, results, split_times

    logging.info(f"Determined robust T = {T:.4f} ({calculation_details['dominant_criterion']}).")

    late_interval_start = tree_height - T
    if late_interval_start < 0:
        late_interval_start = 0

    logging.info(
        f"Analyzing late interval for largest subtree (root's incoming branch starts AFTER {late_interval_start:.4f}): [{late_interval_start:.4f}, {tree_height:.4f}] on original tree.")
    late_internal_branches, late_external_branches = find_largest_subtree_in_interval(tree, late_interval_start,
                                                                                      tree_height)


    logging.info(f"Pruning tree at time T={T:.4f} for early interval analysis using remove_certain_leaves.")


    pruned_tree_for_early_analysis = tree.copy("deepcopy")
    annotate_tree_with_time(pruned_tree_for_early_analysis)


    pruned_tree_for_early_analysis = remove_certain_leaves(pruned_tree_for_early_analysis,
                                                           lambda tip: getattr(tip, TIME) > T)


    if pruned_tree_for_early_analysis is None or not pruned_tree_for_early_analysis.get_leaves():
        logging.warning(
            f"Tree became empty after attempting to prune with remove_certain_leaves at time {T:.4f}. Cannot perform early interval analysis.")
        results['internal'] = results['external'] = None
        return False, results, split_times


    logging.info(
        f"Analyzing early interval for largest subtree (root's incoming branch starts AFTER 0): [0, {T:.4f}] on MODIFIED tree.")
    early_internal_branches, early_external_branches = find_largest_subtree_in_interval(pruned_tree_for_early_analysis,
                                                                                        0, T)


    alpha = 0.05


    if len(early_internal_branches) >= min_mann_whitney_samples and len(
            late_internal_branches) >= min_mann_whitney_samples:
        u_result_internal = scipy.stats.mannwhitneyu(early_internal_branches, late_internal_branches,
                                                     alternative='two-sided')
        results['internal'] = {
            'T': T,
            'T_calculation_method': 'robust',
            'dominant_criterion': calculation_details.get('dominant_criterion', 'unknown'),
            'early_interval': (0, T),
            'late_interval': (late_interval_start, tree_height),
            'early_branches': early_internal_branches,
            'late_branches': late_internal_branches,
            'early_count': len(early_internal_branches),
            'late_count': len(late_internal_branches),
            'u_statistic': u_result_internal.statistic,
            'p_value': u_result_internal.pvalue,
            'method': 'new_strategy_robust'
        }
        logging.info(f"Internal branches - T={T:.4f} ({calculation_details['dominant_criterion']})")
        logging.info(f"  Early subtree (0-{T:.4f}): {len(early_internal_branches)} branches")
        logging.info(
            f"  Late subtree ({late_interval_start:.4f}-{tree_height:.4f}): {len(late_internal_branches)} branches")
        logging.info(
            f"  Mann-Whitney U statistic: {u_result_internal.statistic:.4f}, p-value: {u_result_internal.pvalue:.6f}")
    else:
        logging.warning(
            f"Insufficient internal branches for comparison (early: {len(early_internal_branches)}, late: {len(late_internal_branches)}) (Needed: {min_mann_whitney_samples})")
        results['internal'] = None


    if len(early_external_branches) >= min_mann_whitney_samples and len(
            late_external_branches) >= min_mann_whitney_samples:
        u_result_external = scipy.stats.mannwhitneyu(early_external_branches, late_external_branches,
                                                     alternative='two-sided')
        results['external'] = {
            'T': T,
            'T_calculation_method': 'robust',
            'dominant_criterion': calculation_details.get('dominant_criterion', 'unknown'),
            'early_interval': (0, T),
            'late_interval': (late_interval_start, tree_height),
            'early_branches': early_external_branches,
            'late_branches': late_external_branches,
            'early_count': len(early_external_branches),
            'late_count': len(late_external_branches),
            'u_statistic': u_result_external.statistic,
            'p_value': u_result_external.pvalue,
            'method': 'new_strategy_robust'
        }
        logging.info(f"External branches - T={T:.4f} ({calculation_details['dominant_criterion']})")
        logging.info(f"  Early subtree (0-{T:.4f}): {len(early_external_branches)} branches")
        logging.info(
            f"  Late subtree ({late_interval_start:.4f}-{tree_height:.4f}): {len(late_external_branches)} branches")
        logging.info(
            f"  Mann-Whitney U statistic: {u_result_external.statistic:.4f}, p-value: {u_result_external.pvalue:.6f}")
    else:
        logging.warning(
            f"Insufficient external branches for comparison (early: {len(early_external_branches)}, late: {len(late_external_branches)}) (Needed: {min_mann_whitney_samples})")
        results['external'] = None


    evidence_found = False
    significant_tests = []

    for branch_type in ['internal', 'external']:
        if results[branch_type] is not None:
            if results[branch_type]['p_value'] < alpha:
                evidence_found = True
                significant_tests.append(branch_type)

    logging.info(f'Evidence found (α={alpha}): {evidence_found} ({significant_tests})')

    any_test_run = results['internal'] is not None or results['external'] is not None
    if not any_test_run:
        logging.error(
            "No tests could be performed due to insufficient branch data for the chosen strategy. Cannot conclude SKY test.")
        return False, None, split_times

    return evidence_found, results, split_times


def plot_early_vs_late_results(tree, results, outfile=None):
    """
    Plot branch length distributions for early vs late intervals.
    Requires matplotlib and seaborn installed.

    :param tree: ete3.Tree, the tree of interest
    :param results: dict, results from sky_test_new_strategy
    :param outfile: str, optional output file for plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.error(
            "matplotlib and seaborn required for plotting. Please install them using 'pip install matplotlib seaborn'.")
        return

    valid_results = [k for k, v in results.items() if v is not None]
    if not valid_results:
        logging.warning("No valid results to plot.")
        return

    n_plots = len(valid_results)
    if n_plots == 0:
        logging.warning("No valid results to plot.")
        return

    fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 8))

    if n_plots == 1:
        axes = axes.reshape(2, 1) if n_plots > 0 else np.empty((2, 0))

    colors = ['skyblue', 'lightcoral']

    for i, branch_type in enumerate(valid_results):
        result = results[branch_type]

        if result['early_branches']:
            sns.histplot(result['early_branches'], bins=15, kde=True, alpha=0.7,
                         color=colors[0], edgecolor='black', ax=axes[0, i])
        else:
            axes[0, i].text(0.5, 0.5, 'No data', horizontalalignment='center',
                            verticalalignment='center', transform=axes[0, i].transAxes)
        axes[0, i].set_title(f'{branch_type.capitalize()} Branches - Early Subtree\n'
                             f'[{result["early_interval"][0]:.2f}, {result["early_interval"][1]:.2f}]\n'
                             f'({result["early_count"]} branches)')
        axes[0, i].set_xlabel('Branch Length')
        axes[0, i].set_ylabel('Frequency')

        if result['late_branches']:
            sns.histplot(result['late_branches'], bins=15, kde=True, alpha=0.7,
                         color=colors[1], edgecolor='black', ax=axes[1, i])
        else:
            axes[1, i].text(0.5, 0.5, 'No data', horizontalalignment='center',
                            verticalalignment='center', transform=axes[1, i].transAxes)

        axes[1, i].set_title(f'{branch_type.capitalize()} Branches - Late Subtree\n'
                             f'[{result["late_interval"][0]:.2f}, {result["late_interval"][1]:.2f}]\n'
                             f'({result["late_count"]} branches)')
        axes[1, i].set_xlabel('Branch Length')
        axes[1, i].set_ylabel('Frequency')

        method_text = f'T method: {result.get("dominant_criterion", "unknown")}\n'
        axes[1, i].text(0.05, 0.95,
                        f'{method_text}U stat: {result["u_statistic"]:.4f}\np-value: {result["p_value"]:.6f}',
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
Enhanced BD-Skyline test for Birth-Death Skyline models with robust T selection.

This strategy uses a robust T calculation method that takes the maximum of:
1. T_top: Time to accumulate N/X tips from root
2. T_bottom: Tree height minus maximum time of nodes with N/X tips in their subtree  
3. T_fallback: Tree height divided by 2

This ensures adequate power in both intervals and prevents extreme interval imbalances.
""")

    parser.add_argument('--nwk', required=True, type=str,
                        help="Input tree file in Newick format")
    parser.add_argument('--log', type=str, help="Output log file")
    parser.add_argument('--plot', type=str, help="Output plot file")
    parser.add_argument('--min-mw-samples', type=int, default=DEFAULT_MANN_WHITNEY_MIN_SAMPLES,
                        help=f"Minimum number of branches required in each sample for Mann-Whitney U test. (default: {DEFAULT_MANN_WHITNEY_MIN_SAMPLES})")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")

    args = parser.parse_args()


    level = logging.INFO if args.verbose else logging.WARNING

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    try:

        tree = Tree(args.nwk, format=1)
        total_tips = len(tree.get_leaves())

        print(f"Total tips in tree: {total_tips}")

        logging.info("Attempting BD-Skyline test with N/4 tips criterion using robust T selection...")
        evidence_found, results, split_times = sky_test_new_strategy(
            tree.copy("deepcopy"), args.min_mw_samples, n_tips_fraction_denominator=4
        )

        if results is None:
            logging.warning(
                "N/4 tip analysis failed. Retrying BD-Skyline test with N/2 tips criterion using robust T selection...")
            tree_for_rerun = Tree(args.nwk, format=1)
            evidence_found, results, split_times = sky_test_new_strategy(
                tree_for_rerun, args.min_mw_samples, n_tips_fraction_denominator=2
            )
            if results is not None:
                logging.info("N/2 tip analysis succeeded.")
            else:
                logging.error("N/2 tip analysis also failed. No results available.")

        print("\n" + "=" * 50)
        print("ENHANCED BD-SKYLINE TEST RESULTS (ROBUST T)")
        print("=" * 50)

        tree_height = 0.0
        annotate_tree_with_time(tree)
        if hasattr(tree.get_tree_root(), TIME):
            tree_height = max(getattr(node, TIME) for node in tree.traverse() if hasattr(node, TIME))

        if 'T_from_robust_calculation' in split_times and split_times['T_from_robust_calculation'] is not None:
            split_time_T = split_times['T_from_robust_calculation']
            n_tips_for_T_display = split_times.get('n_tips_for_calculation', 'Unknown')
            used_fraction_denominator = split_times.get('n_tips_fraction_denominator', 'Unknown')
            dominant_criterion = split_times.get('dominant_criterion', 'unknown')

            percentage = (split_time_T / tree_height) * 100 if tree_height > 0 else 0
            print(f"Robust T calculation using N/{used_fraction_denominator} criterion ({n_tips_for_T_display} tips):")
            print(f"  T_top (tip accumulation): {split_times.get('T_top', 'N/A'):.6f}")
            print(f"  T_bottom (bottom structure): {split_times.get('T_bottom', 'N/A')}")
            print(f"  T_fallback (midpoint): {split_times.get('T_fallback', 'N/A'):.6f}")
            print(f"  Selected T: {split_time_T:.6f} ({percentage:.1f}% of tree height)")
            print(f"  Dominant criterion: {dominant_criterion}")
        else:
            print("Robust T calculation: Not available (insufficient tips or calculation error)")

        print(f"Tree height: {tree_height:.6f}")
        print("=" * 50)

        if evidence_found:
            print("\nENHANCED SKY test: Evidence of BD-Skyline model detected")
        else:
            print("\nENHANCED SKY test: No evidence of BD-Skyline model (consistent with simple BD)")

        if results is not None:
            for branch_type in ['internal', 'external']:
                if results[branch_type] is not None:
                    result = results[branch_type]
                    print(f"\n{branch_type.capitalize()} branches (robust strategy comparison):")
                    print(f"  T used = {result['T']:.4f} ({result.get('dominant_criterion', 'unknown')})")
                    print(
                        f"  Early subtree interval [{result['early_interval'][0]:.4f}, {result['early_interval'][1]:.4f}]: {result['early_count']} branches")
                    print(
                        f"  Late subtree interval [{result['late_interval'][0]:.4f}, {result['late_interval'][1]:.4f}]: {result['late_count']} branches")
                    print(f"  Mann-Whitney U statistic: {result['u_statistic']:.4f}")
                    print(f"  p-value: {result['p_value']:.6f}")
                else:
                    print(f"\n{branch_type.capitalize()} branches: Not enough data for comparison.")
        else:
            print(
                "\nNo detailed branch results available due to insufficient data or analysis errors from any attempt.")

        if args.plot:
            if results is not None:
                plot_early_vs_late_results(tree, results, args.plot)
            else:
                logging.warning("Skipping plot generation as no valid results were obtained from any test attempt.")

        if args.log:
            with open(args.log, 'w') as f:
                f.write('Enhanced BD-Skyline test results - Robust T Selection\n')
                f.write('====================================================\n')
                f.write(f'Total tips in tree: {total_tips}\n')
                f.write(f'Tree height: {tree_height:.6f}\n')
                f.write('\nROBUST T CALCULATION:\n')
                f.write('--------------------\n')
                if 'T_from_robust_calculation' in split_times and split_times['T_from_robust_calculation'] is not None:
                    split_time_T = split_times['T_from_robust_calculation']
                    n_tips_for_T_display = split_times.get('n_tips_for_calculation', 'Unknown')
                    used_fraction_denominator = split_times.get('n_tips_fraction_denominator', 'Unknown')
                    dominant_criterion = split_times.get('dominant_criterion', 'unknown')

                    percentage = (split_time_T / tree_height) * 100 if tree_height > 0 else 0
                    f.write(f"Using N/{used_fraction_denominator} criterion ({n_tips_for_T_display} tips):\n")
                    f.write(f"  T_top (tip accumulation): {split_times.get('T_top', 'N/A'):.6f}\n")
                    f.write(f"  T_bottom (bottom structure): {split_times.get('T_bottom', 'N/A')}\n")
                    f.write(f"  T_fallback (midpoint): {split_times.get('T_fallback', 'N/A'):.6f}\n")
                    f.write(f"  Selected T: {split_time_T:.6f} ({percentage:.1f}% of tree height)\n")
                    f.write(f"  Dominant criterion: {dominant_criterion}\n")
                else:
                    f.write("Robust T calculation: Not available (insufficient tips or calculation error)\n")

                f.write(f'\nEvidence of skyline model: {"Yes" if evidence_found else "No"}\n')

                if results is not None:
                    for branch_type in ['internal', 'external']:
                        if results[branch_type] is not None:
                            result = results[branch_type]
                            early_start = result["early_interval"][0]
                            early_end = result["early_interval"][1]
                            late_start = result["late_interval"][0]
                            late_end = result["late_interval"][1]

                            f.write(f'\n{branch_type.capitalize()} branches (robust strategy):\n')
                            f.write(f'  T used = {result["T"]:.6f} ({result.get("dominant_criterion", "unknown")})\n')
                            f.write(
                                f'  Early subtree interval: [{early_start:.6f}, {early_end:.6f}] ({result["early_count"]} branches)\n')
                            f.write(
                                f'  Late subtree interval: [{late_start:.6f}, {late_end:.6f}] ({result["late_count"]} branches)\n')
                            f.write(f'  Mann-Whitney U statistic: {result["u_statistic"]:.6f}\n')
                            f.write(f'  p-value: {result["p_value"]:.6f}\n')
                        else:
                            f.write(f'\n{branch_type.capitalize()} branches: Not enough data for comparison.\n')
                else:
                    f.write(
                        "\nNo detailed branch results could be generated due to insufficient data or analysis errors from any attempt.\n")

    except Exception as e:
        logging.error(f"Error running enhanced BD-Skyline test: {e}", exc_info=True)
        sys.exit(1)

    return 0


if __name__ == '__main__':
    exit(main())