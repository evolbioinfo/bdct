import logging
import numpy as np
import scipy.stats
from ete3 import Tree
from typing import List, Tuple, Optional
import argparse

# Configure logging to display messages by default
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DEFAULT_MIN_BRANCHES = 20 # Renamed for clarity to reflect its use for 'x' tips
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


def find_largest_subtree_in_interval(tree, t_start, t_end):
    """
    Find the largest subtree (by total branches) that falls completely within
    a time interval and extract ALL its internal and external branch lengths.

    :param tree: ete3.Tree, the tree of interest
    :param t_start: float, start time of interval
    :param t_end: float, end time of interval
    :return: tuple of (internal_branches, external_branches) from largest subtree
    """

    def subtree_falls_in_interval(node, t_start_check, t_end_check):
        """Check if entire subtree rooted at node falls within interval.
           A branch (node.up -> node) is considered to fall within the interval
           if its start_time (node.up.time) and end_time (node.time) are both within.
           However, for a *subtree*, all its *nodes* and *their incoming branches*
           must be within the interval.
        """
        # Check the root of the potential subtree itself
        node_time = getattr(node, TIME)
        if node_time < t_start_check or node_time > t_end_check:
            return False

        # Check the incoming branch to the root of this subtree
        if not node.is_root() and node.dist is not None:
            branch_start_time = getattr(node.up, TIME)
            if branch_start_time < t_start_check or node_time > t_end_check:
                return False

        # Check all descendants and their incoming branches
        for descendant in node.traverse():
            if descendant == node: # Skip the root of the current subtree
                continue

            desc_time = getattr(descendant, TIME)
            # This check is for the node's existence within the time boundaries
            if desc_time < t_start_check or desc_time > t_end_check:
                return False

            # Check the incoming branch to the descendant
            if descendant.dist is not None:
                desc_branch_start_time = getattr(descendant.up, TIME)
                if desc_branch_start_time < t_start_check or desc_time > t_end_check:
                    return False
        return True

    def extract_all_branches_from_subtree(node):
        """Extract all internal and external branches from a given subtree."""
        internal = []
        external = []
        for descendant in node.traverse():
            if descendant == node: # Skip the root of the subtree itself when considering its incoming branch
                continue
            if descendant.dist is not None: # Ensure it has a valid branch length
                if descendant.is_leaf():
                    external.append(descendant.dist)
                else:
                    internal.append(descendant.dist)
        return internal, external

    # Find all potential subtree roots
    candidate_roots = []
    max_branches = -1
    best_root = None

    for node in tree.traverse():
        # Only consider nodes that are within the interval as potential subtree roots
        if not hasattr(node, TIME) or node.is_root(): # Root doesn't have an incoming branch
            continue

        if t_start <= getattr(node, TIME) <= t_end:
            if subtree_falls_in_interval(node, t_start, t_end):
                # Count total branches in this potential subtree
                current_internal, current_external = extract_all_branches_from_subtree(node)
                total_branches = len(current_internal) + len(current_external)

                if total_branches > max_branches:
                    max_branches = total_branches
                    best_root = node

    if best_root is None:
        logging.warning(f"No valid subtrees found fully within interval [{t_start:.4f}, {t_end:.4f}]")
        return [], []
    else:
        logging.info(
            f"Selected largest subtree rooted at time {getattr(best_root, TIME):.4f} "
            f"with {max_branches} total branches in interval [{t_start:.4f}, {t_end:.4f}]")
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
                logging.warning(f"Tip {node.name} does not have a 'time' attribute. Ensure annotate_tree_with_time was run.")
                return None

    if len(tip_times) < n_tips_threshold:
        logging.warning(f"Not enough tips ({len(tip_times)}) to find time for {n_tips_threshold} tips.")
        return None

    tip_times.sort()
    # The time it takes to have n_tips_threshold tips is the time of the (n_tips_threshold-1)th tip (0-indexed)
    return tip_times[n_tips_threshold - 1]


def prune_tree_at_time(original_tree: Tree, time_threshold: float) -> Optional[Tree]:
    """
    Prunes a tree by removing all branches and nodes that extend beyond a given time threshold.
    This creates a new tree where all tips are at or before 'time_threshold'.
    Branches that cross the threshold are truncated.

    :param original_tree: The ete3.Tree to prune.
    :param time_threshold: The maximum time from the root allowed in the pruned tree.
    :return: A new ete3.Tree pruned at the specified time, or None if the tree becomes empty.
    """
    # Create a deep copy to avoid modifying the original tree
    pruned_tree = original_tree.copy("deepcopy")
    annotate_tree_with_time(pruned_tree) # Re-annotate times in the copy

    # Collect nodes (including internal) that are beyond the threshold
    nodes_to_remove = []
    for node in pruned_tree.traverse("postorder"):
        node_time = getattr(node, TIME)
        # If the node itself is beyond the threshold, or its incoming branch starts before and ends after
        # the threshold (meaning the branch crosses the threshold), we need to handle it.
        if node_time > time_threshold:
            nodes_to_remove.append(node)
            # If it's a leaf, just mark it for removal
            # If it's an internal node, all its children and its branch are beyond,
            # so we mark it and its branch for removal and propagate up.
        elif not node.is_root() and getattr(node.up, TIME) < time_threshold < node_time:
            # This branch crosses the time_threshold. Truncate it.
            # Adjust the branch length and node time
            node.dist = time_threshold - getattr(node.up, TIME)
            node.time = time_threshold # Update the time attribute for consistency
            # If this node has children, they might now be 'hanging' beyond the new time.
            # Their branches must also be truncated or removed.
            # This implies a recursive truncation, which `remove_certain_leaves` is not designed for.
            # A simpler way is to find all tips that are now beyond T and remove them.
            # Then, ensure internal nodes that become leaves or empty are handled.
            pass # We'll handle this by pruning tips later

    # Create a list of tips that are beyond the threshold
    tips_to_remove = [
        tip for tip in pruned_tree.get_leaves()
        if hasattr(tip, TIME) and getattr(tip, TIME) > time_threshold
    ]

    # Use the provided remove_certain_leaves with a lambda for tips to remove
    # This function needs to be slightly adapted or used carefully since it's designed
    # for *leaves*. For internal nodes whose entire lineage is beyond T, we need a different approach.

    # Simpler and more robust approach: Iteratively remove nodes/subtrees whose root is after T
    # or whose incoming branch means they are effectively "after" T.
    # The most direct way with ETE3 to "prune at a time" is to identify all nodes that *exist*
    # up to time T and then build a new tree from them, or carefully detach.

    # Let's use an iterative approach to remove branches/nodes that start *after* threshold.
    # This is more precise than just removing tips.
    # Iterate in post-order to ensure children are processed before parents.
    nodes_to_detach = []
    for node in pruned_tree.traverse("postorder"):
        if node.is_root(): # Don't remove the root
            continue

        branch_end_time = getattr(node, TIME)
        branch_start_time = getattr(node.up, TIME) if node.up else 0

        # Case 1: The entire branch is beyond the threshold
        if branch_start_time >= time_threshold:
            nodes_to_detach.append(node)
        # Case 2: The branch crosses the threshold. Truncate it.
        elif branch_end_time > time_threshold > branch_start_time:
            node.dist = time_threshold - branch_start_time
            node.time = time_threshold # Update node's time after truncation
            # If this node has children, they are now beyond the new 'time' of this node.
            # We need to remove any children whose start time (now relative to this node) is > 0
            # effectively detaching lineages that extend past T.
            children_to_remove = []
            for child in node.children:
                if hasattr(child, TIME) and getattr(child, TIME) > time_threshold:
                    children_to_remove.append(child)
            for child in children_to_remove:
                node.remove_child(child)
                # If a child was removed, and this node becomes a leaf and its time is T,
                # it's the new effective tip of that lineage.
            if not node.children and not node.is_leaf(): # If internal node became empty after children removal
                node.is_leaf = lambda: True # Make it a pseudo-leaf


    for node in nodes_to_detach:
        if not node.is_root() and node.up:
            parent = node.up
            parent.remove_child(node)
            # If removing a child makes parent have only one child, merge
            if len(parent.children) == 1 and not parent.is_root():
                child_to_merge = parent.children[0]
                child_to_merge.dist += parent.dist
                grandparent = parent.up
                if grandparent:
                    grandparent.remove_child(parent)
                    grandparent.add_child(child_to_merge)
                else: # Parent was root, new root is child_to_merge
                    child_to_merge.up = None
                    pruned_tree = child_to_merge
            elif not parent.children and not parent.is_root(): # If parent becomes an empty internal node
                 # If the parent becomes a leaf, its time must be at or before T
                parent.is_leaf = lambda: True # Make it a pseudo-leaf

    # Final check: if the root itself was removed or became empty
    if not pruned_tree.get_leaves(): # Tree became empty
        logging.warning("Tree became empty after pruning.")
        return None

    # Re-annotate the pruned tree to ensure all times are consistent after truncation/removal
    annotate_tree_with_time(pruned_tree)

    return pruned_tree


def sky_test_new_strategy(tree, min_internal_branches_for_T=DEFAULT_MIN_BRANCHES):
    """
    New strategy BD-Skyline test based on tip accumulation and subtree comparison.

    :param tree: ete3.Tree, the tree of interest
    :param min_internal_branches_for_T: int, the 'x' for x+2 tips to define T
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

    # Step 1: Find T based on x + 2 tips
    n_tips_for_T = min_internal_branches_for_T + 2
    T = find_time_for_n_tips(tree, n_tips_for_T)
    split_times['T_from_tips'] = T

    if T is None or T >= tree_height or T <= 0: # T should be positive and less than tree height
        logging.warning(f"Could not determine a valid T based on {n_tips_for_T} tips, or T is too large/small. T={T:.4f}")
        results['internal'] = results['external'] = None
        return False, results, False, split_times # Early exit if T is invalid

    logging.info(f"Determined T = {T:.4f} based on {n_tips_for_T} tips.")


    # Step 2: Study the largest subtree in the late interval [tree_height - T, tree_height]
    late_interval_start = tree_height - T
    if late_interval_start < 0: # Ensure interval doesn't start before time 0
        late_interval_start = 0

    logging.info(f"Analyzing late interval for largest subtree: [{late_interval_start:.4f}, {tree_height:.4f}] on original tree.")
    late_internal_branches, late_external_branches = find_largest_subtree_in_interval(tree, late_interval_start, tree_height)


    # Step 3: Prune the tree until time T for early interval analysis
    logging.info(f"Pruning tree at time T={T:.4f} for early interval analysis.")
    pruned_tree_for_early_analysis = prune_tree_at_time(tree, T)

    if pruned_tree_for_early_analysis is None:
        logging.warning(f"Tree became empty after pruning at time {T:.4f}. Cannot perform early interval analysis.")
        results['internal'] = results['external'] = None
        return False, results, False, split_times # Early exit if pruned tree is empty

    # Step 4: Study the largest subtree in the early interval [0, T] of the *pruned* tree
    # For the pruned tree, the "first interval" [0, T] is essentially the entire tree.
    # So we're looking for the largest subtree in this (already time-constrained) tree.
    logging.info(f"Analyzing early interval for largest subtree: [0, {T:.4f}] on PRUNED tree.")
    early_internal_branches, early_external_branches = find_largest_subtree_in_interval(pruned_tree_for_early_analysis, 0, T)


    # Step 5: Compare distributions
    alpha = 0.05

    # Internal branches comparison
    if len(early_internal_branches) >= min_internal_branches_for_T and len(late_internal_branches) >= min_internal_branches_for_T:
        u_result_internal = scipy.stats.mannwhitneyu(early_internal_branches, late_internal_branches, alternative='two-sided')
        results['internal'] = {
            'T': T,
            'early_interval': (0, T),
            'late_interval': (late_interval_start, tree_height),
            'early_branches': early_internal_branches,
            'late_branches': late_internal_branches,
            'early_count': len(early_internal_branches),
            'late_count': len(late_internal_branches),
            'u_statistic': u_result_internal.statistic,
            'p_value': u_result_internal.pvalue,
            'method': 'new_strategy'
        }
        logging.info(f"Internal branches - T={T:.4f}")
        logging.info(f"  Early subtree (0-{T:.4f}): {len(early_internal_branches)} branches")
        logging.info(f"  Late subtree ({late_interval_start:.4f}-{tree_height:.4f}): {len(late_internal_branches)} branches")
        logging.info(f"  Mann-Whitney U statistic: {u_result_internal.statistic:.4f}, p-value: {u_result_internal.pvalue:.6f}")
    else:
        logging.warning(f"Insufficient internal branches for comparison (early: {len(early_internal_branches)}, late: {len(late_internal_branches)}) (Needed: {min_internal_branches_for_T})")
        results['internal'] = None

    # External branches comparison
    if len(early_external_branches) >= min_internal_branches_for_T and len(late_external_branches) >= min_internal_branches_for_T:
        u_result_external = scipy.stats.mannwhitneyu(early_external_branches, late_external_branches, alternative='two-sided')
        results['external'] = {
            'T': T,
            'early_interval': (0, T),
            'late_interval': (late_interval_start, tree_height),
            'early_branches': early_external_branches,
            'late_branches': late_external_branches,
            'early_count': len(early_external_branches),
            'late_count': len(late_external_branches),
            'u_statistic': u_result_external.statistic,
            'p_value': u_result_external.pvalue,
            'method': 'new_strategy'
        }
        logging.info(f"External branches - T={T:.4f}")
        logging.info(f"  Early subtree (0-{T:.4f}): {len(early_external_branches)} branches")
        logging.info(f"  Late subtree ({late_interval_start:.4f}-{tree_height:.4f}): {len(late_external_branches)} branches")
        logging.info(f"  Mann-Whitney U statistic: {u_result_external.statistic:.4f}, p-value: {u_result_external.pvalue:.6f}")
    else:
        logging.warning(f"Insufficient external branches for comparison (early: {len(early_external_branches)}, late: {len(late_external_branches)}) (Needed: {min_internal_branches_for_T})")
        results['external'] = None


    # Determine if evidence of skyline model is found
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

    # Count valid results
    valid_results = [k for k, v in results.items() if v is not None]
    if not valid_results:
        logging.warning("No valid results to plot.")
        return

    # Adjust figure size based on the number of valid results
    n_plots = len(valid_results)
    if n_plots == 0:
        logging.warning("No valid results to plot.")
        return

    fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 8))

    # If only one type of branch had valid results, axes might be 1D, reshape for consistent indexing
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    colors = ['skyblue', 'lightcoral']

    for i, branch_type in enumerate(valid_results):
        result = results[branch_type]

        # Plot early interval
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

        # Plot late interval
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
New BD-Skyline test for Birth-Death Skyline models.

This strategy compares the largest subtree in the late interval of the original tree
with the largest subtree in the early interval of a time-pruned tree.
The interval size T is determined by the time at which a certain number of tips (x+2) are accumulated.
""")

    parser.add_argument('--nwk', required=True, type=str,
                        help="Input tree file in Newick format")
    parser.add_argument('--log', type=str, help="Output log file")
    parser.add_argument('--plot', type=str, help="Output plot file")
    parser.add_argument('--x-tips', type=int, default=DEFAULT_MIN_BRANCHES,
                        help=f"The 'x' value for determining T (T is time to accumulate x+2 tips). (default: {DEFAULT_MIN_BRANCHES})")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")

    args = parser.parse_args()

    # Set up logging level based on verbose argument
    level = logging.INFO if args.verbose else logging.WARNING
    # Reconfigure basicConfig to allow changing level after initial import if needed
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


    try:
        # Read tree
        tree = Tree(args.nwk, format=1)
        total_tips = len(tree.get_leaves())

        # Log total tips
        print(f"Total tips in tree: {total_tips}")

        # Run the new strategy test
        evidence_found, results, bonferroni_evidence, split_times = sky_test_new_strategy(tree,
                                                                                           args.x_tips)

        # Print split times prominently
        print("\n" + "=" * 50)
        print("NEW BD-SKYLINE TEST RESULTS")
        print("=" * 50)

        # Ensure tree_height is calculated if annotate_tree_with_time ran successfully
        tree_height = 0.0
        if hasattr(tree.get_tree_root(), TIME):
             tree_height = max(getattr(node, TIME) for node in tree.traverse() if hasattr(node, TIME))

        if 'T_from_tips' in split_times and split_times['T_from_tips'] is not None:
            split_time_T = split_times['T_from_tips']
            percentage = (split_time_T / tree_height) * 100 if tree_height > 0 else 0
            print(f"Time (T) based on {args.x_tips + 2} tips: {split_time_T:.6f} ({percentage:.1f}% of tree height)")
        else:
            print("Time (T) based on tips: Not available (insufficient tips or calculation error)")

        print("=" * 50)

        # Results summary
        if bonferroni_evidence:
            print("\nNEW SKY test: Evidence of BD-Skyline model detected (Bonferroni corrected)")
        elif evidence_found:
            print("\nNEW SKY test: Evidence of BD-Skyline model detected (uncorrected)")
        else:
            print("\nNEW SKY test: No evidence of BD-Skyline model (consistent with simple BD)")

        # Print detailed results
        for branch_type in ['internal', 'external']:
            if results[branch_type] is not None:
                result = results[branch_type]
                print(f"\n{branch_type.capitalize()} branches (new strategy comparison):")
                print(f"  T used = {result['T']:.4f}")
                print(
                    f"  Early subtree interval [{result['early_interval'][0]:.4f}, {result['early_interval'][1]:.4f}]: {result['early_count']} branches")
                print(
                    f"  Late subtree interval [{result['late_interval'][0]:.4f}, {result['late_interval'][1]:.4f}]: {result['late_count']} branches")
                print(f"  Mann-Whitney U statistic: {result['u_statistic']:.4f}")
                print(f"  p-value: {result['p_value']:.6f}")
            else:
                print(f"\n{branch_type.capitalize()} branches: Not enough data for comparison.")

        # Generate plot if requested
        if args.plot:
            plot_early_vs_late_results(tree, results, args.plot)

        # Write log if requested
        if args.log:
            with open(args.log, 'w') as f:
                f.write('New BD-Skyline test results - Largest Subtree comparison\n')
                f.write('=========================================================\n')
                f.write(f'Total tips in tree: {total_tips}\n')
                f.write(f'Tree height: {tree_height:.6f}\n')
                f.write('\nCALCULATED TIME T:\n')
                f.write('-----------------\n')
                if 'T_from_tips' in split_times and split_times['T_from_tips'] is not None:
                    split_time_T = split_times['T_from_tips']
                    percentage = (split_time_T / tree_height) * 100 if tree_height > 0 else 0
                    f.write(f"Time (T) based on {args.x_tips + 2} tips: {split_time_T:.6f} ({percentage:.1f}% of tree height)\n")
                else:
                    f.write("Time (T) based on tips: Not available (insufficient tips or calculation error)\n")

                f.write(f'\nEvidence of skyline model (uncorrected): {"Yes" if evidence_found else "No"}\n')
                f.write(f'Evidence of skyline model (Bonferroni): {"Yes" if bonferroni_evidence else "No"}\n')

                for branch_type in ['internal', 'external']:
                    if results[branch_type] is not None:
                        result = results[branch_type]
                        f.write(f'\n{branch_type.capitalize()} branches (new strategy):\n')
                        f.write(f'  T used = {result["T"]:.6f}\n')
                        f.write(
                            f'  Early subtree interval: [{result["early_interval"][0]:.6f}, {result["early_interval"][1]:.6f}] ({result["early_count"]} branches)\n')
                        f.write(
                            f'  Late subtree interval: [{result["late_interval"][0]:.6f}, {result["late_interval"][1]:.6f}] ({result["late_count"]} branches)\n')
                        f.write(f'  Mann-Whitney U statistic: {result["u_statistic"]:.6f}\n')
                        f.write(f'  p-value: {result["p_value"]:.6f}\n')
                    else:
                        f.write(f'\n{branch_type.capitalize()} branches: Not enough data for comparison.\n')

    except Exception as e:
        logging.error(f"Error running new BD-Skyline test: {e}", exc_info=True) # exc_info for full traceback
        return 1

    return 0


if __name__ == '__main__':
    exit(main())