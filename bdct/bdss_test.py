import logging
import numpy as np
import scipy.stats
from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time

DEFAULT_PERCENTILE = 0.25


def find_closest_non_sibling_branch(target_branch, all_branches):
    """
    Find the branch closest in time to target_branch that doesn't share the same parent.

    :param target_branch: The branch for which to find the closest non-sibling
    :param all_branches: List of all branches in the forest
    :return: The closest non-sibling branch, or None if none exists
    """
    # Get the start time of the target branch (when its parent node split)
    target_start_time = getattr(target_branch.up, TIME, 0.0) if target_branch.up else 0.0

    # Filter out sibling branches (branches with the same parent) and the target itself
    non_sibling_branches = []
    for branch in all_branches:
        if branch != target_branch and branch.up != target_branch.up:
            non_sibling_branches.append(branch)

    if not non_sibling_branches:
        return None

    # Find the closest by start time
    closest_branch = None
    min_time_diff = float('inf')

    for branch in non_sibling_branches:
        branch_start_time = getattr(branch.up, TIME, 0.0) if branch.up else 0.0
        time_diff = abs(branch_start_time - target_start_time)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_branch = branch

    return closest_branch


def trace_lineage_shorter_pattern(starting_branch, all_branches):
    """
    Starting from a branch, trace up the lineage checking if each branch
    is shorter than its closest non-sibling neighbor. Continue until we find
    a branch that is NOT shorter than its non-sibling neighbor.

    :param starting_branch: The branch to start tracing from
    :param all_branches: List of all branches in the forest
    :return: Number of consecutive branches that are shorter than their non-sibling neighbors
    """
    current_branch = starting_branch
    consecutive_shorter = 0

    while current_branch and current_branch.up:  # Don't process the root (which has no parent)
        closest_non_sibling = find_closest_non_sibling_branch(current_branch, all_branches)

        if closest_non_sibling is None:
            # No non-sibling found, can't continue comparison
            break

        if current_branch.dist < closest_non_sibling.dist:
            # Current branch is shorter than its closest non-sibling
            consecutive_shorter += 1
            # Move up to the parent branch
            current_branch = current_branch.up
        else:
            # Current branch is NOT shorter, stop tracing
            break

    return consecutive_shorter


def bdss_test(forest):
    """
    Tests if the input forest was generated under a BDSS (Birth-Death with Superspreading) model.

    Strategy:
    1. Find pairs (child-parent) where both are shorter than their non-sibling neighbors
    2. For each such pair, check if it extends to a triplet (child-parent-grandparent)
       where the grandparent is also shorter than its non-sibling neighbor
    3. Use binomial test: given a pair, probability of extending to triplet should be 0.5 under BD,
       but > 0.5 under BDSS (superspreader lineages)

    :param forest: list of trees
    :return: pval, n_total_pairs
    """
    annotate_forest_with_time(forest)

    # Collect all internal branches (non-leaf, non-root nodes)
    all_branches = []
    for tree in forest:
        for node in tree.traverse():
            if not node.is_leaf() and node.up:  # Internal branch with a parent
                all_branches.append(node)

    n_branches = len(all_branches)
    logging.info(f'Found {n_branches} internal branches.')

    if n_branches < 3:
        logging.warning("Not enough branches to perform BDSS test (need at least 3). Returning p-value of 1.")
        return 1.0, 0

    # Count pairs where both child and parent are shorter than their non-sibling neighbors
    n_total_pairs = 0  # Total number of child-parent pairs where both are shorter
    n_triplets = 0  # Number of those pairs that extend to triplets (grandparent also shorter)

    for branch in all_branches:
        # Check if this branch (child) is shorter than its non-sibling neighbor
        child_closest = find_closest_non_sibling_branch(branch, all_branches)
        if child_closest is None or branch.dist >= child_closest.dist:
            continue  # Child is not shorter, skip this lineage

        # Check if the parent is also shorter than its non-sibling neighbor
        parent = branch.up
        if parent is None or parent.up is None:  # No parent or grandparent
            continue

        parent_closest = find_closest_non_sibling_branch(parent, all_branches)
        if parent_closest is None or parent.dist >= parent_closest.dist:
            continue  # Parent is not shorter, skip this lineage

        # We have a valid pair: both child and parent are shorter than neighbors
        n_total_pairs += 1

        # Check if this extends to a triplet (grandparent also shorter)
        grandparent = parent.up
        if grandparent is None or grandparent.up is None:  # No grandparent or great-grandparent
            continue

        grandparent_closest = find_closest_non_sibling_branch(grandparent, all_branches)
        if grandparent_closest is not None and grandparent.dist < grandparent_closest.dist:
            # Grandparent is also shorter - we have a triplet!
            n_triplets += 1

    if n_total_pairs == 0:
        logging.warning(
            "No valid child-parent pairs found where both are shorter than neighbors. Returning p-value of 1.")
        return 1.0, 0

    # Binomial test: given a pair, what's the probability it extends to a triplet?
    # Under BD model: probability = 0.5
    # Under BDSS model: probability > 0.5 (superspreader lineages extend further)
    pval = scipy.stats.binomtest(
        n_triplets,
        n=n_total_pairs,
        p=0.5,
        alternative='greater'
    ).pvalue

    logging.info(f"Pairs (child+parent both shorter): {n_total_pairs}")
    logging.info(
        f"Triplets (extending to grandparent): {n_triplets} out of {n_total_pairs} ({n_triplets / n_total_pairs:.3f})")

    return pval, n_total_pairs


def main():
    """
    Entry point for BDSS test with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="""BDSS-test (Lineage Extension Strategy).

Tests if the input forest was generated under a BDSS model.

Strategy:
- Finds pairs (child-parent) where both are shorter than their non-sibling neighbors
- Tests how often these pairs extend to triplets (child-parent-grandparent all shorter)
- Uses binomial test: under BD model probability = 0.5, under BDSS > 0.5
""")
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input forest file in newick or nexus format")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    pval, n_total_pairs = bdss_test(forest)

    logging.info(f"BDSS test p-value: {pval} on {n_total_pairs} pairs.")

    with open(params.log, 'w+') as f:
        f.write('BDSS-test p-value\tnumber of pairs\n')
        f.write(f'{pval:g}\t{n_total_pairs}\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Configure logging to show info messages
    main()