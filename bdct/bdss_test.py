import logging
import numpy as np
import scipy.stats
from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time

DEFAULT_PERCENTILE = 0.25


class TripletMotif(object):
    def __init__(self, parent_node, child1_node, child2_node, split_time):
        """
        A motif representing a parent node and its two direct child nodes.
        It's assumed parent_node.dist is the length of the parent branch,
        and child1_node.dist, child2_node.dist are the lengths of the child branches.
        """
        self.parent_node = parent_node
        self.child1_node = child1_node
        self.child2_node = child2_node
        self.split_time = split_time #use parent node time

    def __str__(self):
        return (f"Triplet with parent {self.parent_node.name} (length {self.parent_node.dist}), "
                f"children {self.child1_node.name} (length {self.child1_node.dist}), "
                f"{self.child2_node.name} (length {self.child2_node.dist}) at split time {self.split_time}")


def pick_triplets(tree):
    """
    Picks triplets of a parent branch and its two child branches in the given tree.
    Only considers binary splits (nodes with exactly two children) where BOTH children are internal nodes.

    This restriction ensures we're comparing like with like (internal nodes only).

    :param tree: ete3.Tree, the tree of interest
    :return: list of TripletMotif objects
    """
    all_triplets = []
    for node in tree.traverse("postorder"):
        if len(node.children) == 2 and node.up:
            child1 = node.children[0]
            child2 = node.children[1]

            # KEEP RESTRICTION: Ensure both children are internal nodes (not leaves)
            # This makes sense to compare like with like
            if not child1.is_leaf() and not child2.is_leaf():
                split_time = getattr(node, TIME, 0.0)
                all_triplets.append(TripletMotif(parent_node=node,
                                                 child1_node=child1,
                                                 child2_node=child2,
                                                 split_time=split_time))
    return all_triplets


def get_real_vs_reshuffled_diffs(all_triplets):
    """
    Generate real vs reshuffled differences for triplets.

    Inspired by BDEI cherry logic but adapted for triplets:
    - Real differences: |child.dist - parent.dist| within each triplet
    - Reshuffled: swap parent lengths between neighboring triplets and recalculate differences

    For each pair of triplets, we make 8 individual comparisons:
    1-4: triplet1's real parent-child1 diff against all possible cross-triplet combinations
    5-8: triplet1's real parent-child2 diff against all possible cross-triplet combinations
    """
    n_triplets = len(all_triplets)

    # Arrays for all individual comparisons
    real_diffs = []
    random_diffs = []

    if n_triplets > 1:
        # Process triplets in pairs
        for i in range(0, n_triplets - 1, 2):
            triplet1 = all_triplets[i]
            triplet2 = all_triplets[i + 1]

            # Real differences (within-triplet parent-child)
            real_diff_1_c1 = abs(triplet1.child1_node.dist - triplet1.parent_node.dist)
            real_diff_1_c2 = abs(triplet1.child2_node.dist - triplet1.parent_node.dist)
            real_diff_2_c1 = abs(triplet2.child1_node.dist - triplet2.parent_node.dist)
            real_diff_2_c2 = abs(triplet2.child2_node.dist - triplet2.parent_node.dist)

            # Cross-triplet differences (swapped parents)
            cross_diff_1c1_2p = abs(triplet1.child1_node.dist - triplet2.parent_node.dist)
            cross_diff_1c2_2p = abs(triplet1.child2_node.dist - triplet2.parent_node.dist)
            cross_diff_2c1_1p = abs(triplet2.child1_node.dist - triplet1.parent_node.dist)
            cross_diff_2c2_1p = abs(triplet2.child2_node.dist - triplet1.parent_node.dist)

            # 8 comparisons as in BDEI logic:
            # Triplet1's real diffs against cross combinations
            real_diffs.extend([real_diff_1_c1, real_diff_1_c1, real_diff_1_c2, real_diff_1_c2])
            random_diffs.extend([cross_diff_1c1_2p, cross_diff_2c1_1p, cross_diff_1c2_2p, cross_diff_2c2_1p])

            # Triplet2's real diffs against cross combinations
            real_diffs.extend([real_diff_2_c1, real_diff_2_c1, real_diff_2_c2, real_diff_2_c2])
            random_diffs.extend([cross_diff_2c1_1p, cross_diff_1c1_2p, cross_diff_2c2_1p, cross_diff_1c2_2p])

        # Handle odd number of triplets (last 3 triplets in cycle)
        if n_triplets % 2 == 1 and n_triplets >= 3:
            triplet1 = all_triplets[-3]
            triplet2 = all_triplets[-2]
            triplet3 = all_triplets[-1]

            # Real differences
            real_diff_1_c1 = abs(triplet1.child1_node.dist - triplet1.parent_node.dist)
            real_diff_1_c2 = abs(triplet1.child2_node.dist - triplet1.parent_node.dist)
            real_diff_2_c1 = abs(triplet2.child1_node.dist - triplet2.parent_node.dist)
            real_diff_2_c2 = abs(triplet2.child2_node.dist - triplet2.parent_node.dist)
            real_diff_3_c1 = abs(triplet3.child1_node.dist - triplet3.parent_node.dist)
            real_diff_3_c2 = abs(triplet3.child2_node.dist - triplet3.parent_node.dist)

            # Cyclic cross-triplet comparisons (1↔2, 2↔3, 3↔1)
            # Triplet 1 vs Triplet 2
            cross_1c1_2p = abs(triplet1.child1_node.dist - triplet2.parent_node.dist)
            cross_1c2_2p = abs(triplet1.child2_node.dist - triplet2.parent_node.dist)
            cross_2c1_1p = abs(triplet2.child1_node.dist - triplet1.parent_node.dist)
            cross_2c2_1p = abs(triplet2.child2_node.dist - triplet1.parent_node.dist)

            # Triplet 2 vs Triplet 3
            cross_2c1_3p = abs(triplet2.child1_node.dist - triplet3.parent_node.dist)
            cross_2c2_3p = abs(triplet2.child2_node.dist - triplet3.parent_node.dist)
            cross_3c1_2p = abs(triplet3.child1_node.dist - triplet2.parent_node.dist)
            cross_3c2_2p = abs(triplet3.child2_node.dist - triplet2.parent_node.dist)

            # Triplet 3 vs Triplet 1
            cross_3c1_1p = abs(triplet3.child1_node.dist - triplet1.parent_node.dist)
            cross_3c2_1p = abs(triplet3.child2_node.dist - triplet1.parent_node.dist)
            cross_1c1_3p = abs(triplet1.child1_node.dist - triplet3.parent_node.dist)
            cross_1c2_3p = abs(triplet1.child2_node.dist - triplet3.parent_node.dist)

            # 8 comparisons for each triplet pair in the cycle
            # Triplet 1 vs 2
            real_diffs.extend([real_diff_1_c1, real_diff_1_c1, real_diff_1_c2, real_diff_1_c2])
            random_diffs.extend([cross_1c1_2p, cross_2c1_1p, cross_1c2_2p, cross_2c2_1p])

            # Triplet 2 vs 3
            real_diffs.extend([real_diff_2_c1, real_diff_2_c1, real_diff_2_c2, real_diff_2_c2])
            random_diffs.extend([cross_2c1_3p, cross_3c1_2p, cross_2c2_3p, cross_3c2_2p])

            # Triplet 3 vs 1
            real_diffs.extend([real_diff_3_c1, real_diff_3_c1, real_diff_3_c2, real_diff_3_c2])
            random_diffs.extend([cross_3c1_1p, cross_1c1_3p, cross_3c2_1p, cross_1c2_3p])

    else:
        # Single triplet case - no swapping possible
        if n_triplets == 1:
            triplet = all_triplets[0]
            real_diff_c1 = abs(triplet.child1_node.dist - triplet.parent_node.dist)
            real_diff_c2 = abs(triplet.child2_node.dist - triplet.parent_node.dist)
            real_diffs = [real_diff_c1, real_diff_c2]
            random_diffs = [real_diff_c1, real_diff_c2]  # No swap possible

    return np.array(random_diffs), np.array(real_diffs)


def bdss_test(forest):
    """
    Tests if the input forest was generated under a BDSS (Birth-Death with Superspreading) model.

    Inspired by BDEI test logic but adapted for triplets:
    - Detects triplets (parent + 2 internal children) and sorts by split time
    - Calculates parent-child branch length differences within triplets
    - Swaps parent lengths between neighboring triplets
    - Tests if real differences are smaller than reshuffled differences more often than expected
    - Uses binomial test with alternative='greater' (counting real < reshuffled)

    :param forest: list of trees
    :return: pval, n_triplets
    """
    annotate_forest_with_time(forest)

    all_triplets = []
    for tree in forest:
        all_triplets.extend(pick_triplets(tree))
    all_triplets = sorted(all_triplets, key=lambda x: x.split_time)

    n_triplets = len(all_triplets)
    logging.info(f'Picked {n_triplets} triplets.')

    if n_triplets < 2:
        logging.warning("Not enough triplets to perform BDSS test (need at least 2). Returning p-value of 1.")
        return 1.0, n_triplets

    # Get real vs reshuffled differences using BDEI-inspired logic
    random_diffs, real_diffs = get_real_vs_reshuffled_diffs(all_triplets)

    if len(real_diffs) < 2:
        logging.warning("Not enough comparisons generated. Returning p-value of 1.")
        return 1.0, n_triplets

    # Count how many times real < reshuffled (hypothesis: this should be more common under BDSS)
    count = np.sum(real_diffs < random_diffs)
    total = len(real_diffs)

    # Use alternative='greater' like in BDEI test
    pval = scipy.stats.binomtest(count, n=total, p=0.5, alternative='greater').pvalue

    logging.info(f"Real < Reshuffled: {count} out of {total} comparisons ({count / total:.3f})")

    return pval, n_triplets


def main():
    """
    Entry point for BDSS test with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="""BDSS-test (BDEI-inspired version).

Checks if the input forest was generated under a BDSS model.

Inspired by BDEI test logic but adapted for triplets:
- Detects triplets (parent + 2 internal children) instead of cherries
- Calculates parent-child branch length differences instead of sibling differences
- Uses neighbor swapping logic similar to BDEI cherry approach
- Tests if real differences are systematically smaller than reshuffled differences
""")
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input forest file in newick or nexus format")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    pval, n_triplets = bdss_test(forest)

    logging.info(f"BDSS test p-value: {pval} on {n_triplets} triplets.")

    with open(params.log, 'w+') as f:
        f.write('BDSS-test p-value\tnumber of triplets\n')
        f.write(f'{pval:g}\t{n_triplets}\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Configure logging to show info messages
    main()