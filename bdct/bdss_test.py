import logging
import numpy as np
import scipy.stats
from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time
import math

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
        self.split_time = split_time

    def __str__(self):
        return (f"Triplet with parent {self.parent_node.name} (length {self.parent_node.dist}), "
                f"children {self.child1_node.name} (length {self.child1_node.dist}), "
                f"{self.child2_node.name} (length {self.child2_node.dist}) at split time {self.split_time}")


def pick_triplets(tree):
    """
    Picks triplets of a parent branch and its two child branches in the given tree.
    Only considers binary splits (nodes with exactly two children) where BOTH children are internal nodes.
    Does NOT implement filtering for "impossible" triplets based on state.

    :param tree: ete3.Tree, the tree of interest
    :return: list of TripletMotif objects
    """
    all_triplets = []
    for node in tree.traverse("postorder"):
        if len(node.children) == 2 and node.up:
            child1 = node.children[0]
            child2 = node.children[1]

            # NEW CONDITION: Ensure both children are internal nodes (not leaves)
            if not child1.is_leaf() and not child2.is_leaf():
                split_time = getattr(node, TIME, 0.0)
                all_triplets.append(TripletMotif(parent_node=node,
                                                child1_node=child1,
                                                child2_node=child2,
                                                split_time=split_time))
    return all_triplets


def bdss_test(forest):
    """
    Tests if the input forest was generated under a BDSS (Birth-Death with Superspreading) model.

    The test detects triplets of a parent branch and its two child branches, and sorts them by
    the time at which the parent branch split into the two children.
    For each triplet, the test calculates the length differences between the parent branch and
    each of its child branches: d_i1 = |C_i1 - P_i| and d_i2 = |C_i2 - P_i|.

    It then generates a collection of random differences by *randomly shuffling* parent branches
    across all sorted triplets. The new differences d'_i1 = |C_i1 - P'_i| and d'_i2 = |C_i2 - P'_i| are then calculated.

    Finally, a sign test is performed comparing the reshuffled differences to the real ones.
    The hypothesis is that under BDSS, certain swaps (e.g., creating "impossible" triplets)
    will tend to *increase* the differences (d' > d) more often than under a pure BD model.

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

    real_parent_lengths = np.array([t.parent_node.dist for t in all_triplets], dtype=float)
    real_child1_lengths = np.array([t.child1_node.dist for t in all_triplets], dtype=float)
    real_child2_lengths = np.array([t.child2_node.dist for t in all_triplets], dtype=float)

    real_d1_diffs = np.abs(real_child1_lengths - real_parent_lengths)
    real_d2_diffs = np.abs(real_child2_lengths - real_parent_lengths)

    # Implement Parent Branch Reshuffling (using a true random shuffle)
    # Create a copy of the real parent lengths to shuffle.
    reshuffled_parent_lengths = np.copy(real_parent_lengths)
    # Perform a true random shuffle of the parent branch lengths.
    np.random.shuffle(reshuffled_parent_lengths)


    # Recalculate d'_i1 and d'_i2 (Reshuffled Differences)
    reshuffled_d1_diffs = np.abs(real_child1_lengths - reshuffled_parent_lengths)
    reshuffled_d2_diffs = np.abs(real_child2_lengths - reshuffled_parent_lengths)

    # Assign sign values (d'ij - dij)
    sign_d1 = np.sign(reshuffled_d1_diffs - real_d1_diffs)
    sign_d2 = np.sign(reshuffled_d2_diffs - real_d2_diffs)

    total_comparisons = n_triplets * 2
    k_greater = (sign_d1 == 1).sum() + (sign_d2 == 1).sum()

    pval = scipy.stats.binomtest(k_greater, n=total_comparisons, p=0.5, alternative='greater').pvalue

    return pval, n_triplets


def main():
    """
    Entry point for BDSS test with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="""BDSS-test.

Checks if the input forest was generated under a BDSS model.

The test detects triplets (parent branch and two child branches) in the forest and sorts them by
the time at which the parent branch split into the two children.
For each triplet, it calculates two length differences: |C_i1 - P_i| and |C_i2 - P_i|.
It then generates reshuffled differences by *randomly shuffling* parent branches across all sorted triplets.
New differences |C_i1 - P'_i| and |C_i2 - P'_i| are calculated.
A sign test is performed, counting instances where reshuffled differences are greater than real ones.
A significantly higher count of such instances suggests a BDSS model.
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
    logging.basicConfig(level=logging.INFO) # Configure logging to show info messages
    main()