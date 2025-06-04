import logging

import numpy as np
import scipy

from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time

DEFAULT_CHERRY_BLOCK_SIZE = 100

DEFAULT_NEIGHBOURHOOD_SIZE = 5

RANDOM_REPETITIONS = 1e3

DEFAULT_PERCENTILE = 0.25


class ParentChildrenMotif(object):

    def __init__(self, clustered_children, root=None, motif_type="tip"):
        """
        A motif that includes the subtree's root and root's children picked according to a certain criterion.

        :param root: ete.TreeNode, the root of the motif subtree
        :param clustered_children: list of clustered children
        :param motif_type: "tip" for traditional cherries, "internal" for internal node pairs
        """
        self.root = root
        self.clustered_children = clustered_children
        self.motif_type = motif_type

    def __str__(self):
        return (
            f"{self.motif_type.capitalize()} motif with root {self.root.name} and {len(self.clustered_children)} children: "
            f"{', '.join(_.name for _ in self.clustered_children)}")

    def __len__(self):
        return len(self.clustered_children)


def pick_cherries(tree, include_polytomies=True, include_internal=True):
    """
    Picks cherries in the given tree, including both tip cherries and internal cherries.

    :param include_polytomies: bool, whether to include nodes with > 2 children into consideration.
    :param tree: ete3.Tree, the tree of interest
    :param include_internal: bool, whether to include internal node pairs (not just tips)
    :return: iterator of Motif motifs
    """
    all_motifs = []

    for node in tree.traverse():
        if node.is_leaf():
            continue

        if not include_polytomies and len(node.children) != 2:
            continue

        # Traditional tip cherries
        tips = [child for child in node.children if child.is_leaf()]
        if len(tips) >= 2:
            all_motifs.append(ParentChildrenMotif(clustered_children=tips, root=node, motif_type="tip"))

        # Internal cherries (if enabled)
        if include_internal:
            internal_nodes = [child for child in node.children if not child.is_leaf()]
            if len(internal_nodes) >= 2:
                all_motifs.append(
                    ParentChildrenMotif(clustered_children=internal_nodes, root=node, motif_type="internal"))

            # Mixed cherries: one tip + one internal node
            if len(tips) >= 1 and len(internal_nodes) >= 1:
                # Create pairs of tip + internal
                for tip in tips:
                    for internal in internal_nodes:
                        all_motifs.append(
                            ParentChildrenMotif(clustered_children=[tip, internal], root=node, motif_type="mixed"))

    return all_motifs


def bdei_test(forest, include_internal=True):
    """
    Tests if the input forest was generated under a -BDEI model.

    The test detects cherries in the forest and sorts them by the times of their roots.
    For each cherry the test calculates the difference between its tip times,
    hence obtaining an array of cherry tip differences.

    Extended version: also includes internal cherries (pairs of internal nodes) to increase
    the number of examples and improve statistical power.

    :param forest: list of trees
    :param include_internal: bool, whether to include internal node pairs
    :return: pval, n_cherries, cherries_breakdown
    """
    annotate_forest_with_time(forest)

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=True, include_internal=include_internal))

    all_cherries = sorted(all_cherries, key=lambda _: getattr(_.root, TIME))

    n_cherries = len(all_cherries)

    # Count breakdown by type
    tip_cherries = sum(1 for c in all_cherries if c.motif_type == "tip")
    internal_cherries = sum(1 for c in all_cherries if c.motif_type == "internal")
    mixed_cherries = sum(1 for c in all_cherries if c.motif_type == "mixed")

    cherries_breakdown = {
        "total": n_cherries,
        "tip": tip_cherries,
        "internal": internal_cherries,
        "mixed": mixed_cherries
    }

    logging.info(
        f'Picked {n_cherries} cherries total: {tip_cherries} tip, {internal_cherries} internal, {mixed_cherries} mixed.')

    if n_cherries < 2:
        return 1, n_cherries, cherries_breakdown

    random_diffs, real_diffs = get_real_vs_reshuffled_diffs(all_cherries)
    pval = scipy.stats.binomtest((random_diffs > real_diffs).sum(), n=n_cherries, p=0.5, alternative='less').pvalue

    return pval, n_cherries, cherries_breakdown


def get_real_vs_reshuffled_diffs(all_couples):
    n_motifs = len(all_couples)
    first_dists, other_dists = np.zeros(n_motifs, dtype=float), np.zeros(n_motifs, dtype=float)

    for i, couple in enumerate(all_couples):
        if len(couple.clustered_children) >= 2:
            t1, t2 = np.random.choice(couple.clustered_children, size=2, replace=False)
        else:
            # Fallback for edge cases
            t1, t2 = couple.clustered_children[0], couple.clustered_children[0]

        first_dists[i] = t1.dist
        other_dists[i] = t2.dist

    if n_motifs > 1:
        # swap pairs of children
        reshuffled_other_dists = np.zeros(n_motifs, dtype=float)
        reshuffled_other_dists[:-1:2] = other_dists[1::2]
        reshuffled_other_dists[1::2] = other_dists[:-1:2]
        # if the number of couples is odd, swap the last 3 children in a circle
        if n_motifs % 2:
            reshuffled_other_dists[-1] = reshuffled_other_dists[-2]
            reshuffled_other_dists[-2] = other_dists[-1]
    else:
        reshuffled_other_dists = other_dists

    real_diffs = np.abs(first_dists - other_dists)
    random_diffs = np.abs(first_dists - reshuffled_other_dists)
    return random_diffs, real_diffs


def cherry_diff_plot(forest, outfile=None, include_internal=True):
    """
    Plots cherry tip time differences against cherry root times.
    Requires matplotlib and seaborn installed.

    :param forest: list of trees
    :param outfile: (optional) output file where the plot should be saved.
        If not specified, the plot will be shown instead.
    :param include_internal: bool, whether to include internal cherries
    :return: void
    """

    from matplotlib import pyplot as plt
    from matplotlib.pyplot import show
    import seaborn as sns

    annotate_forest_with_time(forest)

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=True, include_internal=include_internal))

    def get_diff(cherry):
        if len(cherry.clustered_children) >= 2:
            b1, b2 = cherry.clustered_children[0], cherry.clustered_children[1]
        else:
            return 0
        return abs(b1.dist - b2.dist)

    plt.figure(figsize=(12, 8))

    # Separate by cherry type
    tip_cherries = [c for c in all_cherries if c.motif_type == "tip"]
    internal_cherries = [c for c in all_cherries if c.motif_type == "internal"]
    mixed_cherries = [c for c in all_cherries if c.motif_type == "mixed"]

    colors = ['blue', 'red', 'green']
    labels = ['Tip cherries', 'Internal cherries', 'Mixed cherries']
    cherry_types = [tip_cherries, internal_cherries, mixed_cherries]

    for cherries, color, label in zip(cherry_types, colors, labels):
        if len(cherries) == 0:
            continue

        x = np.array([getattr(_.root, TIME) for _ in cherries])
        diffs = np.array([get_diff(_) for _ in cherries])

        plt.scatter(x, diffs, alpha=0.6, color=color, label=f'{label} (n={len(cherries)})', s=20)

    plt.xlabel('Cherry root time')
    plt.ylabel('Cherry difference')
    plt.title('Cherry Differences vs Time (Including Internal Cherries)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if not outfile:
        show()
    else:
        plt.savefig(outfile, dpi=300)


def main():
    """
    Entry point for BDEI test with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="""BDEI-test with Internal Cherries.

Checks if the input forest was generated under a -BDEI model.

Extended version that includes:
- Traditional tip cherries (pairs of tip nodes with same parent)
- Internal cherries (pairs of internal nodes with same parent)  
- Mixed cherries (tip + internal node pairs with same parent)

This increases the number of examples dramatically, improving statistical power.

The test detects cherries in the forest and sorts them by the times of their roots. 
For each cherry the test calculates the difference between its branch lengths,
hence obtaining an array of real cherry differences. 
It then generates a collection of random cherry differences of the same size by
swapping branches between neighboring cherries.

The test therefore reports a probability of partner notification being present in the tree.""")

    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input forest file in newick or nexus format")
    parser.add_argument('--tips-only', action='store_true',
                        help="use only traditional tip cherries (disable internal cherries)")
    parser.add_argument('--plot', type=str, help="optional output plot file")
    params = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forest = read_forest(params.nwk)
    include_internal = not params.tips_only

    pval, n_cherries, breakdown = bdei_test(forest, include_internal=include_internal)

    logging.info(f"BDEI test p-value: {pval:.6f} on {n_cherries} cherries.")
    logging.info(
        f"Breakdown: {breakdown['tip']} tip, {breakdown['internal']} internal, {breakdown['mixed']} mixed cherries.")

    with open(params.log, 'w+') as f:
        f.write('BDEI-test p-value\tnumber of cherries\ttip cherries\tinternal cherries\tmixed cherries\n')
        f.write(f'{pval:g}\t{breakdown["total"]}\t{breakdown["tip"]}\t{breakdown["internal"]}\t{breakdown["mixed"]}\n')

    # Generate plot if requested
    if params.plot:
        cherry_diff_plot(forest, params.plot, include_internal=include_internal)
        logging.info(f"Plot saved to {params.plot}")


if __name__ == '__main__':
    main()



