import logging

import numpy as np
import scipy

from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time

DEFAULT_CHERRY_BLOCK_SIZE = 100

DEFAULT_NEIGHBOURHOOD_SIZE = 5

RANDOM_REPETITIONS = 1e3

DEFAULT_PERCENTILE = 0.25


class ParentChildrenMotif(object):

    def __init__(self, clustered_children, root=None):
        """
        A motif that includes the subtree's root and root's children picked according to a certain criterion.

        :param root: ete.TreeNode, the root of the motif subtree
        :param clustered_children: list of clustered children
        """
        self.root = root
        self.clustered_children = clustered_children

    def __str__(self):
        return (f"Motif with root {self.root.name} and {len(self.clustered_children)} clustered tips: "
                f"{', '.join(_.name for _ in self.clustered_children)}")

    def __len__(self):
        return len(self.clustered_children)


def pick_cherries(tree, include_polytomies=True, cherry_type="tip"):
    """
    Picks cherries in the given tree of a specific type.

    IMPORTANT: Different cherry types must be processed separately because they have
    different biological interpretations:
    - Tips: time to sampling/removal
    - Internal nodes: time to next transmission

    :param include_polytomies: bool, whether to include nodes with > 2 children into consideration.
    :param tree: ete3.Tree, the tree of interest
    :param cherry_type: "tip" for tip cherries, "internal" for internal node pairs
    :return: iterator of Motif motifs
    """
    for node in tree.traverse():
        if node.is_leaf():
            continue

        if not include_polytomies and len(node.children) != 2:
            continue

        if cherry_type == "tip":
            # Traditional tip cherries (pairs of tips that diverged at same time)
            tips = [child for child in node.children if child.is_leaf()]
            if len(tips) >= 2:
                yield ParentChildrenMotif(clustered_children=tips, root=node)

        elif cherry_type == "internal":
            # Internal node cherries (pairs of internal nodes that diverged at same time)
            internal_nodes = [child for child in node.children if not child.is_leaf()]
            if len(internal_nodes) >= 2:
                yield ParentChildrenMotif(clustered_children=internal_nodes, root=node)


def bdei_test(forest):
    """
    Tests if the input forest was generated under a -BDEI model.

    The test detects cherries in the forest and sorts them by the times of their roots.
    For each cherry the test calculates the difference between its tip times,
    hence obtaining an array of cherry tip differences.

    Extended version: processes tip cherries and internal cherries SEPARATELY to avoid
    mixing different biological interpretations (sampling vs transmission times).

    :param forest: list of trees
    :return: pval, n_cherries
    """
    annotate_forest_with_time(forest)

    # Process tip cherries separately
    tip_cherries = []
    for tree in forest:
        tip_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="tip"))
    tip_cherries = sorted(tip_cherries, key=lambda _: getattr(_.root, TIME))

    # Process internal cherries separately
    internal_cherries = []
    for tree in forest:
        internal_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="internal"))
    internal_cherries = sorted(internal_cherries, key=lambda _: getattr(_.root, TIME))

    n_tip_cherries = len(tip_cherries)
    n_internal_cherries = len(internal_cherries)
    n_total_cherries = n_tip_cherries + n_internal_cherries

    logging.info(f'Picked {n_total_cherries} cherries total: {n_tip_cherries} tip, {n_internal_cherries} internal.')

    if n_total_cherries < 2:
        return 1, n_total_cherries

    # Process tip cherries if any
    c_tips = np.array([], dtype=bool)
    if n_tip_cherries >= 2:
        random_diffs_tips, real_diffs_tips = get_real_vs_reshuffled_diffs(tip_cherries)
        c_tips = random_diffs_tips > real_diffs_tips

    # Process internal cherries if any
    c_inodes = np.array([], dtype=bool)
    if n_internal_cherries >= 2:
        random_diffs_internal, real_diffs_internal = get_real_vs_reshuffled_diffs(internal_cherries)
        c_inodes = random_diffs_internal > real_diffs_internal

    # Concatenate comparison arrays
    c_combined = np.concatenate([c_tips, c_inodes])

    # Apply binomial test to combined array
    pval = scipy.stats.binomtest(c_combined.sum(), n=len(c_combined), p=0.5, alternative='less').pvalue

    return pval, n_total_cherries


def get_real_vs_reshuffled_diffs(all_couples):
    n_motifs = len(all_couples)
    first_dists, other_dists = np.zeros(n_motifs, dtype=float), np.zeros(n_motifs, dtype=float)
    for i, couple in enumerate(all_couples):
        t1, t2 = np.random.choice(couple.clustered_children, size=2, replace=False)
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


def cherry_diff_plot(forest, outfile=None):
    """
    Plots cherry tip time differences against cherry root times.
    Requires matplotlib and seaborn installed.

    :param forest: list of trees
    :param outfile: (optional) output file where the plot should be saved.
        If not specified, the plot will be shown instead.
    :return: void
    """

    from matplotlib import pyplot as plt
    from matplotlib.pyplot import show
    import seaborn as sns

    annotate_forest_with_time(forest)

    # Get tip and internal cherries separately
    tip_cherries = []
    internal_cherries = []
    for tree in forest:
        tip_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="tip"))
        internal_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="internal"))

    def get_diff(cherry):
        b1, b2 = cherry.clustered_children
        return abs(b1.dist - b2.dist)

    plt.figure(figsize=(12, 6))

    # Plot tip cherries
    plt.subplot(1, 2, 1)
    if tip_cherries:
        x = np.array([getattr(_.root, TIME) for _ in tip_cherries])
        diffs = np.array([get_diff(_) for _ in tip_cherries])
        perc = np.percentile(diffs, [25, 50, 75])
        mask = np.digitize(diffs, perc)
        colors = sns.color_palette("colorblind")

        for i, label in zip(range(4), ('1st', '2nd', '3rd', '4th')):
            ax = sns.scatterplot(x=x[mask == i], y=diffs[mask == i], alpha=0.75,
                                 label='{} quantile'.format(label), color=colors[i])
    plt.xlabel('Tip cherry root time')
    plt.ylabel('Tip cherry difference')
    plt.title(f'Tip Cherries (n={len(tip_cherries)})')
    plt.legend()

    # Plot internal cherries
    plt.subplot(1, 2, 2)
    if internal_cherries:
        x = np.array([getattr(_.root, TIME) for _ in internal_cherries])
        diffs = np.array([get_diff(_) for _ in internal_cherries])
        perc = np.percentile(diffs, [25, 50, 75])
        mask = np.digitize(diffs, perc)
        colors = sns.color_palette("colorblind")

        for i, label in zip(range(4), ('1st', '2nd', '3rd', '4th')):
            ax = sns.scatterplot(x=x[mask == i], y=diffs[mask == i], alpha=0.75,
                                 label='{} quantile'.format(label), color=colors[i])
    plt.xlabel('Internal cherry root time')
    plt.ylabel('Internal cherry difference')
    plt.title(f'Internal Cherries (n={len(internal_cherries)})')
    plt.legend()

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
        argparse.ArgumentParser(description="""BDEI-test.

Checks if the input forest was generated under a -BDEI model.
The test detects cherries in the forest and sorts them by the times of their roots. 
For each cherry the test calculates the difference between its tip times, 
hence obtaining an array of real cherry tip differences. 
It then generates a collection of random cherry tip differences of the same size: 
Processing the cherries in couples from the two cherries with the oldest roots 
to the two (three if the total number of cherries is odd) cherries with the most recent roots,
we pick one tip per cherry and swap them. We then calculate the tip differences in these swapped cherries.
An array of reshuffled cherry tip differences (of the same size as the real one) is thus obtained. 
Finally, the test reports the sign test between the reshuffled and the real values.

Extended version processes tip cherries and internal cherries separately to avoid mixing
different biological interpretations (sampling vs transmission times).

The test therefore reports a probability of partner notification being present in the tree.""")
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input forest file in newick or nexus format")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    pval, n_cherries = bdei_test(forest)

    logging.info(f"BDEI test {pval} on {n_cherries} cherries.")

    with open(params.log, 'w+') as f:
        f.write('BDEI-test p-value\tnumber of cherries\n')
        f.write(f'{pval:g}\t{n_cherries}\n')


if __name__ == '__main__':
    main()




