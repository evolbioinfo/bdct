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
            tips = [child for child in node.children if child.is_leaf()]
            if len(tips) >= 2:
                yield ParentChildrenMotif(clustered_children=tips, root=node)

        elif cherry_type == "internal":
            internal_nodes = [child for child in node.children if not child.is_leaf()]
            if len(internal_nodes) >= 2:
                yield ParentChildrenMotif(clustered_children=internal_nodes, root=node)


def bdei_test(forest, cherry_strategy="both"):
    """
    Tests if the input forest was generated under a -BDEI model.

    The test detects cherries in the forest and sorts them by the times of their roots.
    For each cherry the test calculates the difference between its tip times,
    hence obtaining an array of cherry tip differences.

    Extended version: allows testing different cherry strategies to compare performance.

    :param forest: list of trees
    :param cherry_strategy: "tips_only", "internal_only", or "both"
    :return: pval, n_cherries
    """
    annotate_forest_with_time(forest)

    c_tips = np.array([], dtype=bool)
    c_inodes = np.array([], dtype=bool)
    n_tip_cherries = 0
    n_internal_cherries = 0

    if cherry_strategy in ["tips_only", "both"]:
        tip_cherries = []
        for tree in forest:
            tip_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="tip"))
        tip_cherries = sorted(tip_cherries, key=lambda _: getattr(_.root, TIME))
        n_tip_cherries = len(tip_cherries)

        if n_tip_cherries >= 2:
            random_diffs_tips, real_diffs_tips = get_real_vs_reshuffled_diffs(tip_cherries)
            c_tips = random_diffs_tips > real_diffs_tips

    if cherry_strategy in ["internal_only", "both"]:
        internal_cherries = []
        for tree in forest:
            internal_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="internal"))
        internal_cherries = sorted(internal_cherries, key=lambda _: getattr(_.root, TIME))
        n_internal_cherries = len(internal_cherries)

        if n_internal_cherries >= 2:
            random_diffs_internal, real_diffs_internal = get_real_vs_reshuffled_diffs(internal_cherries)
            c_inodes = random_diffs_internal > real_diffs_internal

    if cherry_strategy == "tips_only":
        c_combined = c_tips
        n_total_cherries = n_tip_cherries
    elif cherry_strategy == "internal_only":
        c_combined = c_inodes
        n_total_cherries = n_internal_cherries
    else:
        c_combined = np.concatenate([c_tips, c_inodes])
        n_total_cherries = n_tip_cherries + n_internal_cherries

    logging.info(
        f'Strategy: {cherry_strategy}. Picked {n_total_cherries} cherries total: {n_tip_cherries} tip, {n_internal_cherries} internal.')

    if len(c_combined) < 2:
        return 1, n_total_cherries

    pval = scipy.stats.binomtest(c_combined.sum(), n=len(c_combined), p=0.5, alternative='less').pvalue

    return pval, n_total_cherries


def get_real_vs_reshuffled_diffs(all_couples):
    """
    Generate real vs reshuffled differences for cherries.

    Enhanced version: instead of one random swap per cherry pair,
    we do ALL possible swaps between neighboring cherries:
    - child1 vs child1'
    - child1 vs child2'
    - child2 vs child1'
    - child2 vs child2'

    This gives 4x more examples for better statistical power.
    """
    n_motifs = len(all_couples)

    all_children = []
    for couple in all_couples:
        children = list(couple.clustered_children)
        all_children.append(children)

    real_diffs = []
    random_diffs = []

    if n_motifs > 1:
        for i in range(0, n_motifs - 1, 2):
            cherry1_children = all_children[i]
            cherry2_children = all_children[i + 1]

            if len(cherry1_children) < 2 or len(cherry2_children) < 2:
                continue

            c1_child1, c1_child2 = cherry1_children[0], cherry1_children[1]
            c2_child1, c2_child2 = cherry2_children[0], cherry2_children[1]

            real_diff_1 = abs(c1_child1.dist - c1_child2.dist)
            real_diff_2 = abs(c2_child1.dist - c2_child2.dist)
            real_diffs.extend([real_diff_1, real_diff_2])

            swap_diff_1 = abs(c1_child1.dist - c2_child1.dist)
            swap_diff_2 = abs(c1_child1.dist - c2_child2.dist)
            swap_diff_3 = abs(c1_child2.dist - c2_child1.dist)
            swap_diff_4 = abs(c1_child2.dist - c2_child2.dist)

            avg_swap_1 = (swap_diff_1 + swap_diff_2) / 2
            avg_swap_2 = (swap_diff_3 + swap_diff_4) / 2

            random_diffs.extend([avg_swap_1, avg_swap_2])

        if n_motifs % 2 == 1 and n_motifs >= 3:
            last_3_children = all_children[-3:]

            if all(len(children) >= 2 for children in last_3_children):
                c1_child1, c1_child2 = last_3_children[0][0], last_3_children[0][1]
                c2_child1, c2_child2 = last_3_children[1][0], last_3_children[1][1]
                c3_child1, c3_child2 = last_3_children[2][0], last_3_children[2][1]


def get_real_vs_reshuffled_diffs(all_couples):
    """
    Generate real vs reshuffled differences for cherries.

    Enhanced version: instead of averaging, we make 8 individual comparisons
    per cherry pair as suggested:

    For each pair of cherries (cherry1, cherry2):
    1. |tip1.1 - tip1.2| against |tip1.1 - tip2.1|
    2. |tip1.1 - tip1.2| against |tip1.1 - tip2.2|
    3. |tip1.1 - tip1.2| against |tip1.2 - tip2.1|
    4. |tip1.1 - tip1.2| against |tip1.2 - tip2.2|
    5. |tip2.1 - tip2.2| against |tip1.1 - tip2.1|
    6. |tip2.1 - tip2.2| against |tip1.1 - tip2.2|
    7. |tip2.1 - tip2.2| against |tip1.2 - tip2.1|
    8. |tip2.1 - tip2.2| against |tip1.2 - tip2.2|

    This gives 8 comparisons per cherry pair for better statistical power.
    """
    n_motifs = len(all_couples)

    all_children = []
    for couple in all_couples:
        children = list(couple.clustered_children)
        all_children.append(children)

    real_diffs = []
    random_diffs = []

    if n_motifs > 1:
        for i in range(0, n_motifs - 1, 2):
            cherry1_children = all_children[i]
            cherry2_children = all_children[i + 1]

            if len(cherry1_children) < 2 or len(cherry2_children) < 2:
                continue

            c1_child1, c1_child2 = cherry1_children[0], cherry1_children[1]
            c2_child1, c2_child2 = cherry2_children[0], cherry2_children[1]

            real_diff_1 = abs(c1_child1.dist - c1_child2.dist)
            real_diff_2 = abs(c2_child1.dist - c2_child2.dist)

            cross_diff_1 = abs(c1_child1.dist - c2_child1.dist)
            cross_diff_2 = abs(c1_child1.dist - c2_child2.dist)
            cross_diff_3 = abs(c1_child2.dist - c2_child1.dist)
            cross_diff_4 = abs(c1_child2.dist - c2_child2.dist)

            real_diffs.extend([real_diff_1, real_diff_1, real_diff_1, real_diff_1])
            random_diffs.extend([cross_diff_1, cross_diff_2, cross_diff_3, cross_diff_4])

            real_diffs.extend([real_diff_2, real_diff_2, real_diff_2, real_diff_2])
            random_diffs.extend([cross_diff_1, cross_diff_2, cross_diff_3, cross_diff_4])

        if n_motifs % 2 == 1 and n_motifs >= 3:
            last_3_children = all_children[-3:]

            if all(len(children) >= 2 for children in last_3_children):
                c1_child1, c1_child2 = last_3_children[0][0], last_3_children[0][1]
                c2_child1, c2_child2 = last_3_children[1][0], last_3_children[1][1]
                c3_child1, c3_child2 = last_3_children[2][0], last_3_children[2][1]

                real_diff_1 = abs(c1_child1.dist - c1_child2.dist)
                real_diff_2 = abs(c2_child1.dist - c2_child2.dist)
                real_diff_3 = abs(c3_child1.dist - c3_child2.dist)

                cross_1_2a = abs(c1_child1.dist - c2_child1.dist)
                cross_1_2b = abs(c1_child1.dist - c2_child2.dist)
                cross_1_2c = abs(c1_child2.dist - c2_child1.dist)
                cross_1_2d = abs(c1_child2.dist - c2_child2.dist)

                cross_2_3a = abs(c2_child1.dist - c3_child1.dist)
                cross_2_3b = abs(c2_child1.dist - c3_child2.dist)
                cross_2_3c = abs(c2_child2.dist - c3_child1.dist)
                cross_2_3d = abs(c2_child2.dist - c3_child2.dist)

                cross_3_1a = abs(c3_child1.dist - c1_child1.dist)
                cross_3_1b = abs(c3_child1.dist - c1_child2.dist)
                cross_3_1c = abs(c3_child2.dist - c1_child1.dist)
                cross_3_1d = abs(c3_child2.dist - c1_child2.dist)

                real_diffs.extend([real_diff_1] * 4)
                random_diffs.extend([cross_1_2a, cross_1_2b, cross_1_2c, cross_1_2d])

                real_diffs.extend([real_diff_2] * 4)
                random_diffs.extend([cross_1_2a, cross_1_2b, cross_1_2c, cross_1_2d])

                real_diffs.extend([real_diff_2] * 4)
                random_diffs.extend([cross_2_3a, cross_2_3b, cross_2_3c, cross_2_3d])

                real_diffs.extend([real_diff_3] * 4)
                random_diffs.extend([cross_2_3a, cross_2_3b, cross_2_3c, cross_2_3d])

                real_diffs.extend([real_diff_3] * 4)
                random_diffs.extend([cross_3_1a, cross_3_1b, cross_3_1c, cross_3_1d])

                real_diffs.extend([real_diff_1] * 4)
                random_diffs.extend([cross_3_1a, cross_3_1b, cross_3_1c, cross_3_1d])

    else:
        if n_motifs == 1 and len(all_children[0]) >= 2:
            child1, child2 = all_children[0][0], all_children[0][1]
            real_diff = abs(child1.dist - child2.dist)
            real_diffs = [real_diff]
            random_diffs = [real_diff]

    return np.array(random_diffs), np.array(real_diffs)


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

    tip_cherries = []
    internal_cherries = []
    for tree in forest:
        tip_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="tip"))
        internal_cherries.extend(pick_cherries(tree, include_polytomies=True, cherry_type="internal"))

    def get_diff(cherry):
        b1, b2 = cherry.clustered_children
        return abs(b1.dist - b2.dist)

    plt.figure(figsize=(12, 6))

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
        argparse.ArgumentParser(description="""BDEI-test with Cherry Strategy Options.

Checks if the input forest was generated under a -BDEI model.
The test detects cherries in the forest and sorts them by the times of their roots. 

Extended version allows testing different cherry strategies:
- tips_only: Use only traditional tip cherries
- internal_only: Use only internal node cherries  
- both: Use both tip and internal cherries (processed separately)

This allows comparison of performance between different approaches.

The test therefore reports a probability of partner notification being present in the tree.""")
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input forest file in newick or nexus format")
    parser.add_argument('--strategy', type=str, default='both',
                        choices=['tips_only', 'internal_only', 'both'],
                        help="Cherry strategy: tips_only, internal_only, or both (default: both)")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    pval, n_cherries = bdei_test(forest, cherry_strategy=params.strategy)

    logging.info(f"BDEI test (strategy: {params.strategy}) p-value: {pval:.6f} on {n_cherries} cherries.")

    with open(params.log, 'w+') as f:
        f.write('BDEI-test p-value\tnumber of cherries\n')
        f.write(f'{pval:g}\t{n_cherries}\n')


if __name__ == '__main__':
    main()




