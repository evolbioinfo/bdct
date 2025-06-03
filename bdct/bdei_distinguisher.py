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


def pick_cherries(tree, include_polytomies=True):
    """
    Picks cherries in the given tree.

    :param include_polytomies: bool, whether to include nodes with > 2 children into consideration.
    :param tree: ete3.Tree, the tree of interest
    :return: iterator of Motif motifs
    """
    for cherry_root in (set(tip.up for tip in tree) if not tree.is_leaf() else set()):
        if not include_polytomies and len(cherry_root.children) != 2:
            continue
        tips = [_ for _ in cherry_root.children if _.is_leaf()]
        if len(tips) < 2:
            continue
        yield ParentChildrenMotif(clustered_children=tips, root=cherry_root)


def bdei_test(forest):
    """
    Tests if the input forest was generated under a -BDEI model.

    The test detects cherries in the forest and sorts them by the times of their roots.
    For each cherry the test calculates the difference between its tip times,
    hence obtaining an array of cherry tip differences.
    It then generates a collection of random cherry tip differences of the same size:
    It fixed one of the tips for each cherry and then swaps the other tips between neighbouring cherries,
    such that the other tip of cherry 2i is swapped with the other tip of cherry 2i + 1 (i = 0, 1, ...).
    (If the total number of cherries is odd, the last three cherries instead of the last two
    swap their other tips in a cycle). For each hence reshuffled cherry its tip difference is calculated.

    Finally, we calculate the sign test of one by one comparison of real vs reshuffled diffs
    (-1 if the difference for the i-th cherry is smaller in the real array, 1 if larger, 0 is the same).

    The test therefore reports a probability of contact tracing
    being present in the tree.

    :param forest: list of trees
    :return: pval
    """
    annotate_forest_with_time(forest)

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=True))
    all_cherries = sorted(all_cherries, key=lambda _: getattr(_.root, TIME))

    n_cherries = len(all_cherries)
    logging.info(f'Picked {n_cherries} cherries.')

    if n_cherries < 2:
        return 1, n_cherries

    random_diffs, real_diffs = get_real_vs_reshuffled_diffs(all_cherries)
    pval = scipy.stats.binomtest((random_diffs < real_diffs).sum(), n=n_cherries, p=0.75, alternative='less').pvalue  # >#

    return pval, n_cherries


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

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=False))

    def get_diff(cherry):
        b1, b2 = cherry.clustered_children
        return abs(b1.dist - b2.dist)

    plt.clf()
    x = np.array([getattr(_.root, TIME) for _ in all_cherries])
    diffs = np.array([get_diff(_) for _ in all_cherries])
    perc = np.percentile(diffs, [25, 50, 75])
    mask = np.digitize(diffs, perc)
    colors = sns.color_palette("colorblind")

    for i, label in zip(range(4), ('1st', '2nd', '3rd', '4th')):
        ax = sns.scatterplot(x=x[mask == i], y=diffs[mask == i], alpha=0.75,
                             label='{} quantile'.format(label), color=colors[i])
    # col = ax.collections[0]
    # y = col.get_offsets()[:, 1]
    # perc = np.percentile(y, [25, 50, 75])
    # col.set_array(np.digitize(y, perc))
    ax.set_xlabel('cherry root time')
    ax.set_ylabel('cherry tip time difference')
    ax.legend()
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
