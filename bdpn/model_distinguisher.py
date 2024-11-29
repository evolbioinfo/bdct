import logging

import numpy as np
import scipy

from bdpn.tree_manager import TIME, read_forest, annotate_forest_with_time

DEFAULT_CHERRY_BLOCK_SIZE = 100

DEFAULT_NEIGHBOURHOOD_SIZE = 5

RANDOM_REPETITIONS = 1e3

DEFAULT_PERCENTILE = 0.25


class CherryLikeMotif(object):

    def __init__(self, clustered_tips, root=None):
        """
        A motif with exceptions.
        :param root: ete.TreeNode, the root of the motif subtree
        :param clustered_tips: list of clustered child tips
        """
        self.root = root # if root else clustered_tips[0].get_common_ancestor(*clustered_tips)
        self.clustered_tips = clustered_tips

    def __str__(self):
        return "Motif with root {} and {} clustered tips ({})" \
            .format(self.root.name, len(self.clustered_tips), ', '.join(_.name for _ in self.clustered_tips))


def pick_cherries(tree, include_polytomies=True):
    """
    Picks cherries that satisfy the given values of pn_lookback, pn_delay in the given tree.

    :param include_polytomies: bool, whether to transform polytomies in the tree into cherries,
        i.e. a polytomy of n tips will be transformed into n(n-1) cherries.
    :param tree: ete3.Tree, the tree of interest
    :return: iterator of Motif motifs
    """
    for cherry_root in (set(tip.up for tip in tree) if not tree.is_leaf() else set()):
        if not include_polytomies and len(cherry_root.children) != 2:
            continue
        tips = sorted([_ for _ in cherry_root.children if _.is_leaf()], key=lambda _: _.dist)
        if len(tips) < 2:
            continue
        yield CherryLikeMotif(clustered_tips=tips, root=cherry_root)


def sign_test(forest):
    """
    Tests if the input forest was generated under a -PN model.

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

    The test therefore reports a probability of partner notification
    being present in the tree.

    :param forest: list of trees
    :return: pval
    """
    annotate_forest_with_time(forest)

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=False))
    all_cherries = sorted(all_cherries, key=lambda _: getattr(_.root, TIME))

    n_cherries = len(all_cherries)
    logging.info('Picked {} cherries with {} roots.'.format(n_cherries, len({_.root for _ in all_cherries})))

    def get_diff(b1, b2):
        return abs(b1.dist - b2.dist)

    def get_diff_array(cherries):
        return np.array([get_diff(*cherry.clustered_tips) for cherry in cherries])

    first_tips, other_tips = [], []
    for cherry in all_cherries:
        t1, t2 = cherry.clustered_tips
        if np.random.rand() < 0.5:
            t2, t1 = t1, t2
        first_tips.append(t1)
        other_tips.append(t2)

    reshuffled_cherries = []
    for i in range(n_cherries):
        other_tip_i = i + (-1 if i % 2 else 1)
        if n_cherries % 2 and i >= n_cherries - 3:
            other_tip_i = (i + 1) if (i < n_cherries - 1) else (i - 2)
        reshuffled_cherries.append(CherryLikeMotif(clustered_tips=[first_tips[i], other_tips[other_tip_i]]))

    real_diffs, random_diffs = get_diff_array(all_cherries), get_diff_array(reshuffled_cherries)
    k = (random_diffs < real_diffs).sum()

    result = scipy.stats.binomtest(k, n=n_cherries, p=0.5, alternative='less').pvalue

    # logging.info(
    #     'For {n} cherries with roots in [{start}-{stop}), PN test {res}.'
    #     .format(res=result, start=getattr(all_cherries[0].root, TIME), stop=getattr(all_cherries[-1].root, TIME),
    #             n=n_cherries))

    return result


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
        b1, b2 = cherry.clustered_tips
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
    Entry point for PN test with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="""PN-test.
        
Checks if the input forest was generated under a -PN model.
    
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
    pval = sign_test(forest)

    logging.info("PN test {}.".format(pval))

    with open(params.log, 'w+') as f:
        f.write('PN-test\t{}\n'.format(pval))


if __name__ == '__main__':
    main()
