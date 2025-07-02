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

class TripletMotif(object):
    def __init__(self, parent_node, child_node, grandparent_node):
        """
        A motif representing a child node, its parent and its grandparent.
        """
        self.parent_node = parent_node
        self.child_node = child_node
        self.grandparent_node = grandparent_node

    def __str__(self):
        return (f"Triplet with parent {self.parent_node.name} (length {self.parent_node.dist}), "
                f"child {self.child_node.name} (length {self.child_node.dist}), "
                f"and grandparent {self.grandparent_node.name} (length {self.grandparent_node.dist})")


def pick_cherries(tree, include_polytomies=True, external=True):
    """
    Picks cherries in the given tree.

    :param include_polytomies: bool, whether to include nodes with > 2 children into consideration.
    :param tree: ete3.Tree, the tree of interest
    :return: iterator of Motif motifs
    """
    roots = (set(tip.up for tip in tree) if external else set(n for n in tree.traverse() if not n.is_leaf())) \
        if not tree.is_leaf() else set()
    for cherry_root in roots:
        if not include_polytomies and len(cherry_root.children) != 2:
            continue
        tips = [_ for _ in cherry_root.children if external == _.is_leaf()]
        if len(tips) < 2:
            continue
        yield ParentChildrenMotif(clustered_children=tips, root=cherry_root)


def ct_test(forest):
    """
    Tests if the input forest was generated under a -CT model.

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
    n_cherries, n_less = 0, 0

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=True))


    if len(all_cherries) > 2:
        all_cherries = sorted(all_cherries, key=lambda _: getattr(_.root, TIME))
        n_less, n_cherries = get_real_vs_reshuffled_diffs_less(all_cherries)

    pval = scipy.stats.binomtest(n_less, n=n_cherries, p=0.5, alternative='less').pvalue

    return pval, n_cherries

def bdei_test(forest):
    """
    Tests if the input forest was generated under a model with incubation.

    :param forest: list of trees
    :return: pval
    """
    annotate_forest_with_time(forest)
    n_cherries, n_less = 0, 0

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=True, external=False))

    # print(np.array([is_motif_ei(all_cherries, i, n_neighbours=4) for i in range(len(all_cherries))]).sum() / len(all_cherries))

    if len(all_cherries) > 2:
        all_cherries = sorted(all_cherries, key=lambda _: getattr(_.root, TIME))
        n_less, n_cherries = get_real_vs_reshuffled_diffs_less(all_cherries)

    pval = scipy.stats.binomtest(n_less, n=n_cherries, p=0.5, alternative='greater').pvalue

    return pval, (n_less, n_cherries)


def is_motif_ei(motifs, i, n_neighbours=3):
    motif = motifs[i]
    time = getattr(motif.root, TIME)
    neighbours = [m for m in motifs[max(0, i - (n_neighbours * 2)): min(len(motifs), i + 1 + n_neighbours * 2)] \
                      if m.root != motif.root]
    neighbours = sorted(neighbours,
                        key=lambda m: abs(getattr(m.root, TIME) - time))[:n_neighbours]

    first_dists, other_dists = np.zeros(n_neighbours, dtype=float), np.zeros(n_neighbours, dtype=float)
    for i, couple in enumerate(neighbours):
        t1, t2 = np.random.choice(couple.clustered_children, size=2, replace=False)
        first_dists[i] = t1.dist
        other_dists[i] = t2.dist

    t1, t2 = np.random.choice(motif.clustered_children, size=2, replace=False)
    real_diff = np.abs(t1.dist - t2.dist)

    first_vs_first_diffs = np.abs(t1.dist - first_dists)
    first_vs_other_diffs = np.abs(t1.dist - other_dists)
    other_vs_first_diffs = np.abs(t2.dist - first_dists)
    other_vs_other_diffs = np.abs(t2.dist - other_dists)

    n_less = (first_vs_first_diffs < real_diff).sum() + (first_vs_other_diffs < real_diff).sum() \
                + (other_vs_first_diffs < real_diff).sum() + (other_vs_other_diffs < real_diff).sum()
    n_total = len(first_vs_first_diffs) * 4
    return scipy.stats.binomtest(n_less, n=n_total, p=0.5, alternative='greater').pvalue < 0.1

def get_real_vs_reshuffled_diffs_less(all_couples):
    n_motifs = len(all_couples)
    first_dists, other_dists = np.zeros(n_motifs, dtype=float), np.zeros(n_motifs, dtype=float)

    root_times = np.array([getattr(c.root, TIME) for c in all_couples])
    diff_right = np.concatenate((np.abs(root_times[:-1] - root_times[1:]), [np.inf]))
    diff_left = np.concatenate(([np.inf], np.abs(root_times[1:] - root_times[:-1])))

    left_is_closer = diff_left < diff_right

    for i, couple in enumerate(all_couples):
        t1, t2 = np.random.choice(couple.clustered_children, size=2, replace=False)
        first_dists[i] = t1.dist
        other_dists[i] = t2.dist

    real_diffs = np.abs(first_dists - other_dists)
    first_vs_first_diffs = np.abs(first_dists[:-1] - first_dists[1:])
    first_vs_other_diffs = np.abs(first_dists[:-1] - other_dists[1:])

    other_vs_other_diffs = np.abs(other_dists[:-1] - other_dists[1:])
    other_vs_first_diffs = np.abs(other_dists[:-1] - first_dists[1:])

    n_less = np.where(left_is_closer, np.concatenate(([False], first_vs_first_diffs < real_diffs[1:])), np.concatenate((first_vs_first_diffs < real_diffs[:-1], [False]))).sum() \
             + np.where(left_is_closer, np.concatenate(([False], first_vs_other_diffs < real_diffs[1:])), np.concatenate((first_vs_other_diffs < real_diffs[:-1], [False]))).sum() \
             + np.where(left_is_closer, np.concatenate(([False], other_vs_other_diffs < real_diffs[1:])), np.concatenate((other_vs_other_diffs < real_diffs[:-1], [False]))).sum() \
             + np.where(left_is_closer, np.concatenate(([False], other_vs_first_diffs < real_diffs[1:])), np.concatenate((other_vs_first_diffs < real_diffs[:-1], [False]))).sum()
    n_total = n_motifs * 4



    # n_less = (first_vs_first_diffs < real_diffs[:-1]).sum() + (first_vs_first_diffs < real_diffs[1:]).sum() \
    #  + (first_vs_other_diffs < real_diffs[:-1]).sum() + (first_vs_other_diffs < real_diffs[1:]).sum() \
    #  + (other_vs_other_diffs < real_diffs[:-1]).sum() + (other_vs_other_diffs < real_diffs[1:]).sum() \
    #  + (other_vs_first_diffs < real_diffs[:-1]).sum() + (other_vs_first_diffs < real_diffs[1:]).sum()
    #
    # n_total = len(first_vs_first_diffs) * 8

    return n_less, n_total


def get_start_time(node):
    """
    Returns the start time of the node's branch.
    :param node: ete3.TreeNode
    :return: float, start time
    """
    return getattr(node, TIME) - node.dist

def calc_avg_dist(inodes, n_neighbours=6):
    inode2avg_dist = {}

    def do_not_intersect(n1, n2):
        def get_family(n):
            res = {n}
            if not n.is_root():
                res.add(n.up)
                if not n.up.is_root():
                    res.add(n.up.up)
            return res

        return not (get_family(n1) & get_family(n2))


    for i in range(len(inodes)):
        inode = inodes[i]
        itime = get_start_time(inode)
        # 15 internal nodes (including n) in a binary tree can have an intersecting family with node n
        neighbours = [n for n in inodes[max(0, i - (15 + n_neighbours)): min(len(inodes), i + 15 + n_neighbours)] \
                      if do_not_intersect(inode, n)]
        neighbours = sorted(neighbours,
                            key=lambda n: abs(get_start_time(n) - itime))[:n_neighbours]
        inode2avg_dist[inode] = np.mean([n.dist for n in neighbours])
    return inode2avg_dist


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


    inodes = [n for n in forest[0].traverse() if not n.is_leaf() and not n.is_root()]
    inodes = sorted(inodes, key=get_start_time)
    inode2avg_dist = calc_avg_dist(inodes)

    all_triplets = []
    # remove first 5 as they do not have many neighbours
    for node in inodes[5:]:
        for child in node.children:
            if not child.is_leaf():
                all_triplets.append(TripletMotif(parent_node=node,
                                                 child_node=child,
                                                 grandparent_node=node.up))

    n_motifs = len(all_triplets)
    logging.info(f'Picked {n_motifs} triplets.')

    if n_motifs < 2:
        logging.warning("Not enough triplets to perform BDSS test (need at least 2). Returning p-value of 1.")
        return 1.0, n_motifs

    child_comparison = np.array([triplet.child_node.dist < inode2avg_dist[triplet.child_node] for triplet in all_triplets])
    parent_comparison = np.array([triplet.parent_node.dist < inode2avg_dist[triplet.parent_node] for triplet in all_triplets])
    grandparent_comparison = np.array([triplet.grandparent_node.dist < inode2avg_dist[triplet.grandparent_node] for triplet in all_triplets])

    both_compatible = child_comparison & parent_comparison
    all_compatible = both_compatible & grandparent_comparison
    child_grandpa_compatible = child_comparison & grandparent_comparison

    # prob_less_child = child_comparison.sum() / n_motifs
    prob_less_parent = parent_comparison.sum() / n_motifs
    # prob_less_grandpa = grandparent_comparison.sum() / n_motifs
    # print(prob_less, prob_less_child, prob_less_parent, prob_less_grandpa)

    pval = scipy.stats.binomtest(int(all_compatible.sum()),
                                 n=int(child_grandpa_compatible.sum()), p=prob_less_parent, alternative='greater').pvalue

    return pval


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
    Entry point for CT test with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="""CT-test.
        
Checks if the input forest was generated under a -CT model.
    
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
    parser.add_argument('--log', default='/home/azhukova/projects/bdct/simulations/BDCT1000/tree.0.bdeitest', type=str, help="output log file")
    parser.add_argument('--nwk', default='/home/azhukova/projects/bdct/simulations/BDSSCT0/tree.3.nwk', type=str, help="input forest file in newick or nexus format")
    params = parser.parse_args()

    pvals_bdss = np.zeros(100, dtype=float)
    pvals_bdei = np.zeros(100, dtype=float)
    pvals_ct = np.zeros(100, dtype=float)

    result_table = np.zeros((7, 8), dtype=int)

    models = ('BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT')
    for mi, model in enumerate(models):
        print(f"Model: {model}")

        for i in range(100):
            forest = read_forest(f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/500_1000/{model}/tree.{i}.nwk')

            pvals_bdss[i] = bdss_test(forest)
            pvals_bdei[i] = bdei_test(forest)[0]
            pvals_ct[i] = ct_test(forest)[0]

        # for label, pvals in (('CT test', pvals_ct), ('BDEI test', pvals_bdei),
        #                      ('BDSS test', pvals_bdss),
        #                      # ('BDSS or BDEI test', np.minimum(pvals_bdss, pvals_bdei) ),
        #                      # ('BDSS and BDEI test', np.maximum(pvals_bdss, pvals_bdei) )
        #                      ):
        #     print(f"{label}:\t{sum(pvals < 0.05)} {pvals.min()} {pvals.mean()} {np.median(pvals)}")

        result_table[:, mi] = [(pvals_bdei < 0.05).sum(), (pvals_bdss < 0.05).sum(), (pvals_ct < 0.05).sum(), \
                              ((pvals_bdei < 0.05) & (pvals_bdss < 0.05)).sum(), \
                              ((pvals_bdei < 0.05) & (pvals_ct < 0.05)).sum(), \
                              ((pvals_bdss < 0.05) & (pvals_ct < 0.05)).sum(), \
                              ((pvals_bdei < 0.05) & (pvals_ct < 0.05) & (pvals_bdss < 0.05)).sum()]

        # print('-----------------------------------------------\n')
    print('\t' + '\t'.join(models))
    print('EI\t', '\t'.join([f"{x:d}" for x in result_table[0, :]]))
    print('SS\t', '\t'.join([f"{x:d}" for x in result_table[1, :]]))
    print('CT\t', '\t'.join([f"{x:d}" for x in result_table[2, :]]))
    print('EISS\t', '\t'.join([f"{x:d}" for x in result_table[3, :]]))
    print('EICT\t', '\t'.join([f"{x:d}" for x in result_table[4, :]]))
    print('SSCT\t', '\t'.join([f"{x:d}" for x in result_table[5, :]]))
    print('EISSCT\t', '\t'.join([f"{x:d}" for x in result_table[6, :]]))


if __name__ == '__main__':
    main()
