import logging
import numpy as np
import scipy.stats
from bdct.tree_manager import TIME, read_forest, annotate_forest_with_time

DEFAULT_PERCENTILE = 0.25


class ParentChildrenTriplet(object):
    """
    A triplet motif that includes a parent node and its children for superspreading detection.
    """

    def __init__(self, parent, child1, child2, child3=None):
        """
        A motif for superspreading detection using triplets.

        :param parent: ete.TreeNode, the parent node of the triplet
        :param child1: ete.TreeNode, first child branch
        :param child2: ete.TreeNode, second child branch
        :param child3: ete.TreeNode, optional third child branch for polytomies
        """
        self.parent = parent
        self.children = [child1, child2]
        if child3 is not None:
            self.children.append(child3)

    def __str__(self):
        child_names = ', '.join(child.name if hasattr(child, 'name') else str(id(child)) for child in self.children)
        parent_name = self.parent.name if hasattr(self.parent, 'name') else str(id(self.parent))
        return f"Triplet with parent {parent_name} and children: {child_names}"

    def __len__(self):
        return len(self.children)


def pick_triplets(tree, include_polytomies=True):
    """
    Picks triplets in the given tree for superspreading detection.
    Only considers triplets where both children are internal nodes (not leaves).

    :param tree: ete3.Tree, the tree of interest
    :param include_polytomies: bool, whether to include nodes with > 2 children
    :return: iterator of ParentChildrenTriplet motifs
    """
    for node in tree.traverse():
        if node.is_leaf():
            continue

        # Get only internal node children (exclude leaves/tips)
        internal_children = [child for child in node.children if not child.is_leaf()]

        if len(internal_children) < 2:
            continue

        if not include_polytomies and len(internal_children) != 2:
            continue

        # For binary internal nodes, create triplet with parent and 2 internal children
        if len(internal_children) == 2:
            yield ParentChildrenTriplet(node, internal_children[0], internal_children[1])
        # For polytomies, create triplet with parent and 3+ internal children
        elif len(internal_children) >= 3 and include_polytomies:
            # Take first 3 internal children
            yield ParentChildrenTriplet(node, internal_children[0], internal_children[1], internal_children[2])


def ss_test(forest):
    """
    Tests if the input forest was generated under a superspreading model.
    Implements the test as described in the theoretical design: compares
    parent-child branch differences before and after reshuffling parents
    between temporally close triplets.

    :param forest: list of trees
    :return: p-value, number of triplets
    """
    annotate_forest_with_time(forest)

    all_triplets = []
    for tree in forest:
        all_triplets.extend(pick_triplets(tree, include_polytomies=True))

    all_triplets = sorted(all_triplets, key=lambda t: getattr(t.parent, TIME))
    n_triplets = len(all_triplets)
    logging.info(f'Picked {n_triplets} triplets.')

    if n_triplets < 2:
        return 1.0, n_triplets

    real_diffs, reshuffled_diffs = get_real_vs_reshuffled_triplet_diffs(all_triplets)

    flat_real = np.array([v for pair in real_diffs for v in pair])
    flat_resh = np.array([v for pair in reshuffled_diffs for v in pair])

    # Sign test: are real_diffs systematically smaller than reshuffled_diffs?
    count = np.sum(flat_real < flat_resh)
    total = len(flat_real)

    pval = scipy.stats.binomtest(count, n=total, p=0.5, alternative='less').pvalue

    return pval, n_triplets


def get_real_vs_reshuffled_triplet_diffs(triplets):
    """
    Computes parent-child differences for each triplet, and for reshuffled ones.

    Reshuffling swaps the parent nodes between adjacent triplets (sorted by time).
    The test uses differences between each parent and both children.

    :param triplets: list of ParentChildrenTriplet
    :return: (real_diffs, reshuffled_diffs) as lists of (d1, d2)
    """
    real_diffs = []
    reshuffled_diffs = []

    # Calculate real differences
    for t in triplets:
        d1 = abs(t.parent.dist - t.children[0].dist)
        d2 = abs(t.parent.dist - t.children[1].dist)
        real_diffs.append((d1, d2))

    # Reshuffle parents between adjacent triplets
    n = len(triplets)
    reshuffled_triplets = []

    if n % 2 == 0:
        # Even case: pairwise swap
        for i in range(0, n, 2):
            t1, t2 = triplets[i], triplets[i + 1]
            reshuffled_triplets.append((t2.parent, t1.children))
            reshuffled_triplets.append((t1.parent, t2.children))
    else:
        # Odd case: pairwise swap for n-3, then cycle last 3
        for i in range(0, n - 3, 2):
            t1, t2 = triplets[i], triplets[i + 1]
            reshuffled_triplets.append((t2.parent, t1.children))
            reshuffled_triplets.append((t1.parent, t2.children))

        # Last 3 triplets: cycle parents
        tA, tB, tC = triplets[-3], triplets[-2], triplets[-1]
        pA, pB, pC = tA.parent, tB.parent, tC.parent
        cA, cB, cC = tA.children, tB.children, tC.children
        reshuffled_triplets.append((pC, cA))
        reshuffled_triplets.append((pA, cB))
        reshuffled_triplets.append((pB, cC))

    # Compute reshuffled differences
    for parent, children in reshuffled_triplets:
        d1 = abs(parent.dist - children[0].dist)
        d2 = abs(parent.dist - children[1].dist)
        reshuffled_diffs.append((d1, d2))

    assert len(real_diffs) == len(reshuffled_diffs), \
        f"Real and reshuffled lengths do not match: {len(real_diffs)} vs {len(reshuffled_diffs)}"

    return real_diffs, reshuffled_diffs


def triplet_diff_plot(forest, outfile=None):
    """
    Plots triplet branch length differences against parent times.
    Requires matplotlib and seaborn installed.

    :param forest: list of trees
    :param outfile: (optional) output file where the plot should be saved.
        If not specified, the plot will be shown instead.
    :return: void
    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import show
        import seaborn as sns
    except ImportError:
        logging.error("matplotlib and seaborn are required for plotting")
        return

    annotate_forest_with_time(forest)

    all_triplets = []
    for tree in forest:
        all_triplets.extend(pick_triplets(tree, include_polytomies=False))

    def get_diff(triplet):
        c1, c2 = triplet.children[0], triplet.children[1]
        return abs(c1.dist - c2.dist)

    plt.clf()
    x = np.array([getattr(t.parent, TIME) for t in all_triplets])
    diffs = np.array([get_diff(t) for t in all_triplets])

    if len(diffs) == 0:
        logging.warning("No triplets found for plotting")
        return

    perc = np.percentile(diffs, [25, 50, 75])
    mask = np.digitize(diffs, perc)
    colors = sns.color_palette("colorblind")

    for i, label in zip(range(4), ('1st', '2nd', '3rd', '4th')):
        indices = mask == i
        if np.any(indices):
            ax = sns.scatterplot(x=x[indices], y=diffs[indices], alpha=0.75,
                                 label=f'{label} quantile', color=colors[i])

    ax.set_xlabel('triplet parent time')
    ax.set_ylabel('sibling branch length difference')
    ax.legend()
    plt.tight_layout()

    if not outfile:
        show()
    else:
        plt.savefig(outfile, dpi=300)


def main():
    """
    Entry point for SS (superspreading) test with command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="""SS-test.

Checks if the input forest was generated under a superspreading model.

The test detects triplets in the forest and sorts them by the times of their parents.
For each triplet the test calculates the difference between sibling branch lengths,
hence obtaining an array of real branch length differences.
It then generates a collection of reshuffled branch length differences of the same size:
Processing the triplets in pairs from the two triplets with the oldest parents
to the two (three if the total number of triplets is odd) triplets with the most recent parents,
we swap one child per triplet between neighboring triplets. We then calculate the branch length 
differences in these swapped triplets.
An array of reshuffled branch length differences (of the same size as the real one) is thus obtained.
Finally, the test reports the sign test between the real and the reshuffled values.

The test therefore reports a probability of superspreading being present in the tree.""")

    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input forest file in newick or nexus format")
    params = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    forest = read_forest(params.nwk)
    pval, n_triplets = ss_test(forest)

    logging.info(f"SS test p-value: {pval} on {n_triplets} triplets.")

    with open(params.log, 'w+') as f:
        f.write('SS-test p-value\tnumber of triplets\n')
        f.write(f'{pval:g}\t{n_triplets}\n')


if __name__ == '__main__':
    main()