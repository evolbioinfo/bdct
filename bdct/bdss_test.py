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

    :param tree: ete3.Tree, the tree of interest
    :param include_polytomies: bool, whether to include nodes with > 2 children
    :return: iterator of ParentChildrenTriplet motifs
    """
    for node in tree.traverse():
        if node.is_leaf():
            continue

        # Get all children (both internal nodes and leaves)
        children = node.children

        if len(children) < 2:
            continue

        if not include_polytomies and len(children) != 2:
            continue

        # For binary nodes, create triplet with parent and 2 children
        if len(children) == 2:
            yield ParentChildrenTriplet(node, children[0], children[1])
        # For polytomies, create triplet with parent and 3+ children
        elif len(children) >= 3 and include_polytomies:
            # Take first 3 children for simplicity, or create multiple triplets
            yield ParentChildrenTriplet(node, children[0], children[1], children[2])


def ss_test(forest):
    """
    Tests if the input forest was generated under a superspreading model.

    The test detects triplets in the forest and sorts them by the times of their parents.
    For each triplet, it calculates the length differences between sibling branches.
    It then generates reshuffled triplets by swapping children between neighboring triplets
    and compares the original vs reshuffled length differences using a sign test.

    The test reports a probability of superspreading being present in the tree.

    :param forest: list of trees
    :return: pval, n_triplets
    """
    annotate_forest_with_time(forest)

    all_triplets = []
    for tree in forest:
        all_triplets.extend(pick_triplets(tree, include_polytomies=True))

    # Sort triplets by parent node time
    all_triplets = sorted(all_triplets, key=lambda t: getattr(t.parent, TIME))

    n_triplets = len(all_triplets)
    logging.info(f'Picked {n_triplets} triplets.')

    if n_triplets < 2:
        return 1, n_triplets

    random_diffs, real_diffs = get_real_vs_reshuffled_triplet_diffs(all_triplets)

    # Sign test: if superspreading is present, real differences should be larger
    # (superspreaders create more heterogeneous branch lengths)
    pval = scipy.stats.binomtest((real_diffs > random_diffs).sum(),
                                 n=n_triplets, p=0.5, alternative='less').pvalue

    return pval, n_triplets


def get_real_vs_reshuffled_triplet_diffs(all_triplets):
    """
    Calculate real vs reshuffled branch length differences for triplets.

    :param all_triplets: list of ParentChildrenTriplet objects
    :return: random_diffs, real_diffs (numpy arrays)
    """
    n_triplets = len(all_triplets)

    # Calculate real differences
    real_diffs = np.zeros(n_triplets, dtype=float)
    child1_lengths = np.zeros(n_triplets, dtype=float)
    child2_lengths = np.zeros(n_triplets, dtype=float)

    for i, triplet in enumerate(all_triplets):
        # Get branch lengths (dist attribute represents branch length)
        c1_len = triplet.children[0].dist
        c2_len = triplet.children[1].dist

        child1_lengths[i] = c1_len
        child2_lengths[i] = c2_len

        # Calculate absolute difference between sibling branch lengths
        real_diffs[i] = abs(c1_len - c2_len)

    # Generate reshuffled differences by swapping children between neighboring triplets
    if n_triplets > 1:
        reshuffled_child2_lengths = np.zeros(n_triplets, dtype=float)

        # Swap pairs of second children between neighboring triplets
        reshuffled_child2_lengths[:-1:2] = child2_lengths[1::2]  # even indices get odd values
        reshuffled_child2_lengths[1::2] = child2_lengths[:-1:2]  # odd indices get even values

        # Handle odd number of triplets by cycling last 3
        if n_triplets % 2:
            reshuffled_child2_lengths[-1] = reshuffled_child2_lengths[-2]
            reshuffled_child2_lengths[-2] = child2_lengths[-1]
    else:
        reshuffled_child2_lengths = child2_lengths.copy()

    # Calculate reshuffled differences
    random_diffs = np.abs(child1_lengths - reshuffled_child2_lengths)

    return random_diffs, real_diffs


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