import os
import re
from collections import Counter
from datetime import datetime

import numpy as np
from Bio import Phylo
from ete3 import Tree, TreeNode

OLDEST_TIP_TIME = 'oldest_tip_time'

EPSILON = 1e-6

TIME = 'time'

DATE_REGEX = r'[+-]*[\d]+[.\d]*(?:[e][+-][\d]+){0,1}'

NOTIFIERS = 'notifiers'


def read_nexus(tree_path):
    with open(tree_path, 'r') as f:
        nexus = f.read()
    # replace CI_date="2019(2018,2020)" with CI_date="2018 2020"
    nexus = re.sub(r'CI_date="({})\(({}),({})\)"'.format(DATE_REGEX, DATE_REGEX, DATE_REGEX), r'CI_date="\2 \3"',
                   nexus)
    temp = tree_path + '.{}.temp'.format(datetime.timestamp(datetime.now()))
    with open(temp, 'w') as f:
        f.write(nexus)
    trees = list(Phylo.parse(temp, 'nexus'))
    os.remove(temp)
    return trees


def parse_nexus(tree_path):
    trees = []
    for nex_tree in read_nexus(tree_path):
        todo = [(nex_tree.root, None)]
        tree = None
        while todo:
            clade, parent = todo.pop()
            dist = 0
            try:
                dist = float(clade.branch_length)
            except:
                pass
            name = getattr(clade, 'name', None)
            if not name:
                name = getattr(clade, 'confidence', None)
                if not isinstance(name, str):
                    name = None
            node = TreeNode(dist=dist, name=name)
            if parent is None:
                tree = node
            else:
                parent.add_child(node)
            todo.extend((c, node) for c in clade.clades)
        trees.append(tree)
    return trees


def read_tree(tree_path):
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(tree_path, format=f)
            break
        except:
            continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))
    return tree


def resolve_tree(tree, max_extra_brlen=0):
    """
    Resolves polytomies in the tree in a coalescent manner.
    The newly created branch gets a length uniformly drawn
    from ]0; min(max_extra_brlen, min(99% of coallessed child branch lengths)],
    which is then removed from the corresponding child branch lengths.

    :param tree: tree to resolve
    :param max_extra_brlen: maximum branch length for newly created branches
    :return:
    """
    polytomy_counter = Counter()
    for n in tree.traverse('postorder'):
        n_my = len(n.children)
        if n_my > 2:
            polytomy_counter[n_my] += 1
            while len(n.children) > 2:
                child1, child2 = np.random.choice(n.children, 2, replace=False)
                n.remove_child(child1)
                n.remove_child(child2)
                dist = (1 - np.random.random(1)[0]) * min(max_extra_brlen, child1.dist * .99, child2.dist * .99) \
                    if max_extra_brlen > 0 else 0
                parent = n.add_child(dist=dist)
                parent.add_child(child1, dist=child1.dist - dist)
                parent.add_child(child2, dist=child2.dist - dist)
    print('Resolved {} polytomies in a tree of {} tips: {}'
          .format(sum(polytomy_counter.values()), len(tree),
                  ', '.join('{} of {}'.format(v, k)
                            for (k, v) in sorted(polytomy_counter.items(), key=lambda _: -_[0]))))


def resolve_forest(forest, max_extra_brlen=None):
    """
    Resolves polytomies in the forest in a coalescent manner.
    The newly created branch gets a length uniformly drawn
    from ]0; min(max_extra_brlen, min(99% of coallessed child branch lengths)],
    which is then removed from the corresponding child branch lengths.

    :param forest: forest to resolve
    :param max_extra_brlen: maximum branch length for newly created branches.
    If not given (None), will be set to 1% of the length of the shortest non-zero branch in the tree.
    :return:
    """
    if not max_extra_brlen:
        max_extra_brlen = min(min(_.dist for _ in tree.traverse() if _.dist) for tree in forest) * 0.01

    for tree in forest:
        resolve_tree(tree, max_extra_brlen)


def rescale_forest(forest, T_target=1000, T=None):
    annotate_forest_with_time(forest)
    T_initial = get_T(T=T, forest=forest)

    if T_initial == T_target:
        return 1

    scaling_factor = T_target / T_initial

    for tree in forest:
        for n in tree.traverse():
            n.add_feature(TIME, getattr(n, TIME) * scaling_factor)
            n.dist *= scaling_factor

    return scaling_factor


def read_forest(tree_path):
    try:
        roots = parse_nexus(tree_path)
        if roots:
            return roots
    except:
        pass
    if os.path.exists(tree_path):
        with open(tree_path, 'r') as f:
            nwks = f.read().replace('\n', '').split(';')
    else:
        try:
            nwks = tree_path.replace('\n', '').split(';')
        except:
            pass
    if not nwks:
        raise ValueError('Could not find any trees (in newick or nexus format) in the file {}.'.format(tree_path))
    return [read_tree(nwk + ';') for nwk in nwks[:-1]]


def annotate_tree_with_time(tree, start_time=0):
    for n in tree.traverse('preorder'):
        p_time = start_time if n.is_root() else getattr(n.up, TIME)
        n.add_feature(TIME, p_time + n.dist)
    return tree


def annotate_forest_with_time(forest, start_times=None):
    if start_times:
        if len(start_times) < len(forest):
            raise ValueError(f'{len(start_times)} start times are specified but the forest contains {len(forest)} trees. '
                             f'Either specify as many start times as forest trees or set start times to None '
                             f'(to put all the tree start times at 0)')
    else:
        start_times = [0] * len(forest)

    for tree, start_time in zip(forest, start_times):
        if not hasattr(tree, TIME):
            annotate_tree_with_time(tree, start_time)


def get_T(T, forest):
    if T is None:
        T = 0
        for tree in forest:
            T = max(T, max(getattr(_, TIME) for _ in tree))
    return T


def sort_tree(tree, add_root_feature=False, oldest_tip_time_feature=OLDEST_TIP_TIME):
    """
    Reorganise a tree in such a way that for each node its child subtrees are sorted by the time of sampling:
    the subtree containing the oldest tip (with the oldest sampling time) is the first.
    The tree must be time-annotated.

    :param tree: input tree as a ete3 object, with TIME annotations
    :param add_root_feature: if True, the root will get annotated with the time of its oldest sampled tip
    :param oldest_tip_time_feature: feature name to store the root's oldest tip's sampling time
    :return: modified tree (it modifies the input tree object and returns it)
    """
    for n in tree.traverse('postorder'):
        if n.is_leaf():
            n.add_feature(oldest_tip_time_feature, getattr(n, TIME))
            continue
        n.children = sorted(n.children, key=lambda _: getattr(_, oldest_tip_time_feature))
        min_t = np.inf
        for c in n.children:
            min_t = min(min_t, getattr(c, oldest_tip_time_feature))
            delattr(c, oldest_tip_time_feature)
        if not n.is_root() or add_root_feature:
            n.add_feature(oldest_tip_time_feature, min_t)
    return tree


def sort_forest(forest, oldest_tip_time_feature=OLDEST_TIP_TIME):
    """
    Reorganises each tree in the forest in such a way that for each node
    its child subtrees are sorted by the time of sampling:
    the subtree containing the oldest tip (with the oldest sampling time) is the first.
    Then sorts the forest by the time of sampling:
    the tree containing the oldest tip (with the oldest sampling time) is the first.
    The forest must be time-annotated.

    :param forest: list of input trees as ete3 objects, with TIME annotations
    :param oldest_tip_time_feature: feature name to store the root's oldest tip's sampling time
    :return: list of sorted trees
    """

    for tree in forest:
        sort_tree(tree, add_root_feature=True, oldest_tip_time_feature=oldest_tip_time_feature)
    return sorted(forest, key=lambda _: getattr(_, oldest_tip_time_feature))


def get_total_num_notifiers(tree):
    return sum(len(getattr(_, NOTIFIERS)) for _ in tree.traverse())


def get_max_num_notifiers(tree):
    n = len(tree)
    return n ** 2 - 2 * n + 2


def get_min_num_notifiers(tree):
    n = len(tree)
    return n


def preannotate_notifiers(forest):
    """
    Preannotates each tree node with potential notifiers from upper subtree
    :param forest: forest of trees to be annotated
    :return: void, adds NOTIFIERS feature to forest tree nodes.
        This feature contains a (potentially empty) set of upper tree notifiers
    """
    for tree in forest:
        for tip in tree:
            if not tip.is_root():
                parent = tip.up
                for sis in parent.children:
                    if sis != tip:
                        sis.add_feature(NOTIFIERS, getattr(sis, NOTIFIERS, set()) | {tip})
        tree.add_feature(NOTIFIERS, set())
        for node in tree.traverse('preorder'):
            notifiers = getattr(node, NOTIFIERS)
            for child in node.children:
                child.add_feature(NOTIFIERS, getattr(child, NOTIFIERS, set()) | notifiers)


def tree2vector(tree, sort=True):
    if sort:
        sort_tree(tree)

    def node2vec(node):
        result = []
        for child in node.children:
            result.extend(node2vec(child))
        result.append((getattr(node, TIME) - node.dist, getattr(node, TIME)))
        return result

    result = node2vec(tree)
    if tree.dist >= EPSILON:
        start_time = getattr(tree, TIME) - tree.dist
        result.append((start_time, start_time))

    return result


def forest2vector(forest):
    forest = sort_forest(forest)
    result = []
    for tree in forest:
        result.extend(tree2vector(tree, sort=False))
    return result


def vector2forest(vector):
    result = []
    while vector:
        result.append(vector2tree(vector))
    return list(reversed(result))


def vector2tree(vector):

    def vec2tree(vec, tree=None):
        if not vec:
            return tree
        tp, ti = vec.pop()
        node = TreeNode(dist=ti - tp)
        node.add_feature(TIME, ti)

        if not tree or ti > tp or np.abs(getattr(tree, TIME) - tp) > EPSILON:
            # A zero-length branch means it is a root of the next tree
            while vec and np.abs(vec[-1][0] - ti) < EPSILON <= np.abs(vec[-1][0] - vec[-1][1]):
                node = vec2tree(vec, node)

        if not tree:
            tree = node
        elif np.abs(getattr(tree, TIME) - tp) < EPSILON:
            tree.add_child(node)
        elif np.abs(getattr(tree, TIME) - tree.dist - ti) < EPSILON:
            node.add_child(tree)
            tree = node

        return tree

    tree = vec2tree(vector)
    for n in tree.traverse('postorder'):
        n.children = list(reversed(n.children))
    if len(tree.children) == 1 and tree.dist < EPSILON:
        tree = tree.children[0].detach()
    return tree

