import unittest

from ete3 import Tree

from bdpn.tree_manager import tree2vector, vector2tree, \
    sort_tree, annotate_tree_with_time, TIME, read_tree, read_forest, annotate_forest_with_time, forest2vector, vector2forest, \
    sort_forest

VEC = [(2.0, 3.0), (2.5, 4.5), (2.5, 5.5), (2.0, 2.5), (0.0, 2.0),
       (1.2, 6.2), (1.2, 7.2), (0.2, 1.2), (8.2, 11.2), (8.2, 15.2), (0.2, 8.2), (0.0, 0.2), (0.0, 0.0)]

NWK_SORTED = '((a:1, (b:2, c:3)bc:0.5)abc:2, ((d:5, e:6)de:1, (f:3, h:7)fh:8)defh:0.2);'
NWK = '(((b:2, c:3)bc:0.5, a:1)abc:2, ((d:5, e:6)de:1, (h:7, f:3)fh:8)defh:0.2);'
NWK_WITH_ROOT_DIST = '(((b:2, c:3)bc:0.5, a:1)abc:2, ((d:5, e:6)de:1, (h:7, f:3)fh:8)defh:0.2)root:5;'
NWK_MINI_WITH_ROOT_DIST = '((b:0.4, c:0.3)bc:0.25, a:0.1)abc:0.2;'
NWK_MINI = '((b:0.45, c:0.35)bc:0.25, a:0.15);'
NWK_LEAF = 'a:7;'
NWK_UNRESOLVED = '(((b:2, c:3, c1:8, b1:9)bc:0.5, a:1, a1:1.5)abc:2, ((d:5, e:6)de:1, (d2:5, e2:6, e3:7)de2:1.2,  (h:7, f:3)fh:8)defh:0.2);'


class Tree2VecTest(unittest.TestCase):

    def test_sort_tree(self):
        tree = Tree(NWK, format=3)
        annotate_tree_with_time(tree)
        sort_tree(tree)
        print(tree.get_ascii(attributes=[TIME]))
        tree_sorted = Tree(NWK_SORTED, format=3)
        annotate_tree_with_time(tree_sorted)
        for n, n_sorted in zip(tree.traverse('postorder'), tree_sorted.traverse('postorder')):
            self.assertAlmostEqual(n.dist, n_sorted.dist, 6)
            self.assertAlmostEqual(getattr(n, TIME), getattr(n_sorted, TIME), 6)

    def test_tree2vec(self):
        tree = Tree(NWK, format=3)
        annotate_tree_with_time(tree)
        vector = tree2vector(tree)
        print(vector)
        for (tp, ti), (tp2, ti2) in zip(vector, VEC):
            self.assertAlmostEqual(tp, tp2, 6)
            self.assertAlmostEqual(ti, ti2, 6)

    def test_vec2tree(self):
        real_tree = sort_tree(annotate_tree_with_time(Tree(NWK, format=3)))
        print(real_tree.get_ascii(attributes=[TIME]))
        tree = vector2tree(VEC)
        print(tree.get_ascii(attributes=[TIME]))
        for n, n2 in zip(tree.traverse(), real_tree.traverse()):
            self.assertAlmostEqual(n.dist, n2.dist)
            self.assertAlmostEqual(getattr(n, TIME), getattr(n2, TIME), 6)

    def test_forest2vec_and_back(self):
        forest = '\n'.join([NWK_WITH_ROOT_DIST, NWK, NWK_MINI, NWK_MINI_WITH_ROOT_DIST, NWK_LEAF, NWK_UNRESOLVED])
        in_forest = read_forest(forest)
        annotate_forest_with_time(in_forest)
        vector = forest2vector(in_forest)

        for tree in sort_forest(in_forest):
            print(tree.write(format_root_node=True))
        print(vector)
        out_forest = vector2forest(vector)
        for tree in out_forest:
            print(tree.write(format_root_node=True))
        for in_tree, out_tree in zip(sort_forest(in_forest), out_forest):
            for n, n2 in zip(in_tree.traverse(), out_tree.traverse()):
                self.assertAlmostEqual(n.dist, n2.dist)
                self.assertAlmostEqual(getattr(n, TIME), getattr(n2, TIME), 6)

    def test_forest2vec_and_back_with_start_times(self):
        forest = '\n'.join([NWK_WITH_ROOT_DIST, NWK, NWK_MINI, NWK_MINI_WITH_ROOT_DIST, NWK_LEAF, NWK_UNRESOLVED])
        in_forest = read_forest(forest)
        annotate_forest_with_time(in_forest, start_times=[10, 0, 100, 50, 200, 10])
        vector = forest2vector(in_forest)

        for tree in sort_forest(in_forest):
            print(tree.write(format_root_node=True))
        print(vector)
        out_forest = vector2forest(vector)
        for tree in out_forest:
            print(tree.write(format_root_node=True))
        for in_tree, out_tree in zip(sort_forest(in_forest), out_forest):
            for n, n2 in zip(in_tree.traverse(), out_tree.traverse()):
                self.assertAlmostEqual(n.dist, n2.dist)
                self.assertAlmostEqual(getattr(n, TIME), getattr(n2, TIME), 6)
