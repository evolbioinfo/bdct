import unittest

from ete3 import Tree

from bdpn.tree_manager import tree2vector, vector2tree, \
    sort_tree, annotate_tree

VEC = [(8.2, 15.2), (8.2, 11.2), (0.2, 8.2), (1.2, 7.2), (1.2, 6.2), (0.2, 1.2), (0.0, 0.2), (2.5, 5.5), (2.5, 4.5),
       (2.0, 2.5), (2.0, 3.0), (0.0, 2.0), (0.0, 0.0)]

NWK = '((a:1, (b:2, c:3)bc:0.5)abc:2, ((d:5, e:6)de:1, (f:3, h:7)fh:8)defh:0.2);'


class Tree2VecTest(unittest.TestCase):

    def test_tree2vec(self):
        tree = Tree(NWK, format=3)
        annotate_tree(tree)
        print(tree.get_ascii(attributes=['time']))
        vector = tree2vector(tree)
        print(vector)
        for (tp, ti), (tp2, ti2) in zip(vector, VEC):
            self.assertAlmostEqual(tp, tp2, 6)
            self.assertAlmostEqual(ti, ti2, 6)

    def test_vec2tree(self):
        tree = vector2tree(VEC, None)
        print(tree.get_ascii(attributes=['time']))
        real_tree = sort_tree(annotate_tree(Tree(NWK, format=3)))
        print(real_tree.get_ascii(attributes=['time']))
        for n, n2 in zip(tree.traverse(), real_tree.traverse()):
            self.assertAlmostEqual(n.dist, n2.dist)