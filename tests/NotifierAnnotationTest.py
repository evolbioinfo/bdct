import unittest

from ete3 import Tree

from bdct.tree_manager import preannotate_notifiers, NOTIFIERS, get_total_num_notifiers, get_min_num_notifiers, get_max_num_notifiers

NWK = '((A:1, B:1)AB:1, (C:1, (D:1, ((E:1, F:1)EF:1, (G:1, H:1)GH:1)EH:1)DH:1)CH:1)root:0;'


class NotifierAnnotationTest(unittest.TestCase):

    def test_notifiers(self):
        tree = Tree(NWK, format=3)
        preannotate_notifiers([tree])
        name2node = {_.name: _ for _ in tree.traverse()}
        self.assertSetEqual(getattr(name2node['root'], NOTIFIERS), set())
        self.assertSetEqual(getattr(name2node['AB'], NOTIFIERS), set())
        self.assertSetEqual(getattr(name2node['A'], NOTIFIERS), {name2node['B']})
        self.assertSetEqual(getattr(name2node['B'], NOTIFIERS), {name2node['A']})
        self.assertSetEqual(getattr(name2node['CH'], NOTIFIERS), set())
        self.assertSetEqual(getattr(name2node['C'], NOTIFIERS), set())
        self.assertSetEqual(getattr(name2node['DH'], NOTIFIERS), {name2node['C']})
        self.assertSetEqual(getattr(name2node['D'], NOTIFIERS), {name2node['C']})
        self.assertSetEqual(getattr(name2node['EH'], NOTIFIERS), {name2node['C'], name2node['D']})
        self.assertSetEqual(getattr(name2node['EF'], NOTIFIERS), {name2node['C'], name2node['D']})
        self.assertSetEqual(getattr(name2node['E'], NOTIFIERS), {name2node['F'], name2node['C'], name2node['D']})
        self.assertSetEqual(getattr(name2node['F'], NOTIFIERS), {name2node['E'], name2node['C'], name2node['D']})
        self.assertSetEqual(getattr(name2node['GH'], NOTIFIERS), {name2node['C'], name2node['D']})
        self.assertSetEqual(getattr(name2node['G'], NOTIFIERS), {name2node['H'], name2node['C'], name2node['D']})
        self.assertSetEqual(getattr(name2node['H'], NOTIFIERS), {name2node['G'], name2node['C'], name2node['D']})

    def test_num_notifiers_balanced_tree(self):
        tree = Tree('(((A:1, B:1)AB:1, (C:1, D:1)CD:1), ((E:1, F:1)EF:1, (G:1, H:1)GH:1))root:0;', format=3)
        preannotate_notifiers([tree])
        self.assertEqual(get_total_num_notifiers(tree), get_min_num_notifiers(tree))

    def test_num_notifiers_ladder_tree(self):
        tree = Tree('(((((((A:1, B:1)AB:1, C:1)AC:1, D:1)AD:1, E:1)AE:1, F:1)AF:1, G:1)AG:1, H:1)AH:1;', format=3)
        preannotate_notifiers([tree])
        self.assertEqual(get_total_num_notifiers(tree), get_max_num_notifiers(tree))
