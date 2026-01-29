import os
import unittest

from bdct import bd_model
from bdct.tree_manager import get_T, annotate_forest_with_time, read_forest, TIME

NWK = os.path.join(os.path.dirname(__file__), 'data', 'tree.bd.nwk')

"""
Expected output:

,R0,infectious time,sampling probability,transmission rate,removal rate
value,4.009203145300569,5.2416207217218505,0.2987834524384259,0.76487852863638,0.1907806865643465
CI_min,3.8488882795703616,4.578084522347929,0.2987834524384259,0.7342935484859,0.1655754391300192
CI_max,4.174037339591709,6.039543094400272,0.2987834524384259,0.7963257093925245,0.21843196540354332
"""


class BDTest(unittest.TestCase):

    def test_estimate_bd_la(self):
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, _], _ = bd_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.76487852863638, la, places=5)

    def test_estimate_bd_psi(self):
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, _], _ = bd_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.1907806865643465, psi, places=5)

    def test_lk_bd(self):
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        vs, _ = bd_model.infer(forest, T, p=0.2987834524384259)
        lk_bd = bd_model.loglikelihood(forest, *vs, T=T)
        self.assertAlmostEqual(-1972.0450188910957, lk_bd, places=5)

    def test_time_annotations(self):
        forest = read_forest(NWK)
        # duplicate the same tree to have two trees in the forest
        forest = [forest[0], read_forest(NWK)[0]]
        # By default, the start time will be the same
        annotate_forest_with_time(forest)
        T1 = get_T(T=None, forest=forest)
        vs, _ = bd_model.infer(forest, T1, p=0.2987834524384259)
        lk_bd = bd_model.loglikelihood(forest, *vs, T=T1)

        annotate_forest_with_time(forest, start_times=[0, T1/2])
        T2 = get_T(T=None, forest=forest)
        self.assertAlmostEqual(T2, 1.5 * T1, places=5)
        vs, _ = bd_model.infer(forest, T2, p=0.2987834524384259)
        lk_bd2 = bd_model.loglikelihood(forest, *vs, T=T2)

        self.assertNotAlmostEqual(lk_bd, lk_bd2, places=3)


    def test_forest_flat(self):
        tree = read_forest(NWK)[0]
        annotate_forest_with_time([tree])
        T = get_T(T=None, forest=[tree])

        forest = []
        t_start = 0.1 * T
        todo = [tree]
        while todo:
            node = todo.pop()
            time = getattr(node, TIME)
            if time >= t_start:
                node.up = None
                node.dist = min(node.dist, time - t_start)
                forest.append(node)
            else:
                todo.extend(node.children)

        print(len(forest))

        [la, psi, _], _ = bd_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.76487852863638, la, places=2)
        self.assertAlmostEqual(0.1907806865643465, psi, places=2)

        annotate_forest_with_time([tree])
        T2 = get_T(T=None, forest=[tree])
        self.assertAlmostEqual(T2, 0.9 * T, places=5)
        [la2, psi2, _], _ = bd_model.infer(forest, T2, p=0.2987834524384259)
        self.assertAlmostEqual(la2, la, places=5)
        self.assertAlmostEqual(psi2, psi, places=5)



    def test_forest_clusters(self):
        tree = read_forest(NWK)[0]
        annotate_forest_with_time([tree])
        T = get_T(T=None, forest=[tree])

        forest = []
        # let's take clusters that are 4 branches away from the root
        todo = [(tree, 0)]
        while todo:
            node, level = todo.pop()
            if level >= 4:
                node.up = None
                node.dist = 0
                forest.append(node)
            else:
                todo.extend([(c, level + 1) for c in node.children])

        print([getattr(_, TIME) for _ in forest])

        [la, psi, _], _ = bd_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.76487852863638, la, places=2)
        self.assertAlmostEqual(0.1907806865643465, psi, places=2)

