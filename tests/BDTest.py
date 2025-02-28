import os
import unittest

from bdct import bd_model
from bdct.tree_manager import get_T, annotate_forest_with_time, read_forest

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
