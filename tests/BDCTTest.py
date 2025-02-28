import os
import unittest

from bdct import bdct_model
from bdct.tree_manager import get_T, read_forest

NWK_BD = os.path.join(os.path.dirname(__file__), 'data', 'tree.bd.nwk')
NWK = os.path.join(os.path.dirname(__file__), 'data', 'tree.bdct.nwk')

"""
Expected output:

,R0,infectious time,sampling probability,notification probability,removal time after notification,transmission rate,removal rate,partner removal rate
value,4.466445003743529,5.560549926866046,0.2987834524384259,0.4315092057945722,0.017576966436891443,0.8032380002854934,0.17983832771079983,56.8926386467433
CI_min,4.279122707343025,4.775619549946329,0.2987834524384259,0.3918980091688985,0.015508534958986341,0.7695502717578799,0.15259342622360764,50.0138339495266
CI_max,4.657197991936128,6.553362256475047,0.2987834524384259,0.4720363815533576,0.019994467950791153,0.8375426986878883,0.20939691479637204,64.48062325968161
"""


class BDCTTest(unittest.TestCase):

    def test_estimate_bdct_la(self):
        forest = read_forest(NWK)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, phi, _, upsilon], _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.8032380002854934, la, places=3)

    def test_estimate_bdct_psi(self):
        forest = read_forest(NWK)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, phi, _, upsilon], _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.17983832771079983, psi, places=3)

    def test_estimate_bdct_phi(self):
        forest = read_forest(NWK)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, phi, _, upsilon], _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(56.8926386467433, phi, places=3)

    def test_estimate_bdct_upsilon(self):
        forest = read_forest(NWK)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, phi, _, upsilon], _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.4315092057945722, upsilon, places=3)

    def test_lk_bdct(self):
        forest = read_forest(NWK)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        vs, _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        lk_bdct = bdct_model.loglikelihood(forest, *vs, T=T)
        self.assertAlmostEqual(-1282.398070332461, lk_bdct, places=3)

    def test_estimate_bd_la(self):
        forest = read_forest(NWK_BD)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, phi, _, upsilon], _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.76487852863638, la, places=2)

    def test_estimate_bd_psi(self):
        forest = read_forest(NWK_BD)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, phi, _, upsilon], _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0.1907806865643465, psi, places=2)

    def test_estimate_bd_upsilon(self):
        forest = read_forest(NWK_BD)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        [la, psi, phi, _, upsilon], _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        self.assertAlmostEqual(0, upsilon, places=1)

    def test_lk_bd(self):
        forest = read_forest(NWK_BD)
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        vs, _ = bdct_model.infer(forest, T, p=0.2987834524384259)
        lk_bdct = bdct_model.loglikelihood(forest, *vs, T=T)
        self.assertAlmostEqual(-1970.6768466067208, lk_bdct, places=3)
