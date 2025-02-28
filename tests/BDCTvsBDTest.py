import unittest

from ete3 import Tree

from bdct import bdct_model, bd_model
from bdct.tree_manager import get_T

NWK = '((A:1, B:1.2)AB:2.2, (C:1.5, (D:1.6, ((E:1.9, F:1.99)EF:3.4, (G:1.33, H:1.44)GH:1.22)EH:1.88)DH:1.07)CH:1.65)root:0;'


class BDCTvsBDTest(unittest.TestCase):

    def test_bdct_vs_bd_lk(self):
        forest = [Tree(NWK, format=3)]
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        vs_bdct, _ = bdct_model.infer(forest, T, p=0.3)
        vs_bd, _ = bd_model.infer(forest, T, p=0.3)
        lk_bdct = bdct_model.loglikelihood(forest, *vs_bdct, T=T)
        lk_bd = bd_model.loglikelihood(forest, *vs_bd, T=T)
        self.assertGreater(lk_bdct, lk_bd)

    def test_bdct_vs_bd_lk_tip_forest(self):
        forest = [Tree('A:5;', format=3), Tree('B:4;', format=3)]
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        vs_bdct, _ = bdct_model.infer(forest, T, p=0.3)
        vs_bd, _ = bd_model.infer(forest, T, p=0.3)
        lk_bdct = bdct_model.loglikelihood(forest, *vs_bdct, T=T)
        lk_bd = bd_model.loglikelihood(forest, *vs_bd, T=T)
        self.assertAlmostEqual(lk_bdct, lk_bd, places=3)

    def test_bdct_upsilon_0_vs_bd_lk(self):
        forest = [Tree(NWK, format=3)]
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        vs_bd, _ = bd_model.infer(forest, T, p=0.3)
        lk_bdct = bdct_model.loglikelihood(forest, la=vs_bd[0], psi=vs_bd[1], rho=vs_bd[2], phi=1,
                                           upsilon=0, T=T)
        lk_bd = bd_model.loglikelihood(forest, *vs_bd, T=T)
        self.assertAlmostEqual(lk_bdct, lk_bd, places=3)

    def test_bdct_upsilon_0_vs_bd_lk2(self):
        forest = [Tree(NWK, format=3)]
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        vs_bdct, _ = bdct_model.infer(forest, T, p=0.3)
        lk_bdct = bdct_model.loglikelihood(forest, la=vs_bdct[0], psi=vs_bdct[1], rho=vs_bdct[3], phi=1,
                                           upsilon=0, T=T)
        lk_bd = bd_model.loglikelihood(forest, la=vs_bdct[0], psi=vs_bdct[1], rho=vs_bdct[3], T=T)
        self.assertAlmostEqual(lk_bdct, lk_bd, places=3)

    def test_bdct_upsilon_0_vs_bd_params(self):
        forest = [Tree(NWK, format=3)]
        bdct_model.preprocess_forest(forest)
        T = get_T(T=None, forest=forest)
        vs_bdct, _ = bdct_model.infer(forest, T, p=0.3, upsilon=0)
        vs_bd, _ = bd_model.infer(forest, T, p=0.3)
        self.assertAlmostEqual(vs_bdct[0], vs_bd[0], places=3)
        self.assertAlmostEqual(vs_bdct[1], vs_bd[1], places=3)
        self.assertAlmostEqual(vs_bdct[3], vs_bd[2], places=3)
        
