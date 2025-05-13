import os
import unittest

from bdct import bd_model, bdsky_model
from bdct.tree_manager import get_T, annotate_forest_with_time, read_forest



NWK = os.path.join(os.path.dirname(__file__), 'data', 'tree.bd.nwk')

"""
Expected BD output:

,R0,infectious time,sampling probability,transmission rate,removal rate
value,4.009203145300569,5.2416207217218505,0.2987834524384259,0.76487852863638,0.1907806865643465
CI_min,3.8488882795703616,4.578084522347929,0.2987834524384259,0.7342935484859,0.1655754391300192
CI_max,4.174037339591709,6.039543094400272,0.2987834524384259,0.7963257093925245,0.21843196540354332
"""

R0, d, p = 3.965553172971901, 5.109012509433994, 0.2987834524384259
la, psi = R0 / d, 1 / d


class BDSKYTest(unittest.TestCase):

    def test_bd_lk_vs_bdsky_lk(self):
        """
        Compare likelihood calculated by the BD model to the one calculated by BDSKY with one time interval on a BD tree
        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        lk_bd = bd_model.loglikelihood(forest, la=la, psi=psi, rho=p, T=T)
        lk_bdsky = bdsky_model.loglikelihood(forest, la, psi, p, T=T)
        self.assertAlmostEqual(lk_bd, lk_bdsky, 6)

    def test_bdsky1_lk_vs_bdsky2_lk(self): #NOT PASSING
        """
        Compare likelihood calculated by the BDSKY model with one time interval
        vs BDSKY model with two time intervals but the same model parameters on both intervals,
        on a BD tree
        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        lk_bdsky1 = bdsky_model.loglikelihood(forest, la, psi, p, T=T, u=0)
        lk_bdsky2 = bdsky_model.loglikelihood(forest, la, la, psi, psi, p, p, T / 2, T=T, n_intervals=2)
        self.assertAlmostEqual(lk_bdsky1, lk_bdsky2, 6)


    def test_bdsky2_lk_higher_than_bd(self):
        """
        Make sure that the BDSKY model with two time intervals finds more likely parameters than the BD model
        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        vs, _ = bdsky_model.infer_skyline(forest, T, n_intervals=2, p=[p, p], times=[T / 2])
        lk_bdsky = bdsky_model.loglikelihood(forest, *vs, T=T, n_intervals=2)
        vs, _ = bd_model.infer(forest, T, p=p)
        lk_bd = bd_model.loglikelihood(forest, *vs, T=T)
        self.assertGreaterEqual(lk_bdsky, lk_bd)

    def test_bdsky3_lk_higher_than_bdsky2(self):
        """
        Make sure that the BDSKY model with three time intervals finds more likely parameters
        than the BDSKY model with two time intervals
        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        vs, _ = bdsky_model.infer_skyline(forest, T, n_intervals=2, p=[p, p], times=[T / 2])
        lk_bdsky2 = bdsky_model.loglikelihood(forest, *vs, T=T, n_intervals=2)
        vs, _ = bdsky_model.infer_skyline(forest, T, n_intervals=3, p=[p, p, p], times=[T / 3, 2 * T / 3])
        lk_bdsky3 = bdsky_model.loglikelihood(forest, *vs, T=T, n_intervals=3)
        self.assertGreaterEqual(lk_bdsky3, lk_bdsky2)


    def test_estimate_bdsky_la_first_interval(self):
        """
        Test that the BDSKY model with the first interval occupying almost all the time on a BD tree
        estimates similar parameters to BD

        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        [la_bd, psi_bd, _], _ = bd_model.infer(forest, T, p=p)
        [la, _, psi, _, _, _, _], _ = bdsky_model.infer_skyline(forest, T, n_intervals=2, p=[p, p], times=[T * 0.999])
        self.assertAlmostEqual(la_bd, la, places=3)

    def test_estimate_bdsky_psi_first_interval(self):
        """
        Test that the BDSKY model with the first interval occupying almost all the time on a BD tree
        estimates similar parameters to BD

        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        [la_bd, psi_bd, _], _ = bd_model.infer(forest, T, p=p)
        [la, _, psi, _, _, _, _], _ = bdsky_model.infer_skyline(forest, T, n_intervals=2, p=[p, p], times=[T * 0.999])
        self.assertAlmostEqual(psi_bd, psi, places=3)

    def test_estimate_bdsky_la_last_interval(self):
        """
        Test that the BDSKY model with the last interval occupying almost all the time on a BD tree
        estimates similar parameters to BD

        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        [la_bd, psi_bd, _], _ = bd_model.infer(forest, T, p=p)
        [_, la, _, psi, _, _, _], _ = bdsky_model.infer_skyline(forest, T, n_intervals=2, p=[p, p], times=[T * 0.001])
        self.assertAlmostEqual(la_bd, la, places=3)

    def test_estimate_bdsky_psi_last_interval(self):
        """
        Test that the BDSKY model with the last interval occupying almost all the time on a BD tree
        estimates similar parameters to BD

        """
        forest = read_forest(NWK)
        annotate_forest_with_time(forest)
        T = get_T(T=None, forest=forest)
        [la_bd, psi_bd, _], _ = bd_model.infer(forest, T, p=p)
        [_, la, _, psi, _, _, _], _ = bdsky_model.infer_skyline(forest, T, n_intervals=2, p=[p, p], times=[T * 0.001])
        self.assertAlmostEqual(psi_bd, psi, places=3)
