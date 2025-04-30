import os
import unittest
import numpy as np

from bdct.tree_manager import read_forest, annotate_forest_with_time, get_T
from bdct.parameter_estimator import optimize_likelihood_params
from bdct.bdsky_model import loglikelihood as bdsky_loglikelihood
from bdct.bd_model import get_start_parameters

# Known true parameters from simulation
R0, d, p = 3.965553172971901, 5.109012509433994, 0.2987834524384259
la_true, psi_true = R0 / d, 1 / d

NWK = os.path.join(os.path.dirname(__file__), 'data', 'tree.bd.nwk')

class OptimizationBehaviorTest(unittest.TestCase):

    def setUp(self):
        self.forest = read_forest(NWK)
        annotate_forest_with_time(self.forest)
        self.T = get_T(None, forest=self.forest)
        self.bounds = [(1e-3, 10), (0.1, 5), (1e-6, 1)]

    def test_optimizer_recovers_parameters(self):
        """
        Check if optimize_likelihood_params can recover known parameters from a BD tree.
        """
        # Build input
        input_parameters = [None, None, p]  # optimizing lambda and psi, fixing rho
        start_params = [0.3, 0.3, p]  # intentionally poor starting point
        optimise_as_logs = [False, False, False]

        # Run optimization
        estimated_params, final_ll = optimize_likelihood_params(
            forest=self.forest,
            T=self.T,
            input_parameters=input_parameters,
            loglikelihood_function=bdsky_loglikelihood,
            bounds=self.bounds,
            start_parameters=start_params,
            threads=1,
            num_attemps=10,
            optimise_as_logs=optimise_as_logs
        )

        la_est, psi_est, rho_est = estimated_params

        # Check that they are close to the true values
        self.assertAlmostEqual(la_est, la_true, delta=0.02)
        self.assertAlmostEqual(psi_est, psi_true, places=2)
        self.assertAlmostEqual(rho_est, p, places=5)

    def test_optimizer_succeeds_from_multiple_starts(self):
        """
        Try multiple different initializations and make sure optimizer consistently converges to same result.
        """
        input_parameters = [None, None, p]
        optimise_as_logs = [False, False, False]

        errors = []

        for seed in range(5):
            np.random.seed(seed)
            start_params = [
                np.random.uniform(0.1, 1.5),  # lambda
                np.random.uniform(0.1, 1.5),  # psi
                p
            ]

            estimated_params, _ = optimize_likelihood_params(
                forest=self.forest,
                T=self.T,
                input_parameters=input_parameters,
                loglikelihood_function=bdsky_loglikelihood,
                bounds=self.bounds,
                start_parameters=start_params,
                threads=1,
                num_attemps=5,
                optimise_as_logs=optimise_as_logs
            )

            la_est, psi_est, rho_est = estimated_params

            if not (np.isclose(la_est, la_true, rtol=0.2) and np.isclose(psi_est, psi_true, rtol=0.2)):
                errors.append((start_params, estimated_params))

        self.assertEqual(len(errors), 0, f"Failed to converge for some seeds:\n{errors}")


if __name__ == '__main__':
    unittest.main()
