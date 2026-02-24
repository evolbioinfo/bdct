import numpy as np
from scipy.stats import binomtest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compares likelihoods.")
    parser.add_argument('--likelihoods_est', type=str, nargs='+', help="likelihood values")
    parser.add_argument('--likelihoods_real', type=str, nargs='+', help="likelihood values")
    parser.add_argument('--log', type=str, help="likelihood stats")
    params = parser.parse_args()

    n_higher = 0
    n_lower = 0
    n = 0
    for lk_est_file, lk_real_file in zip(params.likelihoods_est, params.likelihoods_real):
        n += 1
        lk_est = float(open(lk_est_file).read())
        lk_real = float(open(lk_real_file).read())
        if np.round(lk_est - lk_real, 0) > 0:
            n_higher += 1
        if np.round(lk_real - lk_est, 0) > 0:
            n_lower += 1
            print(lk_est, lk_real, lk_est_file, lk_real_file)
    log = 'Estimated likelihood\t is higher than real\t in {:.1f}% of cases' \
        .format(100 * n_higher / n)
    n_equal = n - n_higher - n_lower
    log += '\n\t is equal to real\t in {:.1f}% of cases' \
        .format(100 * n_equal / n)
    log += '\n\t is lower than real\t in {:.1f}% of cases' \
        .format(100 * n_lower / n)

    p = binomtest(min(n_higher, n_lower), n_higher + n_lower, 0.5, alternative='two-sided').pvalue
    log += '\n\t is different from real\t with a p-value of {:g}'.format(p)
    print(log)
    open(params.log, 'w+').write(log + '\n')
