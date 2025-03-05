import os
import re


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compares likelihoods.")
    parser.add_argument('--likelihoods_bd', type=str, nargs='+', help="likelihood values")
    parser.add_argument('--likelihoods_bdct', type=str, nargs='+', help="likelihood values")
    parser.add_argument('--log', type=str, help="likelihood stats")
    params = parser.parse_args()

    with open(params.log, 'w+') as f:
        f.write('\tloglikelihood BDPN\tloglikelihood BD\tloglikelihood ratio\n')
        for lk_bd_file, lk_bdct_file in zip(params.likelihoods_bd, params.likelihoods_bdct):
            basename = os.path.basename(lk_bd_file)
            rep = int(re.findall(r'\.(\d+)\.[a-z]+[=]', basename)[0]) + 1
            lk_bd = float(open(lk_bd_file).read())
            lk_bdct = float(open(lk_bdct_file).read())
            f.write('{}\t{}\t{}\t{}\n'.format(rep, lk_bdct, lk_bd, 2 * (lk_bdct - lk_bd)))
