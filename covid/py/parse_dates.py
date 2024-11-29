
if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--ids', required=True, type=str)
    parser.add_argument('--dates', required=True, type=str)
    params = parser.parse_args()

    with open(params.ids, 'r') as f:
        ids = f.read().strip().strip('\n').split('\n')

    with open(params.dates, 'w+') as f:
        f.write(f'{len(ids)}\n')
        for _ in ids:
            f.write(f'{_}\t{_[_.rfind("_") + 1:]}\n')