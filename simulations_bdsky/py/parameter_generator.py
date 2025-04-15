import numpy as np
import argparse
import sys


def random_float(min_value=0, max_value=1):
    """
    Generate a random float in ]min_value, max_value]
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + (1 - np.random.random(size=1)[0]) * (max_value - min_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates parameters for MTBD-CT simulations with N model skylines.")
    parser.add_argument('--log', default='parameters.log', type=str, help="parameter settings output file")
    parser.add_argument('--num_models', default=2, type=int, help="number of models (1-10)")

    # Define parameter arguments for potentially 10 models
    for i in range(1, 11):
        parser.add_argument(f'--min_R0_{i}', default=1, type=float, help=f"min R0 for model {i} (excluded)")
        parser.add_argument(f'--max_R0_{i}', default=10, type=float, help=f"max R0 for model {i} (included)")
        parser.add_argument(f'--min_rho_{i}', default=0.05, type=float, help=f"min rho for model {i} (excluded)")
        parser.add_argument(f'--max_rho_{i}', default=0.75, type=float, help=f"max rho for model {i} (included)")
        parser.add_argument(f'--min_di_{i}', default=1, type=float,
                            help=f"min infectious time for model {i} (excluded)")
        parser.add_argument(f'--max_di_{i}', default=21, type=float,
                            help=f"max infectious time for model {i} (included)")

    # Change times (n-1 change times for n models)
    parser.add_argument('--min_change_time', default=5, type=float, help="min time for model changes (excluded)")
    parser.add_argument('--max_change_time', default=50, type=float, help="max time for model changes (included)")
    parser.add_argument('--sort_change_times', default=True, type=bool, help="sort change times in ascending order")

    params = parser.parse_args()

    # Validate number of models
    num_models = params.num_models
    if num_models < 1 or num_models > 10:
        print(f"Error: num_models must be between 1 and 10, got {num_models}")
        sys.exit(1)

    # Generate parameters for each model
    model_params = []
    for i in range(1, num_models + 1):
        min_R0 = getattr(params, f'min_R0_{i}')
        max_R0 = getattr(params, f'max_R0_{i}')
        min_rho = getattr(params, f'min_rho_{i}')
        max_rho = getattr(params, f'max_rho_{i}')
        min_di = getattr(params, f'min_di_{i}')
        max_di = getattr(params, f'max_di_{i}')

        R0 = random_float(min_R0, max_R0)
        d = random_float(min_di, max_di)
        psi = 1 / d
        la = psi * R0
        rho = random_float(min_rho, max_rho)

        model_params.append({
            'la': la,
            'psi': psi,
            'rho': rho,
            'R0': R0,
            'd': d
        })

    # Generate change times (n-1 times for n models)
    change_times = []
    if num_models > 1:
        for _ in range(num_models - 1):
            change_time = random_float(params.min_change_time, params.max_change_time)
            change_times.append(change_time)

        # Sort change times if requested
        if params.sort_change_times:
            change_times.sort()

    # Write parameters to the log file
    with open(params.log, 'w+') as fo:
        # Create header
        header_parts = []
        for i in range(1, num_models + 1):
            header_parts.extend([f'la_{i}', f'psi_{i}', f'rho_{i}', f'R0_{i}', f'd_{i}'])

        if change_times:
            for i in range(1, num_models):
                header_parts.append(f'change_time_{i}')

        header = ','.join(header_parts)
        fo.write(f'{header}\n')

        # Create data row
        data_parts = []
        for model in model_params:
            data_parts.extend([
                str(model['la']),
                str(model['psi']),
                str(model['rho']),
                str(model['R0']),
                str(model['d'])
            ])

        for time in change_times:
            data_parts.append(str(time))

        data_row = ','.join(data_parts)
        fo.write(f'{data_row}\n')

    # Print summary to stdout
    print(f"Generated parameters for {num_models} models:")
    for i, model in enumerate(model_params, 1):
        print(f"Model {i}: R0={model['R0']:.4f}, d={model['d']:.4f}, "
              f"psi={model['psi']:.4f}, la={model['la']:.4f}, rho={model['rho']:.4f}")

    if change_times:
        print("\nChange times:")
        for i, time in enumerate(change_times, 1):
            print(f"  Change {i}: t={time:.4f}")


