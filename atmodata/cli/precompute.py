import argparse

import atmodata  # noqa: F401
from atmodata.datasets import WeatherBench
from atmodata.reduction import StatisticsSaver


def cli_precompute():
    parser = argparse.ArgumentParser(prog='atmodata-precompute', description='Pre-compute statistics for a dataset.')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--years', type=int, nargs='+', required=True)
    parser.add_argument('--variables', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, default='stats.nc')
    parser.add_argument('--pca-dim', type=int, default=128)
    args = parser.parse_args()

    print(f'Loading data from {args.path} for years {args.years}...')
    print('Calculating statistics for the following variables:')
    for var in args.variables:
        print(f' - {var}')
    print('Output will be saved to ', args.output)
    print('Calculating statistics...')

    dataset = WeatherBench(args.path, args.variables, args.years, shards_per_year=12).prefetch(3)
    saver = StatisticsSaver(dim='time', pca_components=args.pca_dim)
    saver.process(dataset)
    saver.save(args.output)


if __name__ == '__main__':
    cli_precompute()
