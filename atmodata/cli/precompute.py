import argparse

import atmodata  # noqa: F401
from atmodata.datasets import WeatherBench
from atmodata.reduction import StatisticsSaver


def cli_precompute():
    parser = argparse.ArgumentParser(prog='atmodata-precompute', description='Pre-compute statistics for a dataset.')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--years', type=int, nargs='+', default=[1990])
    args = parser.parse_args()

    variables = ['t1000', 'r600']

    dataset = WeatherBench(args.path, variables, args.years, shards_per_year=12).prefetch(3)
    saver = StatisticsSaver(dim='time', daily=True)
    saver.process(dataset)
    print(saver.statistics)
    saver.save('stats.nc')


if __name__ == '__main__':
    cli_precompute()
