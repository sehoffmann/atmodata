'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import argparse
import math

import atmodata  # noqa: F401
from atmodata.builder import AtmodataPipeBuilder
from atmodata.datasets import ERA5, WeatherBench
from atmodata.tasks import ForecastingTask
from atmodata.utils import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='WeatherBench', choices=['WeatherBench', 'ERA5'])
    parser.add_argument('--variables', type=str, nargs='+', default=['z500', 't500'])
    parser.add_argument('--years', type=int, nargs='+', default=[1990])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--parallel-shards', type=int, default=3)
    parser.add_argument('--reuse-factor', type=float, default=1.0)
    parser.add_argument('--synthetic', action='store_true')

    args = parser.parse_args()
    print(f'Benchmarking {args.dataset}')
    print(f'* Vars: {args.variables}')
    print(f'* Years: {args.years}')
    print(f'* Workers: {args.workers}')
    print(f'* Batch size: {args.batch_size}')
    print(f'* Parallel shards: {args.parallel_shards}')
    print(f'* Reuse factor: {args.reuse_factor}')
    print(f'* Synthetic: {args.synthetic}')
    print(f'* Path: {None if args.synthetic else args.path}')

    if args.dataset == 'WeatherBench':
        dataset_cls = WeatherBench
        shards_per_year = 12
        crops_per_sample = int(math.ceil((128 * 256) / (96 * 96) * args.reuse_factor))
    elif args.dataset == 'ERA5':
        dataset_cls = ERA5
        shards_per_year = 12 * 2
        crops_per_sample = int(math.ceil((720 * 1440) / (96 * 96) * args.reuse_factor))

    dataset = dataset_cls(
        args.path, args.variables, args.years, shards_per_year=shards_per_year, shuffle=True, synthetic=args.synthetic
    )
    task = atmodata.utils.SequentialTransform(
        ForecastingTask(10, 6, crop_size={'lat': 96, 'lon': 96}, crops_per_sample=crops_per_sample),
        atmodata.iter.XrVariableSelecter.as_transform(
            {
                'predict': ['z500', 't500'],
            },
            keep_dataset=True,
        ),
    )

    builder = AtmodataPipeBuilder(
        dataset,
        task,
        batch_size=args.batch_size,
        num_parallel_shards=args.parallel_shards,
        dataloading_prefetch_cnt=3,
        device_prefetch_cnt=2,
    )
    dataloader = builder.multiprocess(args.workers).transfer_to_device('cuda').build_dataloader()

    print('-' * 50)
    benchmark(dataloader, 0)


if __name__ == '__main__':
    main()
