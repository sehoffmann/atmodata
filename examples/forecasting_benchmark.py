'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import argparse

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
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--parallel-shards', type=int, default=3)
    parser.add_argument('--crops-per-timestep', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=10)
    parser.add_argument('--dataloading-prefetch-cnt', type=int, default=3)
    parser.add_argument('--device-prefetch-cnt', type=int, default=2)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'WeatherBench':
        dataset_cls = WeatherBench
        shards_per_year = 12
        H, W = 128, 256
    elif args.dataset == 'ERA5':
        dataset_cls = ERA5
        shards_per_year = 24
        H, W = 720, 1440

    reuse_factor = args.crops_per_timestep * args.timesteps * args.patch_size * args.patch_size / (H * W)

    print(f'Benchmarking {args.dataset}')
    print('-' * 50)
    print(f'* Vars: {args.variables}')
    print(f'* Crops per timestep: {args.crops_per_timestep}')
    print(f'* Timesteps: {args.timesteps}')
    print(f'* Patch size: {args.patch_size}')
    print('-' * 50)
    print(f'* Workers: {args.workers}')
    print(f'* Batch size: {args.batch_size}')
    print(f'* Parallel shards: {args.parallel_shards}')
    print(f'* Shuffle: {args.shuffle}')
    print(f'* Dataloading prefetch count: {args.dataloading_prefetch_cnt}')
    print(f'* Device prefetch count: {args.device_prefetch_cnt}')
    print(f'* To GPU: {args.cuda}')
    print('-' * 50)
    print(f'* Path: {None if args.synthetic else args.path}')
    print(f'* Years: {args.years}')
    print(f'* Synthetic: {args.synthetic}')
    print('-' * 50)
    print(f'* Data reuse factor: {reuse_factor:.3f} (calculated)')
    print('-' * 50)

    dataset = dataset_cls(
        args.path,
        args.variables,
        args.years,
        shards_per_year=shards_per_year,
        shuffle=args.shuffle,
        synthetic=args.synthetic,
    )
    task = atmodata.utils.SequentialTransform(
        ForecastingTask(
            args.timesteps,
            6,
            crop_size={'lat': args.patch_size, 'lon': args.patch_size},
            crops_per_sample=args.crops_per_timestep,
            shuffle=args.shuffle,
        ),
        atmodata.iter.XrVariableSelecter.as_transform(
            {
                'predict': ['z500', 't500'],
            },
        ),
    )

    builder = AtmodataPipeBuilder(
        dataset,
        task,
        batch_size=args.batch_size,
        num_parallel_shards=args.parallel_shards,
        dataloading_prefetch_cnt=args.dataloading_prefetch_cnt,
        device_prefetch_cnt=args.device_prefetch_cnt,
    )
    if args.cuda:
        builder.transfer_to_device('cuda')
    dataloader = builder.multiprocess(args.workers).build_dataloader()

    print('Benchmarking...')
    benchmark(dataloader, 0)


if __name__ == '__main__':
    main()
