import argparse

import atmodata  # noqa: F401
from atmodata.builder import AtmodataPipeBuilder
from atmodata.datasets import WeatherBench
from atmodata.tasks import ForecastingTask
from atmodata.utils import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--variables', type=str, nargs='+', default=['z500', 't500', 'r500', 'orography'])
    parser.add_argument('--years', type=int, nargs='+', default=[1990])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_parallel_shards', type=int, default=3)

    args = parser.parse_args()

    dataset = WeatherBench(args.path, args.variables, args.years, shards_per_year=12, shuffle=True)
    task = atmodata.utils.SequentialTransform(
        ForecastingTask(10, 6, crop_size={'lat': 96, 'lon': 96}, crops_per_sample=4),
        atmodata.iter.XrVariableSelecter.as_transform(
            {
                'predict': ['z500', 't500', 'r500'],
                'auxiliary': ['orography'],
                'lat': ['lat'],
            }
        ),
    )

    builder = AtmodataPipeBuilder(
        dataset,
        task,
        batch_size=args.batch_size,
        num_parallel_shards=args.num_parallel_shards,
        dataloading_prefetch_cnt=6,
        device_prefetch_cnt=2,
    )
    dataloader = builder.multiprocess(args.num_workers).transfer_to_device('cuda').build_dataloader()

    benchmark(dataloader, 0)


if __name__ == '__main__':
    main()
