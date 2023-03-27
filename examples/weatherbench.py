import atmodata  # noqa: F401
from atmodata.builder import AtmodataPipeBuilder
from atmodata.datasets import WeatherBench
from atmodata.tasks import ForecastingTask
from atmodata.utils import benchmark


base_dir = '/mnt/qb/datasets/WeatherBench/1.40625deg'


def main():
    years = [1990]
    variables = ['z500', 't500', 'r500', 'orography']
    N_workers = 4

    dataset = WeatherBench(base_dir, variables, years, shards_per_year=12, shuffle=True)
    task = ForecastingTask(10, 6, crop_size={'lat': 96, 'lon': 96}, crops_per_sample=4)

    builder = AtmodataPipeBuilder(
        dataset,
        task,
        batch_size=32,
        num_parallel_shards=3,
        dataloading_prefetch_cnt=6,
        device_prefetch_cnt=2,
    )
    dataloader = builder.multiprocess(N_workers).transfer_to_device('cuda').build_dataloader()

    benchmark(dataloader, 0)


if __name__ == '__main__':
    main()
