import atmodata  # noqa: F401
from atmodata.datasets import Weatherbench
from atmodata.tasks import ForecastingTask
from atmodata.utils import benchmark
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService


base_dir = '/mnt/qb/datasets/WeatherBench/1.40625deg'


def main():
    years = [1990]
    variables = ['z500', 't500', 't600', 'r500', 'r100']
    N_workers = 4

    dataset = Weatherbench(base_dir, variables, years, shards_per_year=12, shuffle=True)
    task = ForecastingTask(10, 6, crop_size={'lat': 96, 'lon': 96}, crops_per_sample=4)

    pipe = dataset.share_memory().prefetch(6)
    pipe = pipe.repeat(N_workers).sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
    pipe = pipe.round_robin_transform(3, task)  # sample from 3 months at the same time
    pipe = pipe.xr_to_numpy().batch(32).collate()
    pipe = pipe.non_replicable().prefetch(N_workers * 1)
    pipe = pipe.th_to_device('cuda').th_interleave_batches(N_workers)
    pipe = pipe.prefetch(2)

    rs = MultiProcessingReadingService(num_workers=N_workers)
    dl = DataLoader2(pipe, reading_service=rs)

    benchmark(dl, 0)


if __name__ == '__main__':
    main()
