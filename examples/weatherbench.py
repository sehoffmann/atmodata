import atmodata  # noqa: F401
import torchdata.datapipes as dp
from atmodata.datasets import WeatherbenchPathBuilder
from atmodata.utils import benchmark
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService


base_dir = '/mnt/qb/datasets/WeatherBench/1.40625deg'


def open_variable(var, pipe):
    pipe = WeatherbenchPathBuilder(pipe, base_dir, var)
    return pipe.xr_open().xr_get_variables(var).xr_sel(level=500).xr_split_dim('time', 12)


def extract_ts_crops(pipe):
    pipe = pipe.xr_extract_timeseries(10, 6, shuffle=True, shard=True)
    pipe = pipe.repeat(4).xr_random_crop({'lat': 96, 'lon': 96}, wraps={'lon': True})
    pipe = pipe.shuffle(buffer_size=16)

    return pipe


def main():
    years = [1990]
    variables = ['z', 't', 'r']
    N_vars = len(variables)
    N_workers = 4

    pipes = dp.iter.IterableWrapper(years).fork(N_vars)
    pipe = dp.iter.Zipper(*[open_variable(var, dp) for var, dp in zip(variables, pipes)])
    pipe = pipe.sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
    pipe = pipe.xr_prefetch(buffer_size=0).xr_merge().share_memory().prefetch(6)

    pipe = pipe.repeat(N_workers).sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
    pipe = pipe.round_robin_transform(3, extract_ts_crops)  # sample from 3 months at the same time
    pipe = pipe.xr_to_numpy().batch(32).collate()
    pipe = pipe.non_replicable().prefetch(N_workers * 1)
    pipe = pipe.th_to_device('cuda').th_interleave_batches(N_workers)
    pipe = pipe.prefetch(2)

    rs = MultiProcessingReadingService(num_workers=N_workers)
    dl = DataLoader2(pipe, reading_service=rs)

    benchmark(dl, 0)


if __name__ == '__main__':
    main()
