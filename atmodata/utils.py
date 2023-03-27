import time

import numpy as np
import torch


class PipeTransform:
    def __init__(self, transform_fn, *args, **kwargs):
        self.transform_fn = transform_fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dp):
        return self.transform_fn(dp, *self.args, **self.kwargs)


@classmethod
def as_transform(pipe_cls: type, *args, **kwargs):
    return PipeTransform(pipe_cls, *args, **kwargs)


class SequentialTransform:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, dp):
        for transform_fn in self.transforms:
            dp = transform_fn(dp)
        return dp


def collate_coordinates(coords, all_coordinates, no_scalar=False):
    """
    Tries to find the most efficient form of indexing for the given coordinates to be used with .isel().
    Either returns a scalar index, a slice (of indices) if possible, or a sorted list of indices.
    """
    if len(coords) == 0:
        raise ValueError('Empty coordinates given')

    if not all(c in all_coordinates for c in coords):
        raise ValueError('Invalid coordinates given')

    indices = list(all_coordinates.index(c) for c in sorted(coords))
    if len(indices) == 1:
        if no_scalar:
            return slice(indices[0], indices[0] + 1)
        else:
            return indices[0]

    step = indices[1] - indices[0]
    if all(idx == indices[0] + i * step for i, idx in enumerate(indices)):
        return slice(indices[0], indices[-1] + 1, step)
    else:
        return indices


def benchmark(dataset, process_time=0, log=True, print_frequency=100):
    it = iter(dataset)
    times = []
    total_size = 0
    i = 0
    while True:
        ts = time.time()
        batch = next(it, None)
        if batch is None:
            break

        times.append(time.time() - ts)
        if isinstance(batch, torch.Tensor):
            total_size += batch.nelement() * batch.element_size()
        elif isinstance(batch, dict):
            total_size += sum(v.nelement() * v.element_size() for v in batch.values())
        elif isinstance(batch, (list, tuple)):
            total_size += sum(v.nelement() * v.element_size() for v in batch)
        i += 1

        if process_time:
            time.sleep(process_time / 1000)

        if log and print_frequency and i % print_frequency == 0:
            last_vals = times[-print_frequency:]
            print(f'{1000 * np.mean(last_vals):.0f} ms/batch')

    del it
    times = np.array(times)

    if log:
        print('----------------------------------------------')
        print(f'{np.sum(times):.0f}s total')
        print(f'{times[0]:.1f}s latency')
        print(f'Throughput: {total_size / np.sum(times) / 1e6:.1f} MB/s')
        print(f'Mean: {np.mean(times)*1000:.0f} ms/batch')
        print(f'Std: {np.std(times)*1000:.0f} ms')
        print(f'Median: {np.median(times)*1000:.0f} ms/batch')
        print(f'25th percentile: {np.percentile(times, 25)*1000:.0f} ms/batch')
        print(f'75th percentile: {np.percentile(times, 75)*1000:.0f} ms/batch')
        print(f'99th percentile: {np.percentile(times, 99)*1000:.0f} ms/batch')

    return times
