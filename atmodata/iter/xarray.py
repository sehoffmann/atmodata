import concurrent
import random
from concurrent.futures import ThreadPoolExecutor

import torch
import xarray as xr
from torch.utils.data import functional_datapipe, IterDataPipe


def _as_iterable(data_arrays):
    try:
        iter(data_arrays)
    except TypeError:
        return list(data_arrays)
    finally:
        return data_arrays


@functional_datapipe("xr_open")
class XrOpener(IterDataPipe):
    def __init__(self, dp, **kwargs):
        self.dp = dp
        self.kwargs = kwargs

    def __iter__(self):
        for path in self.dp:
            yield xr.open_dataset(path, **self.kwargs)


@functional_datapipe("xr_load")
class XrLoader(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for ds in self.dp:
            yield ds.load()


@functional_datapipe("xr_get_variables")
class XrVariableGetter(IterDataPipe):
    def __init__(self, dp, variables):
        self.dp = dp
        self.variables = variables

    def __iter__(self):
        for ds in self.dp:
            yield ds[self.variables]


@functional_datapipe("xr_sel")
class XrSelecter(IterDataPipe):
    def __init__(self, dp, **selects):
        self.dp = dp
        self.selects = selects

    def __iter__(self):
        for ds in self.dp:
            yield ds.sel(**self.selects)


@functional_datapipe("xr_isel")
class XrISelecter(IterDataPipe):
    def __init__(self, dp, **selects):
        self.dp = dp
        self.selects = selects

    def __iter__(self):
        for ds in self.dp:
            yield ds.isel(**self.selects)


@functional_datapipe("xr_merge")
class XrMerge(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for data_arrays in self.dp:
            yield xr.merge(data_arrays)


@functional_datapipe("xr_split_dim")
class XrSplitDim(IterDataPipe):
    def __init__(self, dp, dim, splits):
        self.dp = dp
        self.splits = splits
        self.dim = dim

    def __iter__(self):
        for ds in self.dp:
            size = len(ds.coords[self.dim]) // self.splits
            for i in range(self.splits):
                yield ds.isel(**{self.dim: slice(i * size, (i + 1) * size)})


@functional_datapipe("xr_prefetch")
class XrPrefetcher(IterDataPipe):
    def __init__(self, dp, buffer_size=1, n_threads=4, executor=None):
        self.dp = dp
        self.buffer_size = buffer_size
        self.n_threads = n_threads
        self.executor = executor
        self.buffer = []

    def __iter__(self):
        executor_cls = self.executor if self.executor is not None else ThreadPoolExecutor
        it = iter(self.dp)
        with executor_cls(self.n_threads) as executor:
            try:
                for _ in range(self.buffer_size + 1):
                    data_arrays = _as_iterable(next(it))
                    self.buffer.append([executor.submit(XrPrefetcher._ld, da) for da in data_arrays])

                while True:
                    futures = self.buffer.pop(0)
                    concurrent.futures.wait(futures)

                    if len(futures) == 1:
                        yield futures[0].result()
                    else:
                        yield [future.result() for future in futures]

                    data_arrays = _as_iterable(next(it))
                    self.buffer.append([executor.submit(XrPrefetcher._ld, da) for da in data_arrays])

            except StopIteration:
                while self.buffer:
                    futures = self.buffer.pop(0)
                    concurrent.futures.wait(futures)

                    if len(futures) == 1:
                        yield futures[0].result()
                    else:
                        yield [future.result() for future in futures]

    @staticmethod
    def _ld(arr):
        return arr.load()


@functional_datapipe("xr_unroll_indices")
class XrUnrollIndices(IterDataPipe):
    def __init__(self, dp, dim, shuffle=False):
        self.dp = dp
        self.dim = dim
        self.shuffle = shuffle
        self._shuffle = shuffle
        self._rng = random.Random()
        self._seed = None

    def set_shuffle(self, shuffle=True):
        self._shuffle = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self):
        for i, ds in enumerate(self.dp):
            if self._seed is not None:
                self._rng.seed(self._seed + i)
            else:
                self._rng.seed(int(torch.empty((), dtype=torch.int64).random_().item()))

            indices = list(range(len(ds.coords[self.dim])))
            if self.shuffle and self._shuffle:
                self._rng.shuffle(indices)

            for idx in indices:
                yield (idx, ds)

    def __getstate__(self):
        state = (self.dp, self.dim, self.shuffle, self._shuffle, self._seed, self._rng.getstate())
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        self.dp, self.dim, self.shuffle, self._shuffle, self._seed, rng_state = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)


@functional_datapipe("xr_extract_timeseries")
class XrExtractTimeseries(IterDataPipe):
    def __init__(self, dp, steps, rate, dim='time'):
        self.dp = dp
        self.steps = steps
        self.rate = rate
        self.dim = dim

    def __iter__(self):
        T = self.steps * self.rate
        for idx, ds in self.dp:
            N = len(ds.variables[self.dim])
            if idx + T > N:
                continue
            else:
                yield ds.isel(**{self.dim: slice(idx, idx + T, self.rate)})


@functional_datapipe("xr_random_crop")
class XrRandomCrop(IterDataPipe):
    def __init__(self, dp, crop_size, bounds={}, wraps={}):
        self.dp = dp
        self.crop_size = crop_size
        self.bounds = bounds
        self.wraps = wraps

    def __iter__(self):
        for ds in self.dp:
            indices = self._sample_indices(ds)
            overlaps = {dim: indices[dim] + self.crop_size[dim] > len(ds.variables[dim]) for dim in indices}
            if not any(overlaps.values()):
                # fast path:
                yield ds.isel(**{dim: slice(indices[dim], indices[dim] + size) for dim, size in self.crop_size.items()})
            else:
                # slow path:
                rolled = ds.roll(**{dim: -idx for dim, idx in indices.items()}, roll_coords=True)
                yield rolled.isel(**{dim: slice(0, size) for dim, size in self.crop_size.items()})

    def _sample_indices(self, ds):
        indices = {}
        for dim in self.crop_size:
            index = ds.variables[dim]
            if dim in self.bounds:
                l = self.find_closest_index(index, self.bounds[dim][0])
                r = self.find_closest_index(index, self.bounds[dim][1])
            else:
                l, r = 0, len(index)

            if not self.wraps.get(dim, False) and r + self.crop_size[dim] > len(index):
                r = len(index) - self.crop_size[dim]

            indices[dim] = random.randint(l, r)

        return indices

    def _find_closest_index(self, index, label):
        candidate = index.searchsorted(label)
        if candidate == len(index) - 1:
            return candidate
        elif index[candidate + 1] - label < label - index[candidate]:
            return candidate + 1
        else:
            return candidate


@functional_datapipe("xr_to_numpy")
class XrToNumpy(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for ds in self.dp:
            if isinstance(ds, xr.Dataset):
                yield ds.to_array().to_numpy()
            else:
                yield ds.to_numpy()
