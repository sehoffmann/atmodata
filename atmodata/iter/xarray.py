'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import concurrent
import random
from concurrent.futures import ThreadPoolExecutor

import torch
import xarray as xr
from torch.utils.data import functional_datapipe, IterDataPipe


def _as_iterable(x):
    is_iterable = False
    try:
        iter(x)
        is_iterable = True
    except TypeError:
        pass

    if is_iterable and not isinstance(x, str):
        return x
    else:
        return [x]


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


@functional_datapipe("xr_select_variables")
class XrVariableSelecter(IterDataPipe):
    """
    Selects a subset of variables from a xr.Dataset.
    """

    def __init__(self, dp, variables):
        """
        Variables can be a string, a list of strings, or a dictionary mapping new names to variable names.
        If a string is given, the variable is returned as a xr.DataArray.
        If a list of strings is given, the variables are returned as a xr.Dataset.
        If a dictionary is given, the variables are returned as a dictionary mapping the keys to either xr.DataArray or xr.Dataset.
        """
        self.dp = dp
        self.variables = variables

    def _select_vars(self, ds, vars):
        ds_new = ds.copy()  # only copies metadata
        new_vars = []

        to_data_array = False
        if isinstance(vars, str):
            vars = [vars]
            to_data_array = True

        for var in vars:
            if var in ds.coords:
                ds_new[var + '_coord'] = ds[var]
                new_vars.append(var + '_coord')
            else:
                new_vars.append(var)

        if to_data_array:
            return ds_new[new_vars[0]]
        else:
            return ds_new[new_vars]

    def __iter__(self):
        for ds in self.dp:
            if isinstance(self.variables, dict):
                yield {k: self._select_vars(ds, vars) for k, vars in self.variables.items()}
            else:
                yield self._select_vars(ds, self.variables)


@functional_datapipe("xr_sel")
class XrSelecter(IterDataPipe):
    def __init__(self, dp, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        self.dp = dp
        self.indexers = indexers
        self.method = method
        self.tolerance = tolerance
        self.drop = drop
        self.indexers_kwargs = indexers_kwargs

    def __iter__(self):
        for ds in self.dp:
            yield ds.sel(
                indexers=self.indexers,
                method=self.method,
                tolerance=self.tolerance,
                drop=self.drop,
                **self.indexers_kwargs,
            )


@functional_datapipe("xr_isel")
class XrISelecter(IterDataPipe):
    def __init__(self, dp, indexers=None, drop=False, missing_dims='raise', **indexers_kwargs):
        self.dp = dp
        self.indexers = indexers
        self.drop = drop
        self.missing_dims = missing_dims
        self.indexers_kwargs = indexers_kwargs

    def __iter__(self):
        for ds in self.dp:
            yield ds.isel(
                indexers=self.indexers, drop=self.drop, missing_dims=self.missing_dims, **self.indexers_kwargs
            )


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


@functional_datapipe("xr_parallel_load")
class XrParallelLoader(IterDataPipe):
    def __init__(self, dp, n_threads=4, executor=None):
        self.dp = dp
        self.n_threads = n_threads
        self.executor_cls = executor if executor else ThreadPoolExecutor

    def __iter__(self):
        with self.executor_cls(self.n_threads) as executor:
            for data_arrays in self.dp:
                futures = tuple(executor.submit(xr.DataArray.load, da) for da in data_arrays)
                concurrent.futures.wait(futures)
                yield tuple(future.result() for future in futures)


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
        else:
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


@functional_datapipe("xr_to_array")
class XrToArray(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for ds in self.dp:
            if isinstance(ds, xr.Dataset):
                yield ds.to_array()
            else:
                yield ds


@functional_datapipe("xr_to_numpy")
class XrToNumpy(IterDataPipe):
    """
    Converts a possibly nested collection of xarray `Dataset` or `DataArray` to `numpy.array`.
    Empty Datasets and DataArrays are returned as np.array(nan). Elements that are not Datasets or DataArrays are left unchanged.
    Datasets are flattened to arrays using `Dataset.to_array()`, which stacks all variables along a new dimension.
    """

    def __init__(self, dp):
        self.dp = dp.nested_map(self._to_numpy)

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, xr.Dataset):
            if x.data_vars:
                return x.to_array().to_numpy()
            else:
                return xr.DataArray().to_numpy()
        elif isinstance(x, xr.DataArray):
            return x.to_numpy()
        else:
            return x

    def __iter__(self):
        return iter(self.dp)


@functional_datapipe('xr_rename')
class XrRenamer(IterDataPipe):
    def __init__(self, dp, new_name_or_name_dict=None, **names):
        self.dp = dp
        if isinstance(new_name_or_name_dict, str):
            self.name_dict = new_name_or_name_dict
        else:
            self.name_dict = {} if new_name_or_name_dict is None else dict(new_name_or_name_dict)
            self.name_dict.update(names)
            for k, v in list(self.name_dict.items()):
                if k == v:
                    del self.name_dict[k]

    def __iter__(self):
        for ds in self.dp:
            yield ds.rename(self.name_dict)


@functional_datapipe('xr_rename_vars')
class XrVarRenamer(IterDataPipe):
    def __init__(self, dp, name_dict=None, **names):
        self.dp = dp
        if name_dict is None:
            self.name_dict = {}
        else:
            self.name_dict = dict(name_dict)
        self.name_dict.update(names)
        for k, v in list(self.name_dict.items()):
            if k == v:
                del self.name_dict[k]

    def __iter__(self):
        for ds in self.dp:
            yield ds.rename_var(self.name_dict)


@functional_datapipe('xr_rename_dims')
class XrDimRenamer(IterDataPipe):
    def __init__(self, dp, dims_dict=None, **dims):
        self.dp = dp
        if dims_dict is None:
            self.dims_dict = {}
        else:
            self.dims_dict = dict(dims_dict)
        self.dims_dict.update(dims)
        for k, v in list(self.dims_dict.items()):
            if k == v:
                del self.dims_dict[k]

    def __iter__(self):
        for ds in self.dp:
            yield ds.rename(self.dims_dict)


@functional_datapipe('xr_broadcast')
class XrBroadcaster(IterDataPipe):
    """
    Calls xr.broadcast on each element of the input datapipe.
    If used on a single dataset, this will guarantee that all variables have the same dimensions.
    """

    def __init__(self, dp, **kwargs):
        self.dp = dp
        self.kwargs = kwargs

    def __iter__(self):
        for tupel_or_ds in self.dp:
            if isinstance(tupel_or_ds, (xr.Dataset, xr.DataArray)):
                yield xr.broadcast(tupel_or_ds, **self.kwargs)[0]
            else:
                yield xr.broadcast(*tupel_or_ds, **self.kwargs)


@functional_datapipe('xr_transpose')
class XrTransposer(IterDataPipe):
    def __init__(self, dp, *dims):
        self.dp = dp
        self.dims = dims

    def __iter__(self):
        for ds in self.dp:
            yield ds.transpose(*self.dims)


@functional_datapipe('xr_astype')
class XrAstype(IterDataPipe):
    def __init__(self, dp, dtype):
        self.dp = dp
        self.dtype = dtype

    def __iter__(self):
        for ds in self.dp:
            yield ds.astype(self.dtype)
