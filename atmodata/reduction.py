import warnings
from functools import cached_property

import numpy as np
import tqdm
import xarray as xr
from sklearn.decomposition import IncrementalPCA


def combine_means(n_a, mean_a, n_b, mean_b):
    n = n_a + n_b
    return mean_a + (n_b / n) * (mean_b - mean_a)


def combine_M2(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    n = n_a + n_b
    delta = mean_b - mean_a
    M2_ab = M2_a + M2_b + delta**2 * n_a * n_b / n
    return M2_ab


def combine_mean_and_M2(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    n = n_a + n_b
    delta = mean_b - mean_a
    alpha = n_b / n
    M2_ab = M2_a + M2_b + delta * delta * n_a * alpha
    mean_ab = mean_a + alpha * delta
    return mean_ab, M2_ab


def unnormalized_variance(data, mean, axis=None):
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
    diffs = data - mean
    return (diffs * diffs).sum(axis=axis)


def make_compound_variable(var_name, dim_name):
    return f'{var_name}.{dim_name}'


def split_compound_variable(variable):
    splitted = variable.split('.')
    if len(splitted) != 2:
        raise ValueError(f'Invalid compound variable: {variable}')
    else:
        return splitted[0], splitted[1]


def is_compound_variable(variable):
    return '.' in variable


class Reducer:
    """
    Incrementally reduces arrays along a given axis.
    Computes mean, variance, min, max, and optionally PCA.
    """

    DEFAULT = tuple()

    def __init__(self, default_axis=None, pca=False, pca_components=None, pca_batch_size=1000):
        self.default_axis = default_axis
        if pca:
            self.num_pca_components = pca_components
            self._pca = IncrementalPCA(pca_components, batch_size=pca_batch_size)
        else:
            self.num_pca_components = None
            self._pca = None

        self.shape = None
        self.num_samples = 0
        self._mean = None
        self._M2 = None
        self.min = None
        self.max = None

    @property
    def pca(self):
        return self._pca is not None

    @property
    def mean(self):
        if self.num_samples == 0:
            return None
        if self.pca:
            return self._pca.mean_.reshape(self.shape)
        return self._mean

    @property
    def var(self):
        if self.num_samples == 0:
            return None
        if self.pca:
            return self._pca.var_.reshape(self.shape)
        return self._M2 / self.num_samples

    @property
    def std(self):
        if self.num_samples == 0:
            return None
        return np.sqrt(self.var)

    @property
    def pca_components(self):
        if not self.pca:
            return None
        return self._pca.components_.reshape((-1,) + self.shape)

    @property
    def pca_explained_variance(self):
        if not self.pca:
            return None
        return self._pca.explained_variance_

    @property
    def pca_explained_variance_ratio(self):
        if not self.pca:
            return None
        return self._pca.explained_variance_ratio_

    @property
    def pca_singular_values(self):
        if not self.pca:
            return None
        return self._pca.singular_values_

    @property
    def pca_noise_variance(self):
        if not self.pca:
            return None
        return self._pca.noise_variance_

    def update(self, data, axis=DEFAULT):
        if axis is self.DEFAULT:
            axis = self.default_axis

        if axis is not None and axis >= data.ndim:
            raise ValueError(f"Axis {axis} is out of bounds for data with shape {data.shape}")

        if axis is None:
            reduced_shape = ()
        else:
            reduced_shape = data.shape[:axis] + data.shape[axis + 1 :]

        if self.shape is None:
            self.shape = reduced_shape
        elif self.shape != reduced_shape:
            raise ValueError(f"Shape after reduction {reduced_shape} does not match expected shape {self.shape}")

        num_pca_components = np.prod(self.shape)
        if self.pca and not self.num_pca_components and num_pca_components >= 3e5:
            warnings.warn(
                f'PCA will result in an exorbitant number of components ({num_pca_components}). Explicitely set the number of components to supress this warning.'
            )

        N_old = self.num_samples
        k = int(np.prod(data.shape) // np.prod(reduced_shape))

        if self._pca is None:
            data_mean = data.mean(axis=axis)
            data_M2 = unnormalized_variance(data, data_mean, axis)
        data_min = data.min(axis=axis)
        data_max = data.max(axis=axis)

        if self._pca is not None:
            flattened = np.moveaxis(data, axis, 0).reshape((k, -1))
            self._pca.partial_fit(flattened)

        if self.num_samples == 0:
            if self._pca is None:
                self._mean = data_mean
                self._M2 = data_M2
            self.min = data_min
            self.max = data_max
        else:
            if self._pca is None:
                self._mean, self._M2 = combine_mean_and_M2(N_old, self.mean, self._M2, k, data_mean, data_M2)
            self.min = np.minimum(self.min, data_min)
            self.max = np.maximum(self.max, data_max)

        self.num_samples = N_old + k


class DatasetReducer:
    def __init__(self, dim=None, variables=None, dtype=np.float32, pca=False, pca_components=None, pca_batch_size=1000):
        self.dim = dim
        self.variables = variables
        self.dtype = dtype
        self.pca = pca
        self.pca_components = pca_components
        self.pca_batch_size = pca_batch_size

        self.n_updates = 0
        self.reducers = {}
        self.coords = {}
        self.reduced_dims = {}

    def update(self, ds: xr.Dataset):
        self._clear_statistics()
        self.n_updates += 1
        variables = self.variables if self.variables is not None else ds.data_vars
        for var in variables:
            data_array = ds[var]
            axis = None if self.dim is None else data_array.get_axis_num(self.dim)

            if var not in self.reducers:
                reducer = Reducer(pca=self.pca, pca_components=self.pca_components, pca_batch_size=self.pca_batch_size)
                self.reducers[var] = reducer
                if axis is None:
                    reduced_dims = tuple()
                else:
                    removed = np.atleast_1d(axis) % data_array.ndim
                    reduced_dims = tuple(dim for i, dim in enumerate(data_array.dims) if i not in removed)
                self.reduced_dims[var] = reduced_dims
                self.coords[var] = {d: data_array.coords[d] for d in reduced_dims}
            else:
                reducer = self.reducers[var]

            reducer.update(data_array.values, axis=axis)

    def _as_xr_array(self, var, values):
        return xr.DataArray(values.astype(self.dtype), coords=self.coords[var], dims=self.reduced_dims[var])

    def _pca_as_xr_array(self, var, values):
        coords = {'pca_component': np.arange(self.pca_components)}
        dims = ('pca_component',)
        if values.ndim > 1:
            coords.update(dict(self.coords[var]))
            dims = dims + self.reduced_dims[var]
        return xr.DataArray(values.astype(self.dtype), coords=coords, dims=dims)

    def _clear_statistics(self):
        try:
            del self.statistics
        except AttributeError:
            pass

    @cached_property
    def statistics(self):
        ds = xr.Dataset()
        for var, reducer in self.reducers.items():
            ds[make_compound_variable(var, 'mean')] = self._as_xr_array(var, reducer.mean)
            ds[make_compound_variable(var, 'std')] = self._as_xr_array(var, reducer.std)
            ds[make_compound_variable(var, 'min')] = self._as_xr_array(var, reducer.min)
            ds[make_compound_variable(var, 'max')] = self._as_xr_array(var, reducer.max)

            if self.pca:
                ds[make_compound_variable(var, 'pca_components')] = self._pca_as_xr_array(var, reducer.pca_components)
                ds[make_compound_variable(var, 'pca_singular_values')] = self._pca_as_xr_array(
                    var, reducer.pca_singular_values
                )
                ds[make_compound_variable(var, 'pca_variance')] = self._pca_as_xr_array(
                    var, reducer.pca_explained_variance
                )
                ds[make_compound_variable(var, 'pca_ratio')] = self._pca_as_xr_array(
                    var, reducer.pca_explained_variance_ratio
                )

        return ds


class StatisticsSaver:
    def __init__(self, dim='time', dtype=np.float32, daily=True, hourly=True, pca=True, pca_components=128):
        if hourly and not daily:
            raise ValueError('hourly=True can only be used in conjuction with daily=True')

        self.dim = dim
        self.global_reducer = DatasetReducer(dim=dim, dtype=dtype, pca=pca, pca_components=pca_components)
        if daily:
            self.daily_reducers = [DatasetReducer(dim=dim, dtype=dtype) for _ in range(365)]
        else:
            self.daily_reducers = None

        if hourly:
            self.hourly_reducers = [DatasetReducer(dim=dim, dtype=dtype) for _ in range(24)]
        else:
            self.hourly_reducers = None

    @property
    def daily(self):
        return self.daily_reducers is not None

    @property
    def hourly(self):
        return self.hourly_reducers is not None

    def _reduce_daily(self, ds):
        for dayofyear, grouped_ds in ds.groupby(f'{self.dim}.dayofyear'):
            if dayofyear == 366:
                dayofyear = 365
            self.daily_reducers[dayofyear - 1].update(grouped_ds)

    def _reduce_hourly(self, ds, daily_means):
        anomalies = ds.groupby('time.dayofyear') - daily_means
        for hour, grouped_ds in anomalies.groupby(f'{self.dim}.hour'):
            self.hourly_reducers[hour].update(grouped_ds)

    def process(self, dp):
        self._clear_statistics()

        # global & daily
        for ds in tqdm.tqdm(dp):
            self.global_reducer.update(ds)

            if self.daily:
                self._reduce_daily(ds)

        # hourly
        if self.hourly:
            daily_means = self.statistics[[var for var in self.statistics.data_vars if var.endswith('.daily_mean')]]
            daily_means = daily_means.rename({name: name[: -len('.daily_mean')] for name in daily_means.data_vars})

            for ds in tqdm.tqdm(dp):
                self._reduce_hourly(ds, daily_means)

            self._clear_statistics()

    def _clear_statistics(self):
        try:
            del self.statistics
        except AttributeError:
            pass

    @cached_property
    def statistics(self):
        datasets = [self.global_reducer.statistics]

        if self.daily:
            dailies = xr.concat([reducer.statistics for reducer in self.daily_reducers], dim='dayofyear')
            dailies.coords['dayofyear'] = np.arange(365) + 1
            new_names = {}
            for name in dailies.data_vars:
                var, stat = split_compound_variable(name)
                new_names[name] = make_compound_variable(var, f'daily_{stat}')
            dailies = dailies.rename(new_names)
            datasets += [dailies]

        if self.hourly and self.hourly_reducers[0].n_updates > 0:
            hourly = xr.concat([reducer.statistics for reducer in self.hourly_reducers], dim='hour')
            hourly.coords['hour'] = np.arange(24)
            new_names = {}
            for name in hourly.data_vars:
                var, stat = split_compound_variable(name)
                new_names[name] = make_compound_variable(var, f'hourly_{stat}')
            hourly = hourly.rename(new_names)
            datasets += [hourly]

        return xr.merge(datasets)

    def save(self, path):
        self.statistics.to_netcdf(path)
