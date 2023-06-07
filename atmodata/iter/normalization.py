'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import xarray as xr
from torchdata.dataloader2.graph.utils import list_dps, traverse_dps
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def find_normalization_pipe(dp):
    graph = traverse_dps(dp)
    pipes = [pipe for pipe in list_dps(graph) if isinstance(pipe, NormalizationPipe)]
    # ^ we can't use find_dps here because it matches types exactly
    if len(pipes) == 0:
        return None
    elif len(pipes) == 1:
        return pipes[0]
    else:
        raise ValueError('Multiple normalization pipes found in the data pipeline')


def _identity(x):
    return x


def get_denorm_function(dp):
    normalization_pipe = find_normalization_pipe(dp)
    if normalization_pipe is None:
        return _identity
    else:
        return normalization_pipe.denormalize


class NormalizationPipe(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def normalize(self, x):
        raise NotImplementedError()

    def denormalize(self, x):
        raise NotImplementedError()

    def __iter__(self):
        for x in self.dp:
            yield self.normalize(x)


class XrManualZScoreNormalization(NormalizationPipe):
    def __init__(self, dp, means, stds, inplace=False):
        super().__init__(dp)
        self.means = means
        self.stds = stds
        self.inplace = inplace

    def normalize(self, x):
        if not self.inplace:
            x = x.copy()
        for var in x.data_vars:
            if var not in self.means:
                continue
            data = x[var] if self.inplace else x[var].copy()
            data -= self.means[var]
            data /= self.stds[var]
            x[var] = data
        return x

    def denormalize(self, x):
        if not self.inplace:
            x = x.copy()
        for var in x.data_vars:
            if var not in self.means:
                continue
            data = x[var] if self.inplace else x[var].copy()
            data *= self.stds[var]
            data += self.means[var]
        return x


@functional_datapipe('xr_normalize_zscore')
class XrZScoreNormalization(NormalizationPipe):
    def __init__(self, dp, statistics, spatially_resolved=False, variables=None):
        super().__init__(dp)

        if not isinstance(statistics, xr.Dataset):
            self.statistics = xr.open_dataset(statistics)
        else:
            self.statistics = statistics

        self.spatially_resolved = spatially_resolved
        self._variables = variables
        if self._variables:
            self._load_statistics(self._variables)
        else:
            self.means = None
            self.stds = None

    @property
    def is_loaded(self):
        return self.means is not None and self.stds is not None

    def _load_statistics(self, variables):
        means = self.statistics[[f'{var}.mean' for var in variables]]
        stds = self.statistics[[f'{var}.std' for var in variables]]
        if not self.spatially_resolved:
            means = means.mean(dim=['lat', 'lon'])
            stds = stds.mean(dim=['lat', 'lon'])

        self.means = means.rename_vars({f'{var}.mean': var for var in variables})
        self.stds = stds.rename_vars({f'{var}.std': var for var in variables})

        self.means = self.means.load()
        self.stds = self.stds.load()

    def normalize(self, x):
        if self._variables is None:
            self._load_statistics(list(x.data_vars))

        return (x - self.means) / self.stds

    def denormalize(self, x):
        if not self.is_loaded:
            raise ValueError(
                'Normalization statistics not loaded. Call normalize first or explicitely specify variables.'
            )
        return x * self.stds + self.means
