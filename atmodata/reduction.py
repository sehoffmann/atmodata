import warnings

import numpy as np
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
    diffs = data - mean
    return (diffs * diffs).sum(axis=axis)


class Reducer:
    """
    Incrementally reduces a dataset along a given axis.
    Computes mean, variance, min, max, and optionally PCA.
    """

    def __init__(self, axis=None, pca=False, pca_components=None, pca_batch_size=1000):
        self.axis = axis
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

    def update(self, data):
        if self.axis >= data.ndim:
            raise ValueError(f"Axis {self.axis} is out of bounds for data with shape {data.shape}")

        reduced_shape = data.shape[: self.axis] + data.shape[self.axis + 1 :]
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
        k = data.shape[self.axis]

        if self._pca is None:
            data_mean = data.mean(axis=self.axis)
            data_M2 = unnormalized_variance(data, np.expand_dims(data_mean, self.axis), self.axis)
        data_min = data.min(axis=self.axis)
        data_max = data.max(axis=self.axis)

        if self._pca is not None:
            flattened = np.moveaxis(data, self.axis, 0).reshape((k, -1))
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
