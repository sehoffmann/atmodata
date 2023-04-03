import sys

import atmodata

import numpy as np
import pytest
from atmodata.iter import find_normalization_pipe, get_denorm_function, NormalizationPipe, XrZScoreNormalization
from fakedata import create_fake_dataset
from torchdata.datapipes.iter import IterableWrapper


class FakeNorm(NormalizationPipe):
    def normalize(self, x):
        return 42

    def denormalize(self, x):
        return 21


class TestNormalization:
    def test_find_normalization_pipe(self):
        pipe = IterableWrapper(range(10))
        assert find_normalization_pipe(pipe) is None

        norm_pipe = FakeNorm(pipe)
        pipe = norm_pipe.prefetch(1)
        assert find_normalization_pipe(pipe) is norm_pipe

    def test_get_denorm_function(self):
        pipe = IterableWrapper(range(10))
        denorm_function = get_denorm_function(pipe)
        assert denorm_function(123) == 123

        norm_pipe = FakeNorm(pipe)
        pipe = norm_pipe.prefetch(1)
        assert get_denorm_function(pipe)(123) == 21

    def test_zscore(self):
        statistics = create_fake_dataset('t', value=0).isel(level=0, time=0)
        statistics['t.mean'] = statistics['t'].copy()
        statistics['t.std'] = statistics['t'].copy()
        del statistics['t']

        statistics['t.mean'].values[:] = 20.0
        statistics['t.std'].values[:] = 5.0

        # Test: Non-spatially resolved
        data = create_fake_dataset('t', value=40.0)
        pipe = IterableWrapper([data])
        pipe = XrZScoreNormalization(pipe, statistics, spatially_resolved=False)
        result = list(pipe)[0]
        np.testing.assert_allclose(result['t'].values, 4.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
