import os
import pickle
import unittest

import numpy as np
import pytest
import torch
from atmodata.serialization import _get_offset, _get_owning_base, _have_same_memory, ForkingPickler
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper


class TestSerialization:
    @staticmethod
    def roundtrip(arr):
        serialized = ForkingPickler.dumps(arr)
        return pickle.loads(serialized)

    def _assert_shared(self, rebuilt):
        base = _get_owning_base(rebuilt)
        assert isinstance(rebuilt, np.ndarray)
        assert isinstance(base, torch.Tensor)
        assert base.is_shared()
        return rebuilt

    def test_get_owning_base(self):
        arr = np.arange(10)
        assert _get_owning_base(arr) is arr
        assert _get_owning_base(arr[2:5]) is arr
        assert _get_owning_base(arr[2:5][0:1]) is arr

        tensor = torch.arange(10)
        np_view = np.asarray(tensor)
        assert isinstance(_get_owning_base(np_view), torch.Tensor)
        assert _have_same_memory(_get_owning_base(np_view), tensor)
        assert _have_same_memory(_get_owning_base(np_view[2:5]), tensor)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_basic_contiguous(self, order):
        # Test: basic serialization of nparray via shared memory
        arr = np.arange(10 * 10).reshape(10, 10)
        arr = np.array(arr, order=order)
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_contiguous_slice(self, order):
        arr = np.arange(10 * 10).reshape(10, 10)
        arr = np.array(arr, order=order)[2:5, 3:7]
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_noncontiguous_slice(self, order):
        arr = np.arange(10 * 10).reshape(10, 10)
        arr = np.array(arr, order=order)[2:5:4, 3:9:2]
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

    def test_0d(self):
        arr = np.array(24.2)
        rebuilt = self.roundtrip(arr)
        np.testing.assert_array_equal(rebuilt, arr)

    def test_dtypes(self):
        if hasattr(np, 'float128'):
            arr = np.arange(20).astype(np.float128).reshape(2, 5, 2)
            rebuilt = self.roundtrip(arr)
            self._assert_shared(rebuilt)
            np.testing.assert_array_equal(rebuilt, arr)

        arr = np.arange(20).astype('datetime64[ns]')
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

        # TODO: expand this test to cover more complex dtypes
        #       e.g. structured dtypes, flexible dtypes, etc.
        #       see: https://numpy.org/doc/stable/reference/arrays.dtypes.html

        # Test: non-shareable dtypes
        arr = np.asarray(['a', 'bc', 'def'])
        rebuilt = self.roundtrip(arr)
        np.testing.assert_array_equal(rebuilt, arr)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_nonzero_offset(self, order):
        arr = np.arange(10 * 10).reshape(10, 10)
        arr = np.array(arr, order=order)
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

        arr2 = rebuilt[2:5, 3:7]
        assert _get_offset(arr2) > 0  # we expect arr2 to have nonzero offset
        rebuilt2 = self.roundtrip(arr2)
        self._assert_shared(rebuilt2)
        np.testing.assert_array_equal(rebuilt2, arr2)

    @pytest.mark.parametrize('context', ['fork', 'spawn'])
    def test_readingservice_smoke(self, context):
        if context == 'fork' and not hasattr(os, 'fork'):
            pytest.skip("Forking is not supported on this platform")

        arr = np.arange(10 * 10).reshape(10, 10)
        dp = IterableWrapper([arr])
        dp = dp.share_memory()
        dp = dp.map(self._assert_shared)
        dp = dp.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
        rs = MultiProcessingReadingService(2, multiprocessing_context=context)
        dl = DataLoader2(dp, reading_service=rs)
        result = list(dl)[0]
        self._assert_shared(result)


if __name__ == '__main__':
    unittest.main()
