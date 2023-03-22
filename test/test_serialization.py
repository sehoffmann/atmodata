import pickle
import unittest

import numpy as np

import pytest
import torch
from atmodata.serialization import _get_owning_base, _have_same_memory, ForkingPickler, reduce_ndarray
from expecttest import TestCase


class TestSerialization(TestCase):
    @staticmethod
    def roundtrip(arr):
        serialized = ForkingPickler.dumps(arr)
        return pickle.loads(serialized)

    def _assert_shared(self, rebuilt):
        base = _get_owning_base(rebuilt)
        self.assertIsInstance(rebuilt, np.ndarray)
        self.assertIsInstance(base, torch.Tensor)
        self.assertTrue(base.is_shared())

    def test_get_owning_base(self):
        arr = np.arange(10)
        self.assertIs(_get_owning_base(arr), arr)
        self.assertIs(_get_owning_base(arr[2:5]), arr)
        self.assertIs(_get_owning_base(arr[2:5][0:1]), arr)

        tensor = torch.arange(10)
        np_view = np.asarray(tensor)
        self.assertIsInstance(_get_owning_base(np_view), torch.Tensor)
        self.assertTrue(_have_same_memory(_get_owning_base(np_view), tensor))
        self.assertTrue(_have_same_memory(_get_owning_base(np_view[2:5]), tensor))

    def test_basic_contiguous(self):
        # Test: basic serialization of nparray via shared memory
        arr = np.arange(10 * 10).reshape(10, 10)
        rebuilt = self.roundtrip(arr)

        base = _get_owning_base(rebuilt)
        self.assertIsInstance(rebuilt, np.ndarray)
        self.assertIsInstance(base, torch.Tensor)
        self.assertTrue(base.is_shared())
        np.testing.assert_array_equal(rebuilt, arr)

    def test_contiguous_slice(self):
        arr = np.arange(10 * 10).reshape(10, 10)[2:5, 3:7]
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

    def test_noncontiguous_slice(self):
        arr = np.arange(10 * 10).reshape(10, 10)[2:5:4, 3:9:2]
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

    def test_0d(self):
        arr = np.array(24.2)
        rebuilt = self.roundtrip(arr)
        np.testing.assert_array_equal(rebuilt, arr)

    def test_dtypes(self):
        arr = np.arange(20).astype(np.float128).reshape(2, 5, 2)
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

        arr = np.arange(20).astype('datetime64[ns]')
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

        # Test: non-sharable dtypes
        arr = np.asarray(['a', 'bc', 'def'])
        rebuilt = self.roundtrip(arr)
        np.testing.assert_array_equal(rebuilt, arr)


if __name__ == '__main__':
    unittest.main()
