import pickle
import unittest

import numpy as np

import pytest
import torch
from atmodata.serialization import _get_offset, _get_owning_base, _have_same_memory, ForkingPickler
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

    def test_nonzero_offset(self):
        arr = np.arange(10 * 10).reshape(10, 10)
        rebuilt = self.roundtrip(arr)
        self._assert_shared(rebuilt)
        np.testing.assert_array_equal(rebuilt, arr)

        arr2 = rebuilt[2:5, 3:7]
        self.assertGreater(_get_offset(arr2), 0)  # we expect a nonzero offset
        rebuilt2 = self.roundtrip(arr2)
        self._assert_shared(rebuilt2)
        np.testing.assert_array_equal(rebuilt2, arr2)


if __name__ == '__main__':
    unittest.main()
