import unittest
from expecttest import TestCase
import functools
import atmodata
import torch
from torchdata.datapipes.iter import IterableWrapper
from atmodata.datapipes.iter.tensor import ThSplitter

_assert_tensors_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)

class TestIterTensor(TestCase):
    def test_ThSplitter(self):
        inp = torch.arange(10*10).reshape(10,10)
        in_dp = IterableWrapper([inp])
        
        # Test: Basic
        out_dp = ThSplitter(in_dp, 5, dim=0)
        expected = [inp[:5], inp[5:]]
        _assert_tensors_equal(expected, list(out_dp))

        # Test: dim argument
        out_dp = ThSplitter(in_dp, 5, dim=1)
        expected = [inp[:, :5], inp[:, 5:]]
        _assert_tensors_equal(expected, list(out_dp))

        # Test: partial chunks
        out_dp = ThSplitter(in_dp, 4, dim=0)
        expected = [inp[:4], inp[4:8], inp[8:]]
        _assert_tensors_equal(expected, list(out_dp))

if __name__ == '__main__':
    unittest.main()