import functools
import sys

import atmodata
import pytest
import torch
from atmodata.iter.tensor import ThBatchInterleaver, ThChunker, ThSplitter
from expecttest import TestCase
from torchdata.datapipes.iter import IterableWrapper

_assert_tensors_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


class TestIterTensor(TestCase):
    def test_ThSplitter(self):
        inp = torch.arange(10 * 10).reshape(10, 10)
        in_dp = IterableWrapper([inp])

        # Test: basic
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

    def test_ThChunker(self):
        inp = torch.arange(10 * 10).reshape(10, 10)
        in_dp = IterableWrapper([inp])

        # Test: basic
        out_dp = ThChunker(in_dp, 2, dim=0)
        expected = [inp[:5], inp[5:10]]
        _assert_tensors_equal(expected, list(out_dp))

        # Test: dim argument
        out_dp = ThChunker(in_dp, 2, dim=1)
        expected = [inp[:, :5], inp[:, 5:10]]
        _assert_tensors_equal(expected, list(out_dp))

    def test_ThBatchInterleaver(self):
        inp = torch.arange(11 * 5).reshape(11, 5)
        in_dp = IterableWrapper([inp, inp])

        # Test: basic
        out_dp = ThBatchInterleaver(in_dp, 2)
        expected = [torch.cat([inp[:6], inp[:6]]), torch.cat([inp[6:], inp[6:]])]
        _assert_tensors_equal(expected, list(out_dp))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
