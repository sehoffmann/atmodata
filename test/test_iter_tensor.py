import unittest
from expecttest import TestCase
import torch
from torchdata.datapipes.iter import IterableWrapper
from atmodata.datapipes.iter.tensor import ThSplitter

class TestIterTensor(TestCase):
    def test_ThSplitter(self):
        inp = [torch.arange(10*20).reshape(10,20), torch.arange(10*20).reshape(10,20)]
        in_dp = IterableWrapper(inp)
        
        out_dp = ThSplitter(in_dp, 2, dim=0)
        expected = [inp[0][:5], inp[0][5:], inp[1][:5], inp[1][5:]]
        self.assertEqual(expected, list(out_dp))

        out_dp = ThSplitter(in_dp, 2, dim=1)
        expected = [inp[0][:, :10], inp[0][:, 10:], inp[1][:, :10], inp[1][:, 10:]]
        self.assertEqual(expected, list(out_dp))

if __name__ == '__main__':
    unittest.main()