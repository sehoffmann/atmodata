import unittest

import atmodata
import xarray as xr
from atmodata.iter.xarray import XrSplitDim
from expecttest import TestCase
from fakedata import create_fake_dataset
from torchdata.datapipes.iter import IterableWrapper


class TestIterXarray(TestCase):
    def test_XrSplitDim(self):
        ds = create_fake_dataset('t')
        in_dp = IterableWrapper([ds])

        # Test: basic
        out_dp = XrSplitDim(in_dp, 'time', 4)
        output = list(out_dp)
        for i in range(4):
            N_elems = len(ds.time) // 4
            expected = ds.isel(time=slice(i * N_elems, (i + 1) * N_elems))
            xr.testing.assert_identical(expected, output[i])


if __name__ == '__main__':
    unittest.main()
