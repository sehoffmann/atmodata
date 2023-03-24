import sys

import atmodata
import pytest
import xarray as xr
from atmodata.iter.xarray import XrSplitDim, XrUnrollIndices
from fakedata import create_fake_dataset
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper


class TestIterXarray:
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

    @staticmethod
    def _helper(tpl):
        return tpl[0]

    def test_XrUnrollIndices(self):
        ds = create_fake_dataset('t', time_max=300)
        in_dp = IterableWrapper([ds])

        # Test: basic
        out_dp = XrUnrollIndices(in_dp, 'time')
        out_indices = [idx for idx, ds in out_dp]
        assert out_indices == list(range(len(ds.time)))

        # Test: shuffle
        out_dp = XrUnrollIndices(in_dp, 'time', shuffle=True)
        out_indices = [idx for idx, ds in out_dp]
        assert set(out_indices) == set(range(len(ds.time)))
        assert out_indices != list(range(len(ds.time)))

        # Test: shuffle + seeding
        out_dp = XrUnrollIndices(in_dp, 'time', shuffle=True)
        rs = MultiProcessingReadingService(num_workers=0)
        dl = DataLoader2(out_dp, reading_service=rs)
        dl.seed(123)
        out_indices = [idx for idx, ds in dl]
        assert set(out_indices) == set(range(len(ds.time)))
        assert out_indices != list(range(len(ds.time)))

        dl.seed(123)
        out_indices2 = [idx for idx, ds in dl]
        assert out_indices == out_indices2

        # Test: different seeds are used for successive datasets
        in_dp = IterableWrapper([ds, ds])
        out_dp = XrUnrollIndices(in_dp, 'time', shuffle=True)
        dl = DataLoader2(out_dp, reading_service=rs)
        dl.seed(123)
        out_indices = [idx for idx, ds in dl]
        assert out_indices[: len(out_indices) // 2] == out_indices2
        assert out_indices[: len(out_indices) // 2] != out_indices[len(out_indices) // 2 :]

        # Test: worker shuffle indices identically
        out_dp = in_dp.sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
        out_dp = XrUnrollIndices(out_dp, 'time', shuffle=True)
        out_dp = out_dp.sharding_filter(
            SHARDING_PRIORITIES.MULTIPROCESSING
        )  # <-- this is important, otherwise every worker receives different seeds
        out_dp = out_dp.map(self._helper)  # otherwise we serialize ds for every index
        rs = MultiProcessingReadingService(num_workers=2)
        dl = DataLoader2(out_dp, reading_service=rs)
        dl.seed(123)
        new_out_indices = [idx for idx in dl]
        assert new_out_indices == out_indices


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
