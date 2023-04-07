'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.datapipes.iter import IterDataPipe


class ForecastingIterDataPipe(IterDataPipe):
    def __init__(self, dp, steps, rate, dim='time', crop_size=None, crops_per_sample=1):
        assert crops_per_sample >= 1
        assert rate >= 1
        assert steps >= 1

        pipe = dp.xr_unroll_indices(dim=dim, shuffle=True)
        pipe = pipe.sharding_filter(SHARDING_PRIORITIES.MULTIPROCESSING)
        pipe = pipe.xr_extract_timeseries(steps, rate, dim=dim)
        if crop_size:
            if crops_per_sample > 1:
                pipe = pipe.repeat(crops_per_sample)
            pipe = pipe.xr_random_crop(crop_size, wraps={'lon': True})
            if crops_per_sample > 1:
                pipe = pipe.shuffle(buffer_size=crops_per_sample * 4)
        self.dp = pipe

    def __iter__(self):
        return iter(self.dp)


ForecastingTask = ForecastingIterDataPipe.as_transform
