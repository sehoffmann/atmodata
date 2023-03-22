from torchdata.datapipes.iter import IterDataPipe


class ForecastingIterDataPipe(IterDataPipe):
    def __init__(self, dp, steps, rate, dim='time', crop_size=None, crops_per_sample=1):
        assert crops_per_sample >= 1
        assert rate >= 1
        assert steps >= 1

        pipe = dp.xr_extract_timeseries(steps, rate, dim=dim, shuffle=True, shard=True)
        if crop_size:
            if crops_per_sample > 1:
                pipe = pipe.repeat(crops_per_sample)
            pipe = pipe.xr_random_crop(crop_size, wraps={'lon': True})
            if crops_per_sample > 1:
                pipe = pipe.shuffle(buffer_size=crops_per_sample * 4)
        self.dp = pipe

    def __iter__(self):
        return iter(self.dp)


class ForecastingTask:
    def __init__(self, steps, rate, dim='time', crop_size=None, crops_per_sample=1):
        self.steps = steps
        self.rate = rate
        self.dim = dim
        self.crop_size = crop_size
        self.crops_per_sample = crops_per_sample

    def __call__(self, dp):
        return ForecastingIterDataPipe(
            dp, self.steps, self.rate, dim=self.dim, crop_size=self.crop_size, crops_per_sample=self.crops_per_sample
        )
