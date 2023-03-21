from torchdata.datapipes.iter import IterDataPipe

variable_names = {
    'z': 'geopotential',
    't': 'temperature',
    'r': 'relative_humidity',
}


class WeatherbenchPathBuilder(IterDataPipe):
    def __init__(self, dp, base_dir, variable, suffix='1.40625deg'):
        self.dp = dp
        self.base_dir = base_dir
        self.variable = variable
        self.suffix = suffix

    def __iter__(self):
        for year in self.dp:
            fname = variable_names[self.variable]
            path = f'{self.base_dir}/{fname}/{fname}_{year}'
            if self.suffix:
                yield f'{path}_{self.suffix}.nc'
            else:
                yield f'{path}.nc'
