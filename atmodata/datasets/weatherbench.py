import functools
from pathlib import Path

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, Zipper

from atmodata.utils import collate_coordinates
from atmodata.xarray_utils import unstack_coordinate


VARIABLE_NAMES = {
    'pv': 'potential_vorticity',
    'q': 'specific_humidity',
    'r': 'relative_humidity',
    't': 'temperature',
    't2m': '2m_temperature',  # SL
    'tau_300_700': 'tau_300_700',  # SL
    'tcc': 'total_cloud_cover',  # SL
    'tisr': 'toa_incident_solar_radiation',  # SL
    'tp': 'total_precipitation',  # SL
    'u': 'u_component_of_wind',
    'u10': '10m_u_component_of_wind',  # SL
    'v': 'v_component_of_wind',
    'v10': '10m_v_component_of_wind',  # SL
    'vo': 'vorticity',
    'z': 'geopotential',
}

SINGLE_LEVEL_VARIABLES = [
    'tau_300_700',
    'tisr',
    'tcc',
    'tp',
    'u10',
    'v10',
    't2m',
]

CONSTANTS = [
    'orography',
    'lsm',  # land sea maks
    'slt',  # soil type
    'lat2d',
    'lon2d',
]

COORDINATES = ['time', 'level', 'lat', 'lon']

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def is_constant(variable):
    """
    Returns True if the variable is indexed by (lat, lon) only. (e.g. orography)
    """
    return variable in CONSTANTS


def is_single_level(variable):
    """
    Returns True if the variable is indexed by (time, lat, lon) only.
    """
    return variable in SINGLE_LEVEL_VARIABLES


def get_constants_path(base_dir, suffix='1.40625deg'):
    """
    Returns the path of the file that contains the constant variables.
    """
    base_dir = Path(base_dir)
    if suffix is None:
        return base_dir / 'constants' / 'constants.nc'
    else:
        return base_dir / 'constants' / f'constants_{suffix}.nc'


def get_path(base_dir, variable, year=None, suffix='1.40625deg'):
    """
    Returns the path of the file that contains the given variable for the given year.
    """
    if is_constant(variable):
        return get_constants_path(base_dir, suffix)
    else:
        assert year is not None
        var_name = VARIABLE_NAMES[variable]
        path = Path(base_dir) / var_name
        if suffix:
            path /= f'{var_name}_{year}_{suffix}.nc'
        else:
            path /= f'{var_name}_{year}.nc'
        return path


def aggregate_variables(variables):
    """
    Splits variables into multi-level, single-level, and constant variables
    and aggregates multi-level variables (with format {var}{level}) into a dictionary of {var: levels}.
    """
    single_level = {v for v in variables if is_single_level(v)}
    constants = {v for v in variables if is_constant(v)}
    multi_level = {}
    remaining = set(variables) - single_level - constants
    for var in remaining:
        if var in VARIABLE_NAMES and not is_single_level(var):
            multi_level[var] = set(LEVELS)

        found = False
        for level in LEVELS:
            level_str = str(level)
            if var.endswith(level_str):
                raw = var[: -len(level_str)]
                multi_level.setdefault(raw, set()).add(level)
                found = True
                break

        if not found:
            raise ValueError(f'Unknown variable: {var}')

    return multi_level, single_level, constants


class WeatherbenchPathBuilder(IterDataPipe):
    def __init__(self, dp, base_dir, variable, suffix='1.40625deg'):
        self.dp = dp
        self.base_dir = base_dir
        self.variable = variable
        self.suffix = suffix

    def __iter__(self):
        for year in self.dp:
            yield get_path(self.base_dir, self.variable, year, self.suffix)


class WeatherBench(IterDataPipe):
    def __init__(self, base_dir, variables, years, shards_per_year=1, shuffle=False, suffix='1.40625deg'):
        self.base_dir = base_dir
        self.variables = set(variables)
        self.multi_level, self.single_level, self.constants = aggregate_variables(self.variables)
        self.years = list(iter(years))
        self.shards_per_year = shards_per_year
        self.shuffle = shuffle
        self.suffix = suffix

        dyn_pipe = self._build_dyn_pipe()
        constants_pipe = self._build_constants_pipe()
        self.dp = Zipper(dyn_pipe, constants_pipe).xr_merge()

    def _build_constants_pipe(self):
        pipe = IterableWrapper([get_constants_path(self.base_dir, self.suffix)])
        pipe = pipe.xr_open().xr_select_variables(self.constants).xr_load()
        pipe = pipe.share_memory()
        return pipe.cycle()

    def _build_dyn_pipe(self):
        dyn_vars = set(self.multi_level) | self.single_level
        forked_years = IterableWrapper(self.years).fork(len(dyn_vars))

        shards_single_var = []
        for pipe, var in zip(forked_years, dyn_vars):
            pipe = WeatherbenchPathBuilder(pipe, self.base_dir, var, self.suffix)
            pipe = pipe.xr_open().xr_select_variables(var)

            if not is_single_level(var):
                indices = collate_coordinates(self.multi_level[var], LEVELS, no_scalar=True)
                pipe = pipe.xr_isel(level=indices)

            pipe = pipe.xr_split_dim('time', self.shards_per_year)
            shards_single_var.append(pipe)

        dyn_pipe = Zipper(*shards_single_var)
        if self.shuffle:
            dyn_pipe = dyn_pipe.shuffle(buffer_size=len(self.years) * self.shards_per_year)
        dyn_pipe = dyn_pipe.sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
        dyn_pipe = dyn_pipe.xr_prefetch(buffer_size=0)
        dyn_pipe = dyn_pipe.nested_map(functools.partial(unstack_coordinate, dim='level')).flatten()
        dyn_pipe = dyn_pipe.xr_merge()
        return dyn_pipe

    def __iter__(self):
        return iter(self.dp)
