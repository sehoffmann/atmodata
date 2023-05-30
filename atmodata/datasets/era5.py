'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import functools
from pathlib import Path

import numpy as np
import xarray as xr
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, Zipper

from atmodata.datasets.common import Coord


MULTI_LEVEL_VARIABLES = {
    # Multi-level variables
    'd': 'divergence',
    'z': 'geopotential',
    'pv': 'potential_vorticity',
    'q': 'specific_humidity',
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'w': 'vertical_velocity',
    'vo': 'vorticity',
}

SINGLE_LEVEL_VARIABLES = {
    # Single-level variables
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    't2m': '2m_temperature',
    'cvh': 'high_vegetation_cover',  # Monthly
    'lai_hv': 'leaf_area_index_high_vegetation',  # Monthly
    'lai_lv': 'leaf_area_index_low_vegetation',  # Monthly
    'cvl': 'low_vegetation_cover',  # Monthly
    'mror': 'mean_runoff_rate',  # Monthly
    'mslhf': 'mean_surface_latent_heat_flux',  # Monthly
    'mtnlwrfcs': 'mean_top_net_long_wave_radiation_flux_clear_sky',  # Monthly
    'mtpr': 'mean_total_precipitation_rate',  # Monthly
    # 'z': 'orography',  # Monthly
    'sst': 'sea_surface_temperature',
    'sp': 'surface_pressure',
    'sshf': 'surface_sensible_heat_flux',
    'strd': 'surface_thermal_radiation_downwards',
    'tsr': 'top_net_solar_radiation',  # Monthly
    'ttr': 'top_net_thermal_radiation',
    'tcrw': 'total_column_rain_water',
    'tp': 'total_precipitation',
}

VARIABLE_NAMES = {**MULTI_LEVEL_VARIABLES, **SINGLE_LEVEL_VARIABLES}

MONTHLY = [
    'cvh',
    'lai_hv',
    'lai_lv',
    'cvl',
    'mror',
    'mslhf',
    'mtnlwrfcs',
    'mtpr',
    # 'z',
    'tsr',
]

COORDINATES = {
    'time': Coord.TIME,
    'latitude': Coord.LAT,
    'longitude': Coord.LON,
}

LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]
assert LEVELS == sorted(LEVELS)


def is_single_level(variable):
    return variable in SINGLE_LEVEL_VARIABLES


def is_multi_level(variable):
    return variable in MULTI_LEVEL_VARIABLES


def get_path(base_dir, variable, year, level=None):
    """
    Returns the path of the file that contains the given variable for the given year.
    """
    var_name = VARIABLE_NAMES[variable]
    if is_multi_level(variable):
        if level is None:
            raise ValueError(f'Level must be specified for multi-level variable: {variable}')
        return Path(base_dir) / 'multi_pressure_level' / var_name / str(level) / f'{var_name}_{year}_{level}.nc'
    elif is_single_level(variable):
        if level is not None:
            raise ValueError(f'Level must not be specified for single-level variable: {variable}')
        return Path(base_dir) / 'single_pressure_level' / var_name / f'{var_name}_{year}.nc'


def aggregate_variables(variables):
    """
    Splits variables into multi-level, single-level, and constant variables
    and aggregates multi-level variables (with format {var}{level}) into a dictionary of {var: levels}.
    """
    single_level = {v for v in variables if is_single_level(v)}
    multi_level = {}
    remaining = set(variables) - single_level
    for var in remaining:
        if var in VARIABLE_NAMES and not is_single_level(var):
            multi_level[var] = set(LEVELS)

        found = False
        for level in reversed(LEVELS):  # reversed to match the longest level first
            level_str = str(level)
            if var.endswith(level_str):
                raw = var[: -len(level_str)]
                multi_level.setdefault(raw, set()).add(level)
                found = True
                break

        if not found:
            raise ValueError(f'Unknown variable: {var}')

    return multi_level, single_level


def create_synthetic(timesteps, variable='t2m'):
    data = np.random.randn(timesteps, 720, 1440).astype('float32')
    time = np.datetime64('1979-01-01 00:00') + np.arange(timesteps, dtype='timedelta64[h]')
    da = xr.DataArray(
        data,
        dims=['time', 'latitude', 'longitude'],
        name=variable,
        coords={
            'time': time.astype('datetime64[ns]'),
            'latitude': np.linspace(90, -90, 720, endpoint=False, dtype='float32'),
            'longitude': np.linspace(0, 360, 1440, endpoint=False, dtype='float32'),
        },
    )
    return da


class ERA5(IterDataPipe):
    def __init__(
        self,
        base_dir,
        variables,
        years,
        shards_per_year=1,
        shuffle=False,
        standardize_coordinates=True,
        synthetic=False,
    ):
        self.base_dir = base_dir
        self.variables = set(variables)
        self.multi_level, self.single_level = aggregate_variables(self.variables)
        self.years = list(iter(years))
        self.shards_per_year = shards_per_year
        self.shuffle = shuffle
        self.standardize_coordinates = standardize_coordinates
        self.synthetic = synthetic

        self.dp = self._build_pipe()

    def _open_dataset(self, pipe, var, level=None):
        if self.synthetic:
            synthetic_ds = create_synthetic(366 * 24 // self.shards_per_year, f'{var}{level}' if level else var)
            return IterableWrapper([synthetic_ds]).share_memory().repeat(self.shards_per_year)
        else:
            pipe = pipe.xr_open()
            pipe = pipe.xr_select_variables(var)
            if level:
                pipe = pipe.xr_rename(f'{var}{level}')
            return pipe.xr_split_dim('time', self.shards_per_year)

    def _build_pipe(self):
        n_forks = len(self.single_level) + sum(len(levels) for levels in self.multi_level.values())
        years_pipes = IterableWrapper(self.years).fork(n_forks)
        if isinstance(years_pipes, IterableWrapper):
            years_pipes = [years_pipes]  # c.f. https://github.com/pytorch/data/issues/1164
        idx = 0

        single_level_pipes = []
        for var in self.single_level:
            pipe = years_pipes[idx].map(functools.partial(get_path, self.base_dir, var))
            pipe = self._open_dataset(pipe, var)
            single_level_pipes.append(pipe)
            idx += 1

        multi_level_pipes = []
        for var in self.multi_level:
            levels = self.multi_level[var]
            for level in levels:
                pipe = years_pipes[idx].map(functools.partial(get_path, self.base_dir, var, level=level))
                pipe = self._open_dataset(pipe, var, level)
                multi_level_pipes.append(pipe)
                idx += 1

        pipe = Zipper(*(single_level_pipes + multi_level_pipes))

        if self.shuffle:
            pipe = pipe.shuffle(buffer_size=len(self.years) * self.shards_per_year)

        pipe = pipe.sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
        pipe = pipe.xr_parallel_load()
        pipe = pipe.xr_merge()

        if self.standardize_coordinates:
            pipe = pipe.xr_rename_dims(COORDINATES)

        return pipe

    def __iter__(self):
        return iter(self.dp)

    def __len__(self):
        return len(self.years) * self.shards_per_year
