from pathlib import Path

import numpy as np
import xarray as xr


_coords = None


def _get_coords():
    global _coords
    if _coords is None:
        path = Path(__file__).parent / '_data' / 'onlycoords.nc'
        _coords = xr.open_dataset(path).load()
    return _coords


def create_fake_dataset(var_name, time_max=100, level_max=3, value=None):
    only_coords = _get_coords()
    time = only_coords.time[:time_max] if time_max is not None else only_coords.time
    level = only_coords.level[:level_max] if level_max is not None else only_coords.level
    shape = (len(time), len(level), len(only_coords.lat), len(only_coords.lon))

    if value is not None:
        data = np.full(shape, value)
    else:
        data = np.random.rand(*shape)

    da = xr.DataArray(
        data,
        dims=['time', 'level', 'lat', 'lon'],
        coords={
            'time': time,
            'level': level,
            'lat': only_coords.lat,
            'lon': only_coords.lon,
        },
    )
    return xr.Dataset({var_name: da})
