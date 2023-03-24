from torchdata.datapipes.iter import IterDataPipe

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


def get_path(base_dir, variable, year=None, suffix=None):
    """
    Returns the path of the file that contains the given variable for the given year.
    """
    if is_constant(variable):
        return f'{base_dir}/constants/constants_{suffix}.nc'
    else:
        assert year is not None
        var_name = VARIABLE_NAMES[variable]
        path = f'{base_dir}/{var_name}/{var_name}_{year}'
        if suffix:
            path += f'_{suffix}'
        return path + '.nc'


class WeatherbenchPathBuilder(IterDataPipe):
    def __init__(self, dp, base_dir, variable, suffix='1.40625deg'):
        self.dp = dp
        self.base_dir = base_dir
        self.variable = variable
        self.suffix = suffix

    def __iter__(self):
        for year in self.dp:
            yield get_path(self.base_dir, self.variable, year, self.suffix)
