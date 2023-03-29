import xarray  # usort:skip # noqa: F401
from . import patching

patching.patch_torchdata()  # TODO: remove this once torchdata is fixed

from . import builder, cli, datasets, iter, map, serialization, tasks, utils, xarray_utils

try:
    from . import version  # fmt: skip
    __version__ = version.__version__  # noqa: F401
except ImportError:
    pass

__all__ = [
    'builder',
    'cli',
    'datasets',
    'iter',
    'map',
    'patching',
    'serialization',
    'tasks',
    'utils',
    'xarray_utils',
]
assert __all__ == sorted(__all__)
