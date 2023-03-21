import xarray  # usort:skip # noqa: F401
from . import patching

patching.patch_torchdata()

from . import datasets, iter, map, tasks, utils

try:
    from . import version  # fmt: skip
    __version__ = version.__version__  # noqa: F401
except ImportError:
    pass

__all__ = ['datasets', 'iter', 'map', 'patching', 'tasks', 'utils']
