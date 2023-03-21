from . import datasets, iter, map, patching, tasks, utils

try:
    from . import version

    __version__ = version.__version__  # noqa: F401
except ImportError:
    pass

__all__ = ['datasets', 'iter', 'map', 'patching', 'tasks', 'utils']

# patch torchdata, c.f. pytorch/data#1082 and pytorch/data#1087
patching.patch_torchdata()
