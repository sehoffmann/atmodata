from . import iter, map, patching

try:
    from . import version

    __version__ = version.__version__  # noqa: F401
except ImportError:
    pass

# patch torchdata, c.f. pytorch/data#1082 and pytorch/data#1087
patching.patch_torchdata()

__all__ = ['iter', 'map', 'patching']
