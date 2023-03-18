from . import datapipes

try:
    from . import version
    __version__ = version.__version__  # noqa: F401
except ImportError:
    pass

__all__ = ['datapipes']