'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import xarray  # usort:skip # noqa: F401
from . import patching

patching.patch_torchdata()  # TODO: remove this once torchdata is fixed

from . import builder, cli, datasets, iter, map, reading_service, serialization, tasks, utils, xarray_utils

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
    'reading_service',
    'serialization',
    'tasks',
    'utils',
    'xarray_utils',
]
assert __all__ == sorted(__all__)
