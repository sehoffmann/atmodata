'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

from .forecasting import (
    # ------------------------------------------------------ USORT NO SPLIT
    ForecastingIterDataPipe,
    ForecastingTask,
)

__all__ = [
    'ForecastingIterDataPipe',
    'ForecastingTask',
]
assert __all__ == sorted(__all__)
