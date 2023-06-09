'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import functools
import os

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

import atmodata.serialization


@functional_datapipe('assure_tuple')
class TupleAssurer(IterDataPipe):
    """
    Assures that the data pipe returns tuples.
    If the data pipe returns a single element, it is wrapped in a singleton tuple.
    """

    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for x in self.dp:
            if isinstance(x, tuple):
                yield x
            else:
                yield (x,)


@functional_datapipe('non_replicable')
class NonReplicableIterDataPipe(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        return iter(self.dp)

    def is_replicable(self):
        return False


@functional_datapipe('debug_print')
class DebugPrinter(IterDataPipe):
    def __init__(self, dp, prefix='', print_index=False, print_element=True, print_pid=True, flush=True):
        self.dp = dp
        self.prefix = prefix
        self.print_index = print_index
        self.print_element = print_element
        self.print_pid = print_pid
        self.flush = flush

    def __iter__(self):
        for i, x in enumerate(self.dp):
            string = self.prefix
            if self.print_pid:
                string += f'@{os.getpid()}'
            if self.print_index:
                string += f'#{i}'
            if self.print_element:
                string += f' = {x}'
            print(string, flush=self.flush)
            yield x


@functional_datapipe('round_robin_transform')
class RoundRobinTransformer(IterDataPipe):
    def __init__(self, dp, forks, transform_func):
        pipes = dp.round_robin_demux(forks)
        pipes = [transform_func(dp) for dp in pipes]
        self.dp = pipes[0].mux_longest(*pipes[1:])

    def __iter__(self):
        return iter(self.dp)


@functional_datapipe('round_robin_map')
class RoundRobinMapper(IterDataPipe):
    def __init__(self, dp, forks, fn):
        self.dp = dp.round_robin_transform(forks, functools.partial(RoundRobinMapper._transform, fn=fn))

    def __iter__(self):
        return iter(self.dp)

    @staticmethod
    def _transform(pipe, fn):
        return pipe.map(fn)


@functional_datapipe('nested_map')
class NestedMapper(IterDataPipe):
    def __init__(self, dp, fn, max_level=None):
        self.dp = dp
        self.fn = fn
        self.max_level = max_level

    def _nested_map(self, x, level=0):
        if self.max_level is not None and level > self.max_level:
            return self.fn(self.x)
        elif isinstance(x, tuple):
            return tuple(self._nested_map(elem, level + 1) for elem in x)
        elif isinstance(x, list):
            return [self._nested_map(elem, level + 1) for elem in x]
        elif isinstance(x, dict):
            return {k: self._nested_map(elem, level + 1) for k, elem in x.items()}
        else:
            return self.fn(x)

    def __iter__(self):
        for x in self.dp:
            yield self._nested_map(x)


@functional_datapipe('share_memory')
class MemorySharer(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for x in self.dp:
            yield atmodata.serialization.share_memory(x)
