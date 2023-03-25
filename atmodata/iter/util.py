import functools

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

import atmodata.serialization


@functional_datapipe('non_replicable')
class NonReplicableIterDataPipe(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        return iter(self.dp)

    def is_replicable(self):
        return False


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
    def __init__(self, dp, fn):
        self.dp = dp
        self.fn = fn

    def __iter__(self):
        for x in self.dp:
            if isinstance(x, tuple):
                yield tuple(self.fn(elem) for elem in x)
            if isinstance(x, list):
                yield [self.fn(elem) for elem in x]
            else:
                raise ValueError(f'NestedMapper only supports single-level tuples and lists for now, got {type(x)}')


@functional_datapipe('share_memory')
class MemorySharer(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for x in self.dp:
            yield atmodata.serialization.share_memory(x)
