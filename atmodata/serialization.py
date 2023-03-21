import pickle
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch

__all__ = [
    'rebuild_ndarray',
    'reduce_ndarray',
    'share_memory',
]
assert __all__ == sorted(__all__)


def rebuild_ndarray(tensor, metainfo):
    offset, shape, strides, typestr = metainfo
    buffer = tensor.numpy()
    return np.ndarray(buffer=buffer, offset=offset, shape=shape, strides=strides, dtype=typestr)


def reduce_ndarray(arr: np.ndarray):
    if arr.dtype.hasobject:  # fall back to default impl for python objects
        return arr.__reduce__()

    if arr.ndim == 0:  # we can't safely cast 0d arrays, nor is it necessary
        return arr.__reduce__()

    shape = arr.__array_interface__['shape']
    strides = arr.__array_interface__['strides']
    typestr = arr.__array_interface__['typestr']

    base = arr.base
    while type(base) is np.ndarray and base.base is not None:  # only support pure np.ndarray's for now
        base = base.base

    if isinstance(base, torch.Tensor):
        tensor = base
        offset = np.asarray(base).__array_interface__['data'][0] - arr.__array_interface__['data'][0]
    else:
        tensor = torch.as_tensor(arr.view(np.int8))
        offset = 0

    return (rebuild_ndarray, (tensor, (offset, shape, strides, typestr)))


def share_memory(obj):
    if isinstance(obj, torch.Tensor):
        return obj.share_memory()
    else:
        serialized = ForkingPickler.dumps(obj)
        return pickle.loads(serialized)


ForkingPickler.register(np.ndarray, reduce_ndarray)
