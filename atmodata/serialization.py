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


def _get_owning_base(arr: np.ndarray):
    assert isinstance(arr, np.ndarray)
    base = arr
    while hasattr(base, 'base') and base.base is not None:
        base = base.base
    return base


def _have_same_memory(tensor1: torch.Tensor, tensor2: torch.Tensor):
    if tensor1.untyped_storage().data_ptr() != tensor2.untyped_storage().data_ptr():
        return False

    if tensor1.storage_offset() * tensor1.element_size() != tensor2.storage_offset() * tensor2.element_size():
        return False

    if tensor1.size() != tensor2.size():
        return False

    if tensor1.stride() != tensor2.stride():
        return False

    return True


def _to_bytes(arr: np.ndarray):
    """
    Gurantees that we can always cast down to bytes, even if the last dimension
    is not contiguous, by adding an additional dimension.
    """
    if arr.dtype.hasobject:
        raise ValueError(f'Cannot serialize np.ndarray with dtype {arr.dtype}')
    if arr.ndim == 0:
        raise ValueError('Cannot serialize 0d np.ndarray')
    if arr.strides[-1] != arr.dtype.itemsize:
        arr = arr.reshape(arr.shape + (1,))
    return arr.view(np.byte)


def _from_bytes(arr: np.ndarray, dtype: np.dtype):
    """
    Inverse of _to_bytes
    """
    if arr.ndim == 0:
        raise ValueError('Cannot deserialize 0d np.ndarray')
    if arr.strides[-1] != 1:
        arr = arr.reshape(arr.shape[:-1])
    return arr.view(dtype)


def rebuild_ndarray(tensor, metainfo):
    shape, strides, typestr = metainfo
    buffer = np.asarray(tensor)
    dtype = np.dtype(typestr)
    try:
        arr = _from_bytes(buffer, dtype)
        arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        assert _have_same_memory(_get_owning_base(arr), tensor)
        return arr
    except Exception as e:
        msg = f'Failed to deserialize np.ndarray from shared-mem torch.Tensor: {e}'
        msg += f'\n  * shape: {shape}'
        msg += f'\n  * strides: {strides}'
        msg += f'\n  * typestr: {typestr}'
        msg += f'\n  * itemsize: {dtype.itemsize}'
        msg += f'\n  * tensor storage: addr {tensor.untyped_storage().data_ptr()} - size: {tensor.untyped_storage().nbytes()}'
        msg += f'\n  * tensor __array_interface__: {np.asarray(tensor).__array_interface__}'
        raise ValueError(msg) from e


def reduce_ndarray(arr: np.ndarray):
    if arr.dtype.hasobject:  # fall back to default impl for python objects
        return arr.__reduce__()

    if arr.ndim == 0:  # we can't safely cast 0d arrays, nor is it necessary
        return arr.__reduce__()

    shape = arr.__array_interface__['shape']
    strides = arr.__array_interface__['strides']
    typestr = arr.__array_interface__['typestr']

    base = _get_owning_base(arr)
    if isinstance(base, torch.Tensor):
        tensor = base
        assert arr.__array_interface__['data'][0] - np.asarray(base).__array_interface__['data'][0] == 0
    else:
        tensor = torch.as_tensor(_to_bytes(arr))

    return (rebuild_ndarray, (tensor, (shape, strides, typestr)))


def share_memory(obj):
    if isinstance(obj, torch.Tensor):
        return obj.share_memory()
    else:
        serialized = ForkingPickler.dumps(obj)
        return pickle.loads(serialized)


ForkingPickler.register(np.ndarray, reduce_ndarray)
