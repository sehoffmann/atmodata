'''
Copyright (c) 2023 Sebastian Hoffmann - All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
'''

import importlib


class PatchedFunction:
    def __init__(self, module_name, attr, func):
        self.module_name = module_name
        self.attr = attr
        self.func = func
        self.original = None
        self.patched = False

    def __call__(self, *args, **kwargs):
        if self.patched:
            return self.func(*args, **kwargs)
        else:
            return self.original(*args, **kwargs)

    def patch(self):
        self.patched = True
        module = importlib.import_module(self.module_name)
        self.original = getattr(module, self.attr)
        setattr(module, self.attr, self)

    def unpatch(self):
        self.patched = False
        module = importlib.import_module(self.module_name)
        setattr(module, self.attr, self.original)


def patched_apply_sharding(datapipe, num_of_instances: int, instance_id: int, sharding_group):
    r"""
    Apply dynamic sharding over the ``sharding_filter`` DataPipe that has a method ``apply_sharding``.
    RuntimeError will be raised when multiple ``sharding_filter`` are presented in the same branch.
    """
    import inspect

    from torch.utils.data.graph import traverse_dps
    from torch.utils.data.graph_settings import _is_sharding_datapipe

    graph = traverse_dps(datapipe)

    def _helper(graph, prev_applied=None):
        for _, (dp, sub_graph) in graph.items():
            applied = None
            if _is_sharding_datapipe(dp):
                # For BC, only provide sharding_group if accepted
                sig = inspect.signature(dp.apply_sharding)
                if len(sig.parameters) < 3:
                    dp.apply_sharding(num_of_instances, instance_id)
                else:
                    dp.apply_sharding(num_of_instances, instance_id, sharding_group=sharding_group)
                applied = dp
            if applied is None:
                applied = prev_applied
            _helper(sub_graph, applied)

    _helper(graph)

    return datapipe


def patch_torchdata():
    PatchedFunction('torch.utils.data.graph_settings', 'apply_sharding', patched_apply_sharding).patch()

    # adds as_transform() to IterDataPipe and MapDataPipe
    from torchdata.datapipes.iter import IterDataPipe
    from torchdata.datapipes.map import MapDataPipe

    from atmodata.utils import as_transform

    if not hasattr(IterDataPipe, 'as_transform'):
        IterDataPipe.as_transform = as_transform

    if not hasattr(MapDataPipe, 'as_transform'):
        IterDataPipe.as_transform = as_transform


def unpatch_torchdata():
    # fmt: off
    import torch.utils.data.graph_settings as m
    func = getattr(m, 'apply_sharding')
    if isinstance(func, PatchedFunction):
        func.unpatch()
