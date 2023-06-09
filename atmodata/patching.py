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
    """
    This must be patched to remove the check that prevents mutiple sharding_filters to be present in the datapipe.
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


def patched_process_init_fn(
    datapipe,
    worker_info,
    custom_init_fn=None,
    worker_prefetch_cnt=0,
    dispatching_req_queue=None,
    dispatching_res_queue=None,
):
    """
    This must be patched to enable sharding within the worker processes.
    Otherwise, torchdata only shards using sharding_round_robin_dispatch().
    C.f. https://github.com/pytorch/data/issues/1174
    """
    from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
    from torchdata.dataloader2 import communication
    from torchdata.dataloader2.graph import find_dps, replace_dp, traverse_dps
    from torchdata.dataloader2.utils.dispatch import _DummyIterDataPipe
    from torchdata.datapipes.iter import IterDataPipe
    from torchdata.datapipes.map import MapDataPipe

    # Find if there is non-replicable DataPipe
    graph = traverse_dps(datapipe)
    non_replicable_dp = find_dps(graph, _DummyIterDataPipe)  # type: ignore

    # There are two cases for DataPipe graph in terms of mp sharding:
    # 1) All DataPipes are replicable, apply mp sharding to the whole graph
    if len(non_replicable_dp) == 0:
        patched_apply_sharding(
            datapipe, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
        )
        assert dispatching_req_queue is None and dispatching_res_queue is None
    # 2) There is non-replicable DataPipe. Since we have replaced the lowest common
    #    ancestor by a `_DummyIterDataPipe`, we would only apply mp sharding
    #    to replicable branches that don't have `_DummyIterDataPipe`.
    else:
        assert len(non_replicable_dp) == 1
        assert not (dispatching_req_queue is None and dispatching_res_queue is None)
        dispatching_req_queue.cancel_join_thread()  # type: ignore[union-attr]
        patched_apply_sharding(
            datapipe, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
        )

        queue_wrapper = communication.iter.QueueWrapper(
            communication.protocol.IterDataPipeQueueProtocolClient(dispatching_req_queue, dispatching_res_queue)
        )
        dispatch_process_dp = communication.iter._IterateQueueDataPipes([queue_wrapper])
        graph = replace_dp(graph, non_replicable_dp[0], dispatch_process_dp)
        datapipe = list(graph.values())[0][0]

    if custom_init_fn is not None:
        datapipe = custom_init_fn(datapipe, worker_info)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    if worker_prefetch_cnt > 0:
        datapipe = datapipe.prefetch(worker_prefetch_cnt)

    return datapipe


def patch_torchdata():
    PatchedFunction('torchdata.dataloader2.reading_service', 'process_init_fn', patched_process_init_fn).patch()
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
    import torchdata.dataloader2.reading_service as m
    func = getattr(m, 'process_init_fn')
    if isinstance(func, PatchedFunction):
        func.unpatch()

    import torch.utils.data.graph_settings as m
    func = getattr(m, 'apply_sharding')
    if isinstance(func, PatchedFunction):
        func.unpatch()
