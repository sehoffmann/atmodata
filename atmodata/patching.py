class PatchedFunction:
    def __init__(self, module, attr, func):
        self.module = module
        self.attr = attr
        self.func = func
        self.original = getattr(module, attr)
        self.patched = False

    def __call__(self, *args, **kwargs):
        if self.patched:
            return self.func(*args, **kwargs)
        else:
            return self.original(*args, **kwargs)

    def patch(self):
        self.patched = True
        setattr(self.module, self.attr, self)

    def unpatch(self):
        self.patched = False
        setattr(self.module, self.attr, self.original)


def patched_apply_sharding(datapipe, num_of_instances: int, instance_id: int, sharding_group):
    from torch.utils.data.datapipes.iter.sharding import _ShardingIterDataPipe
    from torch.utils.data.graph import traverse_dps

    graph = traverse_dps(datapipe)

    def _helper(graph, prev_applied=None):
        for _, (dp, sub_graph) in graph.items():
            applied = None
            if isinstance(dp, _ShardingIterDataPipe):
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
    worker_prefetch_cnt: int = 0,
    dispatching_req_queue=None,
    dispatching_res_queue=None,
):
    from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
    from torchdata.dataloader2 import communication
    from torchdata.dataloader2.graph import find_dps, replace_dp, traverse_dps
    from torchdata.dataloader2.utils.dispatch import _DummyIterDataPipe
    from torchdata.datapipes.iter import IterDataPipe
    from torchdata.datapipes.map import MapDataPipe

    # Find if there is non-replicable DataPipe
    graph = traverse_dps(datapipe)
    non_replicable_dp = find_dps(graph, _DummyIterDataPipe)  # type: ignore

    patched_apply_sharding(
        datapipe, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
    )

    # 2) There is non-replicable DataPipe. Since we have replaced the lowest common
    #    ancestor by a `_DummyIterDataPipe`, we would only apply mp sharding
    #    to replicable branches that don't have `_DummyIterDataPipe`.
    if len(non_replicable_dp) > 0:
        assert len(non_replicable_dp) == 1
        assert not (dispatching_req_queue is None and dispatching_res_queue is None)
        dispatching_req_queue.cancel_join_thread()  # type: ignore[union-attr]

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
    # fmt: off
    import torchdata.dataloader2.reading_service as m
    PatchedFunction(m, 'process_init_fn', patched_process_init_fn).patch()

    import torch.utils.data.graph_settings as m
    PatchedFunction(m, 'apply_sharding', patched_apply_sharding).patch()


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
