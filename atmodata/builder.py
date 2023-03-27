from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService


class AtmodataPipeBuilder:
    def __init__(
        self,
        dataset,
        task,
        batch_size,
        num_parallel_shards=1,
        dataloading_prefetch_cnt=1,
        worker_prefetch_cnt=10,
        main_prefetch_cnt=10,
        device_prefetch_cnt=1,
    ):
        self.dataset = dataset
        self.task = task
        self.batch_size = batch_size
        self.num_parallel_shards = num_parallel_shards

        self.dataloading_prefetch_cnt = dataloading_prefetch_cnt
        self.worker_prefetch_cnt = worker_prefetch_cnt
        self.main_prefetch_cnt = main_prefetch_cnt
        self.device_prefetch_cnt = device_prefetch_cnt

        self.pre_share_memory(main_to_worker=True, worker_to_main=True)
        self.interleave_batches(False)
        self.transfer_to_device(None)
        self.custom_collate(None)
        self.cache_data(False)
        self.multiprocess(0)

    def pre_share_memory(self, main_to_worker=True, worker_to_main=True):
        self.share_main_to_worker = main_to_worker
        self.share_worker_to_main = worker_to_main
        return self

    def interleave_batches(self, value=True):
        self.interleaves_batches = value
        return self

    def transfer_to_device(self, device):
        self.device = device
        return self

    def custom_collate(self, collate_tranform_fn):
        self.collate_transform_fn = collate_tranform_fn
        return self

    def cache_data(self, value=True):
        self.caches_data = value
        return self

    def multiprocess(self, n_workers):
        assert n_workers >= 0
        self.n_workers = n_workers
        return self

    def build_pipe(self):
        # 1. Main process: Dataloading
        pipe = self.dataset

        if self.share_main_to_worker and self.n_workers > 0:
            pipe = pipe.share_memory()

        if self.dataloading_prefetch_cnt:
            pipe = pipe.prefetch(self.dataloading_prefetch_cnt)

        if self.caches_data:
            pipe = pipe.in_memory_cache()

        # 2. Worker process: Data processing
        if self.n_workers >= 1:
            if self.n_workers > 1:
                pipe = pipe.repeat(self.n_workers)
            pipe = pipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)

        if self.num_parallel_shards > 1:
            pipe = pipe.round_robin_transform(self.num_parallel_shards, self.task)
        else:
            pipe = self.task(pipe)

        pipe = pipe.xr_to_numpy()

        if self.batch_size:
            pipe = pipe.batch(self.batch_size)

        if self.collate_transform_fn is not None:
            pipe = self.collate_transform_fn(pipe)
        else:
            pipe = pipe.collate()

        if self.share_worker_to_main and self.n_workers > 0:
            pipe = pipe.share_memory()

        if self.worker_prefetch_cnt and self.n_workers > 0:
            pipe = pipe.prefetch(self.worker_prefetch_cnt)

        # 3. Main process: Aggregation
        need_device_prefetch = False
        pipe = pipe.non_replicable()

        if self.main_prefetch_cnt:
            pipe = pipe.prefetch(self.main_prefetch_cnt)

        if self.device is not None:
            pipe = pipe.th_to_device(self.device, non_blocking=True)
            need_device_prefetch = True

        if self.interleaves_batches and self.n_workers > 1:
            pipe = pipe.th_interleave_batches(self.n_workers)
            need_device_prefetch = True

        if self.device_prefetch_cnt and need_device_prefetch:
            pipe = pipe.prefetch(self.device_prefetch_cnt)

        return pipe

    def build_dataloader(self, horovod=False):
        if self.n_workers > 0:
            mprs = MultiProcessingReadingService(
                self.n_workers,
                worker_prefetch_cnt=0,
                main_prefetch_cnt=0,
            )
            reading_service = mprs
        else:
            reading_service = None

        dataloader = DataLoader2(
            self.build_pipe(),
            reading_service=reading_service,
        )
        return dataloader
