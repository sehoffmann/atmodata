import torch.utils.data.graph_settings
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import ReadingServiceInterface
from torchdata.dataloader2.graph import set_graph_random_seed

from atmodata.iter import HorovodFullSync


class DistributedReadingService(ReadingServiceInterface):
    """
    A base class for distributed reading services that shard data based on rank and world size.

    Can be used as a standalone class, in which case the user is responsible for setting the rank and world size manually.
    Otherwise, subclasses should call ``set_rank`` and ``set_world_size`` in their ``initialize`` methods and then call the super method.
    They should also override ``_distribute_seed`` to distribute the random seed from rank 0 across the distributed workers.
    """

    def __init__(self, rank=None, world_size=None):
        self._rank = rank
        self._world_size = world_size
        self._datapipe = None

    def set_rank(self, rank):
        self._rank = rank

    def set_world_size(self, world_size):
        self._world_size = world_size

    def initialize(self, datapipe):
        if self._rank is not None and self._world_size is not None:
            torch.utils.data.graph_settings.apply_sharding(
                datapipe, self._world_size, self._rank, SHARDING_PRIORITIES.DISTRIBUTED
            )
        self._datapipe = datapipe
        return datapipe

    def _distribute_seed(self, seed):
        """Distributes the random seed from rank 0 across the distributed workers."""
        return seed

    def initialize_iteration(self, seed_generator, iter_reset_fn=None):
        assert self._datapipe is not None

        shared_seed = seed_generator.generate_shared_seed()
        if self._rank is not None and self._world_size is not None:
            shared_seed = self._distribute_seed(shared_seed)

        seed_generator.seed(shared_seed)
        seed_generator.spawn(self._rank, inplace=True)
        set_graph_random_seed(self._datapipe, seed_generator)
        return None


try:
    import horovod.torch as hvd

    HAS_HOROVOD = True

    class HorovodReadingService(DistributedReadingService):
        """
        A reading service that shards data based on rank and world size using Horovod.

        This reading service will add a ``horovod_full_sync`` datapipe to the end of the pipeline.
        """

        def __init__(self):
            super().__init__()

        def initialize(self, datapipe):
            if not hvd.is_initialized():
                raise RuntimeError("Horovod is not initialized")
            self.set_rank(hvd.rank())
            self.set_world_size(hvd.size())

            datapipe = super().initialize(datapipe)
            if not isinstance(datapipe, HorovodFullSync):
                datapipe = HorovodFullSync(datapipe)
            self._datapipe = datapipe
            return datapipe

        def _distribute_seed(self, seed):
            return hvd.broadcast_object(seed)

except ImportError:
    HAS_HOROVOD = False
