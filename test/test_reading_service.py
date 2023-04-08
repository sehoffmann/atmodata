import sys

import pytest

from atmodata.reading_service import DistributedReadingService
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes.iter import IterableWrapper


class TestRS(DistributedReadingService):
    workers = {}
    global_seed = None

    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        TestRS.workers[rank] = self

    def _distribute_seed(self, seed):
        print(f'{self._rank}: {TestRS.global_seed}', file=sys.stderr)
        if self._rank == 0:
            TestRS.global_seed = seed
        return TestRS.global_seed


class TestReadingService:
    def test_distributed_rs(self):
        # Test: Basic sharding
        pipe1 = IterableWrapper(range(10)).sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
        pipe2 = IterableWrapper(range(10)).sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)

        dl1 = DataLoader2(pipe1, reading_service=TestRS(rank=0, world_size=2))
        dl2 = DataLoader2(pipe2, reading_service=TestRS(rank=1, world_size=2))

        assert list(dl1) == [0, 2, 4, 6, 8]
        assert list(dl2) == [1, 3, 5, 7, 9]

        # Test: Seed sharing
        pipe1 = IterableWrapper(range(40)).shuffle().sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
        pipe2 = IterableWrapper(range(40)).shuffle().sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)

        dl1 = DataLoader2(pipe1, reading_service=TestRS(rank=0, world_size=2))
        dl2 = DataLoader2(pipe2, reading_service=TestRS(rank=1, world_size=2))

        res1 = list(dl1)
        res2 = list(dl2)

        assert len(res1) == 20
        assert len(res2) == 20
        assert sorted(res1 + res2) == list(range(40))
        assert res1 + res2 != list(range(40))

        # Test: Uneven sharding
        pipe1 = IterableWrapper(range(11)).sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
        pipe2 = IterableWrapper(range(11)).sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)

        dl1 = DataLoader2(pipe1, reading_service=TestRS(rank=0, world_size=2))
        dl2 = DataLoader2(pipe2, reading_service=TestRS(rank=1, world_size=2))

        assert list(dl1) == [0, 2, 4, 6, 8, 10]
        assert list(dl2) == [1, 3, 5, 7, 9]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
