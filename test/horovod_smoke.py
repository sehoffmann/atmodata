import horovod.torch as hvd
import torch
from atmodata.reading_service import HorovodReadingService
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes.iter import IterableWrapper


def main():
    hvd.init()
    print(f'rank: {hvd.rank()}, size: {hvd.size()}', flush=True)

    N = 11
    pipe = IterableWrapper(range(N * 10))
    pipe = pipe.batch(10).sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
    dl = DataLoader2(pipe, reading_service=HorovodReadingService())

    hvd.barrier()

    # Test: Basic iteration and sharding
    for batch in dl:
        print(f'rank: {hvd.rank()}, batch: {batch}', flush=True)

    hvd.barrier()
    if hvd.rank() == 0:
        print('done')
        print(flush=True)
    hvd.barrier()

    # Test: End-to-end training, tests for hanging
    A = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    optim = torch.optim.SGD([A], lr=0.1)
    optim = hvd.DistributedOptimizer(optim)

    for i, batch in enumerate(dl):
        print(f'rank: {hvd.rank()}, it: {i},  batch: {batch}', flush=True)
        loss = A.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()


if __name__ == '__main__':
    main()
