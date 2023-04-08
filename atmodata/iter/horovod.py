import torch
from torchdata.datapipes import functional_datapipe

from torchdata.datapipes.iter import IterDataPipe

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    pass

if hvd is not None:

    @functional_datapipe('horovod_full_sync')
    class HorovodFullSync(IterDataPipe):
        """
        Place this at the end of the pipeline to prevent hanging when using Horovod.

        An allreduce operation is performed before yielding to determine if any workers are exhausted early.
        If so, the pipeline will stop early.
        """

        TRUE_TENSOR = torch.tensor(True, dtype=torch.float, device='cpu')
        FALSE_TENSOR = torch.tensor(False, dtype=torch.float, device='cpu')

        def __init__(self, dp):
            self.dp = dp

        def __iter__(self):
            it = iter(self.dp)
            exhausted = False
            while not exhausted:
                try:
                    x = next(it)
                except StopIteration:
                    exhausted = True

                tensor = HorovodFullSync.TRUE_TENSOR if exhausted else HorovodFullSync.FALSE_TENSOR
                result = hvd.allreduce(tensor, op=hvd.Sum).item()
                if result:
                    break  # some worker is exhausted early

                if not exhausted:
                    yield x
