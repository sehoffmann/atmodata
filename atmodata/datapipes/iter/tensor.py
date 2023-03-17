import torch
from torch.utils.data import functional_datapipe, IterDataPipe

@functional_datapipe("th_split")
class ThSplitter(IterDataPipe):
    
    def __init__(self, dp, split_size, dim=0):
        self.dp = dp
        self.split_size = split_size
        self.dim=dim
        
    def __iter__(self):
        for x in self.dp:
            for chunk in torch.split(x, self.split_size, dim=self.dim):
                yield chunk


@functional_datapipe("th_concat")
class ThConcatter(IterDataPipe):
    
    def __init__(self, dp, dim=0):
        self.dp = dp
        self.dim=dim
        
    def __iter__(self):
        for tensors in self.dp:
            yield torch.cat(tensors, dim=self.dim)


@functional_datapipe("th_interleave_batches")
class ThBatchInterleaver(IterDataPipe):
    
    def __init__(self, dp, n_interleaves, dim=0):
        self.dp = dp
        self.n_interleaves = n_interleaves
        self.dim=dim
        
    def __iter__(self):
        it = iter(self.dp)
        for batches in zip(*([it]*self.n_interleaves)):
            chunks = [torch.chunk(batch, self.n_interleaves, dim=self.dim) for batch in batches]
            for i in range(self.n_interleaves):
                yield torch.cat([chunk[i] for chunk in chunks], dim=self.dim)


@functional_datapipe("th_to_device")
class ThToDevice(IterDataPipe):
    
    def __init__(self, dp, device, non_blocking=False):
        self.dp = dp
        self.device = device
        self.non_blocking=non_blocking
        
    def __iter__(self):
        for x in self.dp:
            yield x.to(self.device, non_blocking=self.non_blocking)