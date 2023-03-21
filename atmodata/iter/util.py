from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe('non_replicable')
class NonReplicableIterDataPipe(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        return iter(self.dp)

    def is_replicable(self):
        return False
