import xarray as xr
from torch.utils.data import functional_datapipe, IterDataPipe

@functional_datapipe("xr_open")
class XrOpener(IterDataPipe):
    
    def __init__(self, dp, **kwargs):
        self.dp = dp
        self.kwargs = kwargs
        
    def __iter__(self):
        for path in self.dp:
            yield xr.open_dataset(path, **self.kwargs)


@functional_datapipe("xr_load")
class XrLoader(IterDataPipe):
    
    def __init__(self, dp):
        self.dp = dp
        
        
    def __iter__(self):
        for ds in self.dp:
            yield ds.load()


@functional_datapipe("xr_get_variables")
class XrVariableGetter(IterDataPipe):
    
    def __init__(self, dp, variables):
        self.dp = dp
        self.variables = variables
        
        
    def __iter__(self):
        for ds in self.dp:
            yield ds[self.variables]

@functional_datapipe("xr_sel")
class XrSelecter(IterDataPipe):
    
    def __init__(self, dp, **selects):
        self.dp = dp
        self.selects = selects
        
        
    def __iter__(self):
        for ds in self.dp:
            yield ds.sel(**self.selects)


@functional_datapipe("xr_isel")
class XrISelecter(IterDataPipe):
    
    def __init__(self, dp, **selects):
        self.dp = dp
        self.selects = selects
        
        
    def __iter__(self):
        for ds in self.dp:
            yield ds.isel(**self.selects)


@functional_datapipe("xr_merge")
class XrMerge(IterDataPipe):
    
    def __init__(self, dp):
        self.dp = dp
        

    def __iter__(self):
        for data_arrays in self.dp:
            yield xr.merge(data_arrays)


@functional_datapipe("xr_split_dim")
class XrSplitDim(IterDataPipe):
    
    def __init__(self, dp, dim, splits):
        self.dp = dp
        self.splits = splits
        self.dim = dim
        
    def __iter__(self):
        for ds in self.dp:
            size = len(ds.coords[self.dim]) // self.splits
            for i in range(self.splits):
                yield ds.isel(**{self.dim: slice(i*size, (i+1)*size)})