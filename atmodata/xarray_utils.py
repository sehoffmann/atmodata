import numpy as np
import xarray as xr


def unstack_coordinate(array: xr.DataArray, dim):
    """
    Unstack a coordinate of a DataArray, returning a tuple of DataArrays.
    The new DataArrays will have "{var}{coord}" as name, where {var} is the
    name of the original DataArray and {coord} is the value of the coordinate.
    """
    if dim not in array.coords:
        return array
    else:
        coords = np.atleast_1d(array.coords[dim].values)  # 0d arrays are not iterable
        return tuple(array.sel({dim: coord}, drop=True).rename(f"{array.name}{coord}") for coord in coords)
