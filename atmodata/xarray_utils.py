import xarray as xr


def unstack_coordinate(array: xr.DataArray, dim):
    """
    Unstack a coordinate of a DataArray, returning a tuple of DataArrays.
    The new DataArrays will have "{var}{dim}" as name, where {var} is the
    name of the original DataArray and {dim} is the value of the coordinate.
    """
    if dim not in array.coords:
        return array
    else:
        return tuple(
            array.sel({dim: value}, drop=True).rename(f"{array.name}{value}") for value in array.coords[dim].values
        )
