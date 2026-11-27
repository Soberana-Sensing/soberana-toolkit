"""Discovery helpers for derived NetCDF products."""

from pathlib import Path

import xarray as xr

_X_COORDINATES = ("x", "lon", "longitude")
_Y_COORDINATES = ("y", "lat", "latitude")


def discover_products(root: Path) -> list[Path]:
    """Return NetCDF products recursively under a derived-data directory."""
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.nc") if path.is_file())


def spatial_coordinates(data: xr.DataArray) -> tuple[str, str] | None:
    """Return the one-dimensional spatial coordinate names for a raster."""
    x_name = next(
        (
            name
            for name in _X_COORDINATES
            if name in data.dims and name in data.coords and data[name].ndim == 1
        ),
        None,
    )
    y_name = next(
        (
            name
            for name in _Y_COORDINATES
            if name in data.dims and name in data.coords and data[name].ndim == 1
        ),
        None,
    )
    if x_name is None or y_name is None:
        return None
    return x_name, y_name


def spatial_variables(dataset: xr.Dataset) -> list[str]:
    """Return data-variable names with one-dimensional spatial coordinates."""
    return [
        name
        for name, data in dataset.data_vars.items()
        if spatial_coordinates(data) is not None
    ]
