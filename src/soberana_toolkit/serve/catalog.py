"""Discovery helpers for derived NetCDF products."""

from pathlib import Path

import numpy as np
import xarray as xr

_X_COORDINATES = ("x", "lon", "longitude")
_Y_COORDINATES = ("y", "lat", "latitude")


def discover_products(root: Path) -> list[Path]:
    """Return NetCDF products recursively under a derived-data directory."""
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.nc") if path.is_file())


def find_valid_count_product(product: Path) -> Path | None:
    """Find the sibling valid-observation count file for a temporal product."""
    if product.name == "n_valid.nc":
        return None
    count_product = product.with_name("n_valid.nc")
    return count_product if count_product.is_file() else None


def matching_valid_observation_slice(
    dataset: xr.Dataset, time_value: object
) -> xr.DataArray | None:
    """Load the n_valid raster matching one product timestamp, if present."""
    if "n_valid" not in dataset.data_vars:
        raise ValueError("valid-observation count file has no 'n_valid' data variable")
    data = dataset["n_valid"]
    time_dim = next((name for name in ("t", "time") if name in data.dims), None)
    if time_dim is None:
        return None
    matches = np.flatnonzero(dataset[time_dim].values == time_value)
    if matches.size > 1:
        raise ValueError(f"multiple valid-observation slices match {time_value}")
    if matches.size == 0:
        return None
    return data.isel({time_dim: int(matches[0])}).load()


def summarize_valid_observations(values: np.ndarray) -> tuple[float, float, float] | None:
    """Return median, minimum, and maximum finite valid-observation counts."""
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    median, minimum, maximum = np.median(finite_values), finite_values.min(), finite_values.max()
    return float(median), float(minimum), float(maximum)


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
