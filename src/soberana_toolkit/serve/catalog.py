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


def discover_datasets(root: Path) -> list[Path]:
    """Return dataset directories that contain NetCDF products."""
    return sorted({product.parent for product in discover_products(root)})


def discover_dataset_variables(dataset_dir: Path) -> dict[str, Path]:
    """Map spatial variable names in one dataset directory to their NetCDF files."""
    variables: dict[str, Path] = {}
    for path in sorted(dataset_dir.glob("*.nc")):
        with xr.open_dataset(path, decode_coords="all") as dataset:
            for name in spatial_variables(dataset):
                if name in variables and variables[name] != path:
                    raise ValueError(
                        f"{dataset_dir}: spatial variable {name!r} appears in multiple files"
                    )
                variables[name] = path
    return variables


def find_valid_count_product(product: Path) -> Path | None:
    """Find the sibling valid-observation count file for a temporal product."""
    if product.name == "n_valid.nc":
        return None
    count_product = product.with_name("n_valid.nc")
    return count_product if count_product.is_file() else None


def find_scene_count_product(product: Path) -> Path | None:
    """Find the sibling monthly source-observation count file."""
    if product.name == "n_images.nc":
        return None
    count_product = product.with_name("n_images.nc")
    return count_product if count_product.is_file() else None


def matching_count_slice(
    dataset: xr.Dataset, variable: str, time_value: object
) -> xr.DataArray | None:
    """Load a count raster matching one product timestamp, if present."""
    if variable not in dataset.data_vars:
        raise ValueError(f"count file has no {variable!r} data variable")
    data = dataset[variable]
    time_dim = next((name for name in ("t", "time") if name in data.dims), None)
    if time_dim is None:
        return None
    matches = np.flatnonzero(dataset[time_dim].values == time_value)
    if matches.size > 1:
        raise ValueError(f"multiple valid-observation slices match {time_value}")
    if matches.size == 0:
        return None
    return data.isel({time_dim: int(matches[0])}).load()


def single_count_value(values: np.ndarray) -> float | None:
    """Return a single finite count value; reject a spatially varying scene count."""
    finite_values = np.unique(np.asarray(values)[np.isfinite(values)])
    if finite_values.size == 0:
        return None
    if finite_values.size != 1:
        raise ValueError("monthly source-observation count varies across pixels")
    return float(finite_values[0])


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
