"""Fractional SCL-class statistics for monthly raster products."""

from collections.abc import Mapping

import numpy as np
import xarray as xr

from soberana_toolkit.serve.catalog import spatial_coordinates

SCL_CLASS_STYLES = {
    "scl_water": ("Water", "#2878b5"),
    "scl_soil": ("Soil", "#9a6b3f"),
    "scl_vegetation": ("Vegetation", "#399447"),
}
_TIME_DIMS = ("t", "time")


def fractional_class_statistics(
    values: xr.DataArray,
    class_fractions: Mapping[str, xr.DataArray],
    *,
    time_index: int | None = None,
    include_histograms: bool = True,
) -> dict:
    """Compute soft-membership histograms and weighted medians for each time slice."""
    time_dim = next((name for name in _TIME_DIMS if name in values.dims), None)
    if time_dim is None:
        raise ValueError("SCL-class statistics require a temporal variable")
    if not class_fractions:
        raise ValueError("no SCL class-fraction rasters are available")

    times = values[time_dim].values
    if time_index is not None and not 0 <= time_index < len(times):
        raise IndexError(f"time index {time_index} is out of range")
    time_indices = range(len(times)) if time_index is None else (time_index,)
    classes = {
        name: {
            "label": SCL_CLASS_STYLES[name][0],
            "color": SCL_CLASS_STYLES[name][1],
            "medians": [],
            "support": [],
            "histograms": [],
        }
        for name in SCL_CLASS_STYLES
        if name in class_fractions
    }

    for current_index in time_indices:
        time_value = times[current_index]
        value_slice = _spatial_slice(values.isel({time_dim: current_index}, drop=True))
        value_values = np.asarray(value_slice.values)
        finite_values = value_values[np.isfinite(value_values)]
        edges = (
            _histogram_edges(finite_values)
            if include_histograms and finite_values.size
            else None
        )

        for name, stats in classes.items():
            fraction_slice = _time_slice(class_fractions[name], time_value)
            if fraction_slice is None:
                stats["medians"].append(None)
                stats["support"].append(0.0)
                stats["histograms"].append(None)
                continue

            fraction_slice = _spatial_slice(fraction_slice)
            value_aligned, fraction_aligned = xr.align(
                value_slice, fraction_slice, join="exact", copy=False
            )
            fraction_values = np.asarray(fraction_aligned.values)
            valid = np.isfinite(value_values) & np.isfinite(fraction_values)
            valid_fractions = fraction_values[valid]
            if np.any((valid_fractions < -1e-6) | (valid_fractions > 1 + 1e-6)):
                raise ValueError(f"{name}: class fractions must be between 0 and 1")
            weights = np.clip(valid_fractions, 0, 1)
            valid_data = np.asarray(value_aligned.values)[valid]
            support = float(weights.sum())

            if support == 0:
                stats["medians"].append(None)
                stats["support"].append(0.0)
                stats["histograms"].append(None)
                continue

            stats["medians"].append(_weighted_median(valid_data, weights))
            stats["support"].append(support)
            if not include_histograms:
                stats["histograms"].append(None)
                continue

            clipped_data = np.clip(valid_data, edges[0], edges[-1])
            counts, _ = np.histogram(clipped_data, bins=edges, weights=weights)
            stats["histograms"].append(
                {
                    "edges": edges.tolist(),
                    "probability": (counts / support).tolist(),
                    "support": support,
                    "lower_tail_fraction": float(weights[valid_data < edges[0]].sum() / support),
                    "upper_tail_fraction": float(weights[valid_data > edges[-1]].sum() / support),
                }
            )

    return {
        "times": [str(times[index])[:10] for index in time_indices],
        "classes": classes,
    }


def _time_slice(data: xr.DataArray, time_value: object) -> xr.DataArray | None:
    time_dim = next((name for name in _TIME_DIMS if name in data.dims), None)
    if time_dim is None:
        return data
    matches = np.flatnonzero(data[time_dim].values == time_value)
    if matches.size > 1:
        raise ValueError(f"multiple SCL class slices match {time_value}")
    if matches.size == 0:
        return None
    return data.isel({time_dim: int(matches[0])}, drop=True)


def _spatial_slice(data: xr.DataArray) -> xr.DataArray:
    axes = spatial_coordinates(data)
    if axes is None:
        raise ValueError(f"{data.name}: raster needs one-dimensional x/y coordinates")
    x_name, y_name = axes
    for dim in data.dims:
        if dim not in axes:
            if data.sizes[dim] != 1:
                raise ValueError(f"{data.name}: unsupported non-spatial dimension {dim!r}")
            data = data.isel({dim: 0}, drop=True)
    return data.transpose(y_name, x_name)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    positive = weights > 0
    values = values[positive]
    weights = weights[positive]
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - sorted_weights / 2) / sorted_weights.sum()
    return float(np.interp(0.5, positions, sorted_values))


def _histogram_edges(values: np.ndarray) -> np.ndarray:
    if values.size >= 100:
        lower, upper = np.percentile(values, (1, 99))
    else:
        lower, upper = float(values.min()), float(values.max())
    if lower >= upper:
        lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        return np.linspace(lower - 0.5, upper + 0.5, 21)

    q25, q75 = np.percentile(values, (25, 75))
    width = 2 * (q75 - q25) / np.cbrt(values.size)
    if width <= 0:
        bin_count = 20
    else:
        requested_bins = (upper - lower) / width
        bin_count = 50 if not np.isfinite(requested_bins) else int(np.ceil(requested_bins))
    return np.linspace(lower, upper, min(50, max(1, bin_count)) + 1)
