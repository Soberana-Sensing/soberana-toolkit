"""Alignment gate for NetCDF products.

A product is only usable once it proves it sits on the grid it claims: same CRS, same pixel
size allowed by the grid policy, axis-aligned, and a sane time axis. This module never repairs
anything; it reports, and the caller decides.

Run from the repository root in the project environment:

    python -m soberana_toolkit.verify

    python -m soberana_toolkit.verify --dataset-config config/datasets/s2_l2a_monthly.yaml

    python -m soberana_toolkit.verify --grid utm28_30m some.nc

Without arguments, all YAML files in config/datasets/ are checked. Use --dataset-config to
check a specific dataset config, or pass NetCDF files with --grid to check files directly.
Do not run this file by path; package mode is required for its relative imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import yaml
from pyproj import CRS

from .grid import Grid, load_grid

DATASET_CONFIG_DIR = Path("config/datasets")
TIME_NAMES = ("t", "time")
TOLERANCE = 1e-3  # relative, applied to pixel size


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warning"
    check: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.check}: {self.message}"


def _error(check: str, message: str) -> Finding:
    return Finding("error", check, message)


def _warning(check: str, message: str) -> Finding:
    return Finding("warning", check, message)


def _spatial_axes(ds: xr.Dataset):
    """Return (x_name, y_name, x_spacing, y_spacing) from 1D coordinate arrays."""
    x_name = next((n for n in ("x", "lon", "longitude") if n in ds.coords), None)
    y_name = next((n for n in ("y", "lat", "latitude") if n in ds.coords), None)
    if x_name is None or y_name is None:
        raise ValueError(f"no 1D x/y coordinates, saw coords {list(ds.coords)}")
    x, y = ds[x_name].values, ds[y_name].values
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"coordinate arrays must be 1D, got {x.ndim}D/{y.ndim}D")
    if x.size < 2 or y.size < 2:
        raise ValueError("coordinate arrays need at least 2 cells")
    return x_name, y_name, float(x[1] - x[0]), float(y[1] - y[0])


def _uniform(axis: np.ndarray) -> bool:
    steps = np.diff(axis)
    return bool(np.allclose(steps, steps[0], rtol=TOLERANCE, atol=0))


def check_dataset(ds: xr.Dataset, grid: Grid, native_resolution: float | None = None,
                  source: str = "<dataset>") -> list[Finding]:
    """Check one open dataset against a grid."""
    findings: list[Finding] = []

    actual_crs = ds.rio.crs
    if actual_crs is None:
        findings.append(_error("crs", f"{source}: no CRS found (no spatial_ref or grid_mapping)"))
    else:
        if not CRS.from_user_input(actual_crs).equals(CRS.from_user_input(grid.crs)):
            findings.append(_error("crs", f"{source}: {actual_crs.to_string()} != grid {grid.crs_label}"))

    try:
        x_name, y_name, dx, dy = _spatial_axes(ds)
    except ValueError as exc:
        findings.append(_error("axes", f"{source}: {exc}"))
        return findings

    expected = grid.expected_pixel_size(native_resolution)
    tol = max(abs(expected) * TOLERANCE, 1e-9)
    px, py = abs(dx), abs(dy)
    if abs(px - py) > tol:
        findings.append(_error("pixel_size", f"{source}: anisotropic pixels {px:g} x {py:g}"))
    if abs(px - expected) > tol:
        findings.append(_error(
            "pixel_size",
            f"{source}: {px:g} != expected {expected:g} (grid {grid.resolution:g}, native {native_resolution})"))
    if native_resolution is not None and px < native_resolution - tol:
        findings.append(_error(
            "resolution_policy",
            f"{source}: pixel {px:g} is finer than native {native_resolution:g}; upsampling not allowed"))

    if not _uniform(ds[x_name].values):
        findings.append(_error("axes", f"{source}: x coordinate not evenly spaced"))
    if not _uniform(ds[y_name].values):
        findings.append(_error("axes", f"{source}: y coordinate not evenly spaced"))

    ds_y = ds[y_name].values
    if not (np.all(np.diff(ds_y) < 0) or np.all(np.diff(ds_y) > 0)):
        findings.append(_error("axes", f"{source}: y coordinate not monotonic"))
    ds_x = ds[x_name].values
    if not (np.all(np.diff(ds_x) < 0) or np.all(np.diff(ds_x) > 0)):
        findings.append(_error("axes", f"{source}: x coordinate not monotonic"))

    time_name = next((n for n in TIME_NAMES if n in ds.coords), None)
    if time_name is None:
        findings.append(_warning("time", f"{source}: no time coordinate; fine for a static layer"))
    else:
        t = ds[time_name].values
        if not np.issubdtype(t.dtype, np.datetime64):
            findings.append(_error("time", f"{source}: time dtype {t.dtype}, not datetime64 (CF units not decoded?)"))
        else:
            if t.size < 2:
                findings.append(_warning("time", f"{source}: single time step, nothing to compare"))
            elif np.any(np.diff(t) <= np.timedelta64(0, "s")):
                findings.append(_error("time", f"{source}: time not strictly increasing or has duplicates"))
            if np.any(np.isnat(t)):
                findings.append(_error("time", f"{source}: time coordinate contains NaT"))

    return findings


def verify_file(path: str | Path, grid: Grid, native_resolution: float | None = None) -> list[Finding]:
    path = Path(path)
    if not path.is_file():
        return [_error("file", f"{path}: not found")]
    with xr.open_dataset(path, decode_coords="all") as ds:
        return check_dataset(ds, grid, native_resolution, source=str(path))


def load_dataset_configs(directory: str | Path = DATASET_CONFIG_DIR) -> list[Path]:
    directory = Path(directory)
    return sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])


def verify_path(path: str | Path, grid: Grid, native_resolution: float,
                dataset_name: str) -> list[Finding]:
    """Verify one NetCDF file or every top-level NetCDF in an output directory."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.nc"))
        if not files:
            return [_warning("file", f"{dataset_name}: no NetCDF files found in {path}")]
        return [
            finding
            for file in files
            for finding in verify_file(file, grid, native_resolution)
        ]
    if not path.is_file():
        return [_warning("file", f"{dataset_name}: output path {path} not present yet")]
    return verify_file(path, grid, native_resolution)


def check_dataset_config(path: str | Path) -> list[Finding]:
    """Check products declared by one dataset configuration."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return [_error("dataset_config", f"{path}: expected a YAML mapping")]

    name = data.get("name", path.stem)
    grid_name = data.get("grid")
    source = data.get("source")
    output = data.get("output")
    if not isinstance(grid_name, str) or not grid_name:
        return [_error("dataset_config", f"{name}: missing or invalid field 'grid'")]
    if not isinstance(source, dict):
        return [_error("dataset_config", f"{name}: missing field 'source.native_resolution'")]
    if source.get("native_resolution") is None:
        return [_error("dataset_config", f"{name}: missing field 'source.native_resolution'")]
    if not isinstance(output, dict):
        return [_error("dataset_config", f"{name}: missing field 'output.dir'")]
    output_dir = output.get("dir")
    if not isinstance(output_dir, str) or not output_dir:
        return [_error("dataset_config", f"{name}: missing or invalid field 'output.dir'")]

    try:
        native_resolution = float(source["native_resolution"])
    except (TypeError, ValueError):
        return [_error("dataset_config", f"{name}: source.native_resolution must be numeric")]
    if native_resolution <= 0:
        return [_error("dataset_config", f"{name}: source.native_resolution must be positive")]

    grid = load_grid(grid_name)
    return verify_path(output_dir, grid, native_resolution, str(name))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="soberana-toolkit.verify",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help="NetCDF files to check")
    parser.add_argument("--grid", help="grid name for the files given on the command line")
    parser.add_argument(
        "--dataset-config",
        action="append",
        dest="dataset_configs",
        help="dataset YAML to check; may be supplied more than once",
    )
    parser.add_argument("--json", help="write the findings to this path")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    if args.files:
        if args.dataset_configs:
            parser.error("--dataset-config cannot be combined with NetCDF files")
        if not args.grid:
            parser.error("--grid is required when files are given")
        grid = load_grid(args.grid)
        for f in args.files:
            findings.extend(verify_file(f, grid))
    else:
        if args.grid:
            parser.error("--grid is only used when NetCDF files are given")
        configs = args.dataset_configs or load_dataset_configs()
        findings = [f for config in configs for f in check_dataset_config(config)]
        if not configs:
            print(f"no dataset YAML files found in {DATASET_CONFIG_DIR}")
            return 0

    for finding in findings:
        print(finding)
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(f) for f in findings], indent=2) + "\n")

    errors = sum(f.level == "error" for f in findings)
    warnings = sum(f.level == "warning" for f in findings)
    print(f"{errors} error(s), {warnings} warning(s)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
