"""Alignment gate for NetCDF products.

A product is only usable once it proves it sits on the grid it claims: same CRS, same pixel
size allowed by the grid policy, axis-aligned, and a sane time axis. This module never repairs
anything; it reports, and the caller decides.
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

INVENTORY = Path("data/inventory/datasets.yaml")
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


def load_inventory(path: str | Path = INVENTORY) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    entries = data.get("datasets") or []
    if not isinstance(entries, list):
        raise TypeError(f"{path}: 'datasets' must be a list")
    return entries


def check_entry(entry: dict, base: Path = Path(".")) -> list[Finding]:
    """Check one inventory row: declarations first, then the file itself if present."""
    name = entry.get("name", "<unnamed>")
    findings: list[Finding] = []
    for key in ("name", "path", "grid", "source", "native_resolution", "acquisition", "retrieved"):
        if key not in entry:
            findings.append(_error("inventory", f"{name}: missing field '{key}'"))
    if findings:
        return findings

    grid = load_grid(entry["grid"])
    declared_crs = CRS.from_user_input(entry["crs"]) if "crs" in entry else None
    if declared_crs is not None and not declared_crs.equals(CRS.from_user_input(grid.crs)):
        findings.append(_error("inventory", f"{name}: declared crs {entry['crs']} != grid {grid.crs_label}"))
    expected = grid.expected_pixel_size(entry["native_resolution"])
    if "resolution" in entry and abs(float(entry["resolution"]) - expected) > expected * TOLERANCE:
        findings.append(_error(
            "inventory", f"{name}: declared resolution {entry['resolution']} != expected {expected:g}"))

    path = Path(entry["path"])
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        return findings + [_warning("file", f"{name}: {path} not present yet, declarations only")]
    return findings + verify_file(path, grid, entry["native_resolution"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soberana-toolkit.verify", description=__doc__)
    parser.add_argument("files", nargs="*", help="NetCDF files to check")
    parser.add_argument("--grid", help="grid name for the files given on the command line")
    parser.add_argument("--inventory", default=str(INVENTORY), help="inventory YAML to check")
    parser.add_argument("--json", help="write the findings to this path")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    if args.files:
        if not args.grid:
            parser.error("--grid is required when files are given")
        grid = load_grid(args.grid)
        for f in args.files:
            findings.extend(verify_file(f, grid))
    else:
        entries = load_inventory(args.inventory)
        findings = [f for entry in entries for f in check_entry(entry)]
        if not entries:
            print(f"inventory {args.inventory} has no datasets")
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
