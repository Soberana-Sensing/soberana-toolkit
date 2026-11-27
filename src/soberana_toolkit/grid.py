"""Named grids. One CRS, one pixel size, one policy, loaded from ``config/grids/``.

A grid is the only place a CRS or a resolution may be set. Datatype configs name a grid and
inherit the rest, so there is nothing to contradict.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ALIGN = {"upper-left", "upper-right", "lower-left", "lower-right"}
POLICIES = {"native_or_grid"}
REQUIRED = {"name", "crs", "resolution", "resolution_policy", "align"}


def grids_dir() -> Path:
    """Directory holding grid definitions. Override with SOBERANA_TOOLKIT_GRIDS."""
    return Path(os.environ.get("SOBERANA_TOOLKIT_GRIDS", "config/grids"))


def parse_crs(crs) -> int | str:
    """EPSG:<code> becomes an int the backend accepts. Anything else passes through as WKT."""
    if not crs:
        raise ValueError("grid.crs is required")
    if isinstance(crs, int):
        return crs
    if isinstance(crs, str) and crs.upper().startswith("EPSG:"):
        code = crs.split(":", 1)[1]
        if not code.isdigit():
            raise ValueError(f"grid.crs '{crs}' is not a valid EPSG code")
        return int(code)
    return crs


@dataclass(frozen=True)
class Grid:
    name: str
    crs: int | str
    resolution: float
    policy: str
    align: str
    path: Path

    @property
    def crs_label(self) -> str:
        return f"EPSG:{self.crs}" if isinstance(self.crs, int) else str(self.crs)

    def expected_pixel_size(self, native_resolution: float | None = None) -> float:
        """Pixel size a product on this grid must have, given its native resolution."""
        if native_resolution is None:
            return self.resolution
        if self.policy == "native_or_grid":
            return max(self.resolution, float(native_resolution))
        return self.resolution

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def load_grid(name: str | Path, directory: str | Path | None = None) -> Grid:
    path = Path(name)
    if path.suffix not in {".yaml", ".yml"}:
        path = Path(directory or grids_dir()) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"grid definition not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    missing = REQUIRED - data.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    if data["resolution_policy"] not in POLICIES:
        raise ValueError(f"{path}: resolution_policy '{data['resolution_policy']}' not in {sorted(POLICIES)}")
    if data["align"] not in ALIGN:
        raise ValueError(f"{path}: align '{data['align']}' not in {sorted(ALIGN)}")
    resolution = float(data["resolution"])
    if resolution <= 0:
        raise ValueError(f"{path}: resolution must be positive, got {resolution}")
    return Grid(
        name=str(data["name"]),
        crs=parse_crs(data["crs"]),
        resolution=resolution,
        policy=str(data["resolution_policy"]),
        align=str(data["align"]),
        path=path,
    )
