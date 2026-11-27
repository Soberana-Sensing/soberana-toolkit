"""Build an openEO Sentinel-2 datacube from a YAML config.

Usage:
    python -m soberana_toolkit.openeo.s2_l2a_processing cfg.yaml
    python -m soberana_toolkit.openeo.s2_l2a_processing cfg.yaml --check
    python -m soberana_toolkit.openeo.s2_l2a_processing cfg.yaml --run

Pipeline order: load -> SCL mask -> scale -> reproject to the grid -> resample/clip -> indices
                -> temporal aggregation -> gap fill -> spatial window aggregation -> save

Run from the consuming project root so project config, AOI, and output paths resolve correctly.
When the output format is NetCDF, the downloaded result is split into one file per requested
output band, named <band>.nc.
CRS and resolution come from the named grid in the consuming project's config/grids/; the grid
and its hash are recorded in run_metadata.json.
"""
import ast
import json
import math
import operator
import sys
import tempfile
from datetime import date, datetime, timedelta
from functools import reduce
from pathlib import Path
 
import openeo
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import yaml

from soberana_toolkit.grid import Grid, load_grid
 
# ---------------------------------------------------------------- allowed values (openEO spec)
PERIODS = {"hour", "day", "week", "dekad", "month", "season", "tropical-season", "year", "decade", "decade-ad"}
RESAMPLE_METHODS = {"near", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode",
                    "max", "min", "med", "q1", "q3", "rms", "sum"}
REDUCERS = {"mean", "median", "min", "max", "sum", "sd", "variance", "first", "last", "count", "product"}
GAP_FILL = {"linear"}
N_VALID_DESCRIPTION = (
    "Number of valid input observations contributing to the composite at each pixel."
)
N_IMAGES_DESCRIPTION = (
    "Number of input observations selected for the composite period before per-pixel masking."
)
 
 
def _check(value, allowed, label):
    if value not in allowed:
        raise ValueError(f"{label}: '{value}' not valid. Use one of: {', '.join(sorted(allowed))}")
 
 
# ---------------------------------------------------------------- formulas
def _bm(x) -> bool:
    """True if x is a cube still in openEO client 'band math mode' (result of .band() and arithmetic on it)."""
    return getattr(x, "_in_bandmath_mode", lambda: False)()
 
 
def _unary(x, bm_fn, pb_fn):
    """Band-math-preserving op when possible (so it can be combined further), else a per-pixel apply()."""
    return bm_fn(x) if _bm(x) else x.apply(pb_fn)
 
 
def _safe_div(a, b):
    """a / b, nodata where b == 0."""
    if isinstance(b, (int, float)):
        return a / b
    return (a / b).mask((b == 0) * 1.0)
 
 
BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow}
FUNCS = {  # name: (n_args, function)
    "sqrt": (1, lambda x: _unary(x, lambda c: c ** 0.5, lambda v: v.sqrt())),
    "abs": (1, lambda x: _unary(x, lambda c: (c ** 2) ** 0.5, lambda v: v.absolute())),
    "ln": (1, lambda x: _unary(x, lambda c: c.ln(), lambda v: v.ln())),
    "log10": (1, lambda x: _unary(x, lambda c: c.log10(), lambda v: v.log(10))),
    "exp": (1, lambda x: _unary(x, lambda c: math.e ** c, lambda v: v.exp())),
    "sqrt_clip": (1, lambda x: _unary(x, lambda c: ((c + (c ** 2) ** 0.5) / 2) ** 0.5,
                                      lambda v: ((v + v.absolute()) / 2).sqrt())),  # sqrt(max(x, 0))
    "safe_div": (2, _safe_div),
}
 
 
def eval_formula(expr: str, env: dict):
    """Evaluate an index formula over single-band cubes. Whitelisted syntax only (no eval())."""
 
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise KeyError(f"Unknown name '{n.id}' in formula '{expr}'. Known: {sorted(env)}")
            return env[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return ev(n.operand) * -1
        if isinstance(n, ast.BinOp) and type(n.op) in BINOPS:
            return BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS:
            arity, fn = FUNCS[n.func.id]
            if len(n.args) != arity:
                raise ValueError(f"{n.func.id}() takes {arity} argument(s) in formula '{expr}'")
            return fn(*[ev(a) for a in n.args])
        raise ValueError(f"Unsupported syntax in formula '{expr}': {ast.dump(n)}")
 
    return ev(ast.parse(expr, mode="eval"))
 
 
# ---------------------------------------------------------------- masking
def build_mask(cube, scl_cfg: dict):
    """Return a mask cube (1 = masked) from the SCL band, or None if masking is off."""
    if not scl_cfg:
        return None
    dil = scl_cfg.get("dilation")
    if not dil:
        raise KeyError("mask.scl.dilation is required; the plain class-list mask is no longer supported")
    # CDSE to_scl_dilation_mask; input must be a cube with only SCL, output 1 = masked
    scl_only = cube.filter_bands(["SCL"])
    return scl_only.process("to_scl_dilation_mask", data=scl_only, **dil)


# ---------------------------------------------------------------- spatial / temporal helpers
def get_aoi(aoi: dict):
    """Return the AOI geometry. The AOI is always a GeoJSON file."""
    if "geojson" not in aoi:
        raise KeyError("aoi.geojson is required: the AOI is a GeoJSON file")
    return json.loads(Path(aoi["geojson"]).read_text())


def apply_spatial(cube, sp: dict, geojson, method_key: str, grid: Grid):
    """Reproject to the grid CRS and resolution, then clip to the AOI polygon.

    Resolution and alignment come from the grid so a datatype config cannot contradict them.
    """
    rs = sp.get("resample") or {}
    default_method = "bilinear" if method_key == "method" else "near"
    method = rs.get(method_key, default_method)
    _check(method, RESAMPLE_METHODS, f"spatial.resample.{method_key}")
    cube = cube.resample_spatial(resolution=grid.resolution, projection=grid.crs,
                                 method=method, align=grid.align)
    if sp.get("clip_to_aoi"):
        cube = cube.mask_polygon(geojson)
    return cube


def temporal_aggregate(cube, agg: dict, reducer: str):
    _check(reducer, REDUCERS, "temporal reducer")
    if "intervals" in agg:  # takes precedence over period
        labels = agg.get("labels")
        if labels is not None and len(labels) != len(agg["intervals"]):
            raise ValueError(f"temporal.aggregate.labels has {len(labels)} entries, "
                             f"intervals has {len(agg['intervals'])}")
        return cube.aggregate_temporal(intervals=agg["intervals"], reducer=reducer, labels=labels)
    _check(agg["period"], PERIODS, "temporal.aggregate.period")
    return cube.aggregate_temporal_period(period=agg["period"], reducer=reducer)
 
 
# ---------------------------------------------------------------- build
def build_cube(cfg: dict, conn, grid: Grid):
    src, proc, out = cfg["source"], cfg.get("processing", {}), cfg["output"]
    roles: dict = src["roles"]
    spatial_cfg = proc.get("spatial", {}) or {}
    temporal_cfg = proc.get("temporal", {}) or {}
    end = (date.fromisoformat(str(src["time_end"])) + timedelta(days=1)).isoformat()
    geojson = get_aoi(cfg["aoi"])

    # 1. load only what is needed (+ SCL for masking)
    codes = list(dict.fromkeys(roles.values()))
    cube = conn.load_collection(
        src["collection"],
        spatial_extent=geojson,
        temporal_extent=[str(src["time_start"]), end],
        bands=codes + ["SCL"],
        max_cloud_cover=src.get("max_cloud_cover"),
    )

    # 2. scale spectral bands (indices like SI1 expect reflectance); SCL stays untouched
    scale = src.get("scale", 1.0)
    spectral = cube.filter_bands(codes)
    if scale != 1.0:
        spectral = spectral.apply(lambda x: x * scale)

    # 3. mask on the native grid (before any resampling)
    mask = build_mask(cube, proc.get("mask", {}).get("scl", {}))
    if mask is not None:
        spectral = spectral.mask(mask)

    # 4. spatial: always reproject to the grid, then optional resample and clip
    spectral = apply_spatial(spectral, spatial_cfg, geojson, "method", grid)

    # 5. formula environment: role name -> single-band cube
    env = {role: spectral.band(code) for role, code in roles.items()}
    for name, spec in (cfg.get("indices") or {}).items():
        env[name] = eval_formula(spec["formula"], env)

    # 6. SCL class layers: 0/1 indicator per observation, masked like the bands; "mean" later = fraction
    scl_layers = {}
    scl_cube = cube.filter_bands(["SCL"])
    for name, classes in (cfg.get("scl_classes") or {}).items():
        if name in env:
            raise KeyError(f"scl_classes name '{name}' clashes with a role/index")
        classes = classes if isinstance(classes, list) else [classes]
        ind = reduce(lambda a, b: a | b, [scl_cube == c for c in classes]) * 1.0
        if mask is not None:
            ind = ind.mask(mask)
        ind = apply_spatial(ind, spatial_cfg, geojson, "scl_method", grid)
        scl_layers[name] = ind.rename_labels("bands", [name])

    # 7. assemble requested outputs as (name, cube) with a band dimension
    parts = []
    for name in out["bands"]:
        if name in scl_layers:
            parts.append((name, scl_layers[name]))
        elif name in env:
            parts.append((name, env[name].add_dimension("bands", name, type="bands")))  # .band() drops band dim
        else:
            raise KeyError(f"output.bands has unknown '{name}'. Known: {sorted(list(env) + list(scl_layers))}")

    # 8. temporal aggregation (+ valid observation count) and gap fill
    agg = temporal_cfg.get("aggregate")
    if agg:
        parts = [(n, temporal_aggregate(c, agg, "mean" if n in scl_layers else agg["reducer"])) for n, c in parts]
        if _include_n_valid(cfg):
            first = next(iter(roles.values()))
            n = temporal_aggregate(spectral.band(first), agg, "count")
            parts.append(("n_valid", n.add_dimension("bands", "n_valid", type="bands")))
        if _include_n_images(cfg):
            scene_cube = cube.filter_bands(["SCL"]).band("SCL")
            scene_cube = scene_cube.process(
                "replace_nodata", data=scene_cube, replacement=1
            )
            scene_cube = apply_spatial(
                scene_cube, spatial_cfg, geojson, "scl_method", grid
            )
            n = temporal_aggregate(scene_cube, agg, "count")
            parts.append(("n_images", n.add_dimension("bands", "n_images", type="bands")))
        gap_fill = temporal_cfg.get("gap_fill")
        if gap_fill is not None:
            _check(gap_fill, GAP_FILL, "temporal.gap_fill")
            parts = [(n, c if n in {"n_valid", "n_images"} else c.apply_dimension(
                dimension="t", process="array_interpolate_linear"
            ))
                     for n, c in parts]
 
    # 9. spatial window aggregation (experimental in openEO)
    win = spatial_cfg.get("aggregate_window")
    if win:
        _check(win.get("reducer", "mean"), REDUCERS, "spatial.aggregate_window.reducer")
        parts = [(n, c.aggregate_spatial_window(reducer=win.get("reducer", "mean"), size=win["size"],
                                                boundary=win.get("boundary", "pad"),
                                                align=win.get("align", "upper-left")))
                 for n, c in parts]
 
    return reduce(lambda a, b: a.merge_cubes(b), [c for _, c in parts])
 
 
# ---------------------------------------------------------------- run metadata
def write_metadata(cfg: dict, out_dir: Path, grid: Grid, job_id=None) -> Path:
    """Record what produced this output, including the CRS. The NetCDF carries its own CRS
    from the backend grid; this file is the readable, diffable record next to it."""
    proc = cfg.get("processing") or {}
    meta = {
        "name": cfg["name"],
        "collection": cfg["source"]["collection"],
        "provider": cfg["source"].get("provider"),
        "citation": cfg["source"].get("citation"),
        "grid": grid.name,
        "grid_path": str(grid.path),
        "grid_sha256": grid.sha256(),
        "crs": grid.crs_label,
        "resolution": grid.resolution,
        "resolution_policy": grid.policy,
        "time_start": str(cfg["source"]["time_start"]),
        "time_end": str(cfg["source"]["time_end"]),
        "aoi_geojson": cfg["aoi"]["geojson"],
        "temporal": (proc.get("temporal") or {}).get("aggregate"),
        "gap_fill": (proc.get("temporal") or {}).get("gap_fill"),
        "bands": cfg["output"]["bands"],
        "n_valid": _include_n_valid(cfg),
        "n_images": _include_n_images(cfg),
        "format": cfg["output"].get("format", "NetCDF"),
        "config": str(cfg.get("_path", "")) or None,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    if job_id:
        meta["job_id"] = job_id
    path = out_dir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


# ---------------------------------------------------------------- backend check
def used_processes(node) -> set:
    found = set()
    if isinstance(node, dict):
        if "process_id" in node:
            found.add(node["process_id"])
        for v in node.values():
            found |= used_processes(v)
    elif isinstance(node, list):
        for v in node:
            found |= used_processes(v)
    return found
 
 
def split_netcdf_by_band(
    source: Path,
    bands: list[str],
    out_dir: Path,
    crs: int | str,
    descriptions: dict[str, str] | None = None,
) -> list[Path]:
    """Write one NetCDF per requested band from an openEO multi-band result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with xr.open_dataset(source) as ds:
        if "bands" in ds.dims:
            labels = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in ds["bands"].values
            ]
            missing = [band for band in bands if band not in labels]
            if missing:
                raise ValueError(f"{source}: requested bands not present: {missing}; found {labels}")
            band_indices = {label: index for index, label in enumerate(labels)}
        else:
            missing = [band for band in bands if band not in ds.data_vars]
            if missing:
                raise ValueError(
                    f"{source}: no 'bands' dimension and requested variables not present: "
                    f"{missing}; found {list(ds.data_vars)}"
                )
            band_indices = None

        for band in bands:
            if not band or Path(band).name != band or band in {".", ".."}:
                raise ValueError(f"invalid output band name for a filename: {band!r}")
            selected = (
                ds.isel(bands=band_indices[band], drop=True)
                if band_indices is not None
                else ds[[band]]
            )
            if "openEO" in selected.data_vars and band not in selected.data_vars:
                selected = selected.rename({"openEO": band})
            elif len(selected.data_vars) == 1 and band not in selected.data_vars:
                selected = selected.rename({next(iter(selected.data_vars)): band})
            if descriptions and band in descriptions:
                if band not in selected.data_vars:
                    raise ValueError(f"{source}: no data variable found for output band {band!r}")
                selected[band].attrs["description"] = descriptions[band]
                selected.attrs["description"] = descriptions[band]
            selected = selected.rio.write_crs(crs, grid_mapping_name="crs")
            for variable in selected.variables.values():
                variable.encoding.pop("chunksizes", None)
            path = out_dir / f"{band}.nc"
            selected.to_netcdf(path)
            paths.append(path)
    return paths


def _requested_output_bands(cfg: dict) -> list[str]:
    bands = list(cfg["output"]["bands"])
    if _include_n_valid(cfg):
        bands.append("n_valid")
    if _include_n_images(cfg):
        bands.append("n_images")
    return bands


def _include_n_images(cfg: dict) -> bool:
    """Include the selected source-observation count for temporal composites."""
    processing = cfg.get("processing") or {}
    return bool((processing.get("temporal") or {}).get("aggregate"))


def _include_n_valid(cfg: dict) -> bool:
    """Include per-pixel valid-observation counts by default for temporal composites."""
    processing = cfg.get("processing") or {}
    has_temporal_aggregate = bool((processing.get("temporal") or {}).get("aggregate"))
    return has_temporal_aggregate and cfg["output"].get("n_valid", True)


def _output_descriptions(cfg: dict) -> dict[str, str]:
    descriptions = {
        name: spec["description"]
        for name, spec in (cfg.get("indices") or {}).items()
        if spec.get("description")
    }
    configured = (cfg.get("output") or {}).get("descriptions") or {}
    if not isinstance(configured, dict):
        raise TypeError("output.descriptions must be a mapping of band names to descriptions")
    descriptions.update(configured)
    if _include_n_valid(cfg):
        descriptions.setdefault("n_valid", N_VALID_DESCRIPTION)
    if _include_n_images(cfg):
        descriptions.setdefault("n_images", N_IMAGES_DESCRIPTION)
    if any(not isinstance(name, str) or not isinstance(text, str) or not text.strip()
           for name, text in descriptions.items()):
        raise ValueError("output descriptions must be non-empty strings")
    return descriptions


def main(path: str, run: bool, check: bool):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_path"] = path
    grid = load_grid(cfg["grid"])
    conn = openeo.connect(cfg["source"].get("backend_url", "openeo.dataspace.copernicus.eu")).authenticate_oidc()
    cube = build_cube(cfg, conn, grid)
    result = cube.save_result(format=cfg["output"].get("format", "NetCDF"), options=cfg["output"].get("options"))

    out_dir = Path(cfg["output"].get("dir", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_json = result.to_json()
    (out_dir / "process_graph.json").write_text(graph_json)
    print(f"grid: {grid.name} {grid.crs_label} {grid.resolution:g} m ({grid.policy})")
    print(f"process graph written to {out_dir / 'process_graph.json'}")

    if check:
        available = {p["id"] for p in conn.list_processes()}
        missing = sorted(used_processes(json.loads(graph_json)) - available)
        print("all processes supported by backend" if not missing else f"NOT supported by backend: {missing}")

    job = None
    if run:
        job = result.create_job(title=cfg["name"], job_options=cfg["output"].get("job_options"))
        job.start_and_wait()
        if str(cfg["output"].get("format", "NetCDF")).lower() == "netcdf":
            with tempfile.TemporaryDirectory(prefix="s2-l2a-") as temp_dir:
                downloaded = job.get_results().download_files(temp_dir)
                netcdf_files = [file for file in downloaded if file.suffix.lower() == ".nc"]
                if len(netcdf_files) != 1:
                    raise ValueError(
                        f"expected one NetCDF result to split, found {len(netcdf_files)}: "
                        f"{netcdf_files}"
                    )
                outputs = split_netcdf_by_band(
                    netcdf_files[0],
                    _requested_output_bands(cfg),
                    out_dir,
                    grid.crs,
                    _output_descriptions(cfg),
                )
            for output in outputs:
                print(f"output written to {output}")
        else:
            job.get_results().download_files(out_dir)

    meta_path = write_metadata(cfg, out_dir, grid, job_id=getattr(job, "job_id", None))
    print(f"run metadata written to {meta_path}")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], run="--run" in sys.argv, check="--check" in sys.argv)
 