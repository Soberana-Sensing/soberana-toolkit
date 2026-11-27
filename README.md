# soberana-toolkit

Reusable tooling for Soberana Sensing Earth-observation work.

This is a shared library that holds the pieces we expect to use more than once: openEO helpers for Sentinel-1 and Sentinel-2, datacube assembly and merging, analysis and
index calculations, machine-learning utilities, and a thin serving layer for turning a produced
cube into a small web application.

The intended shape of a job is:

1. Read configuration describing area, period, collection and analysis.
2. Build a process graph and run it on an openEO backend (or load local data).
3. Merge the result into one datacube and apply the analysis.
4. Write the cube out (NetCDF, Zarr, COG).
5. Optionally serve it through a small web app for inspection and machine learning.

Nothing here is specific to one site, one campaign or one paper. Project-specific code stays in
the project that needs it and imports from this toolkit.

## Status

Early, as of 2026-11-27. The openEO Sentinel-2 pipeline builds a process graph from YAML, and
`soberana_toolkit.verify` exists to check that produced NetCDF files sit on the grid they claim.
Everything else is still layout. Code in `archive/` is earlier work kept on disk for reuse, is
git-ignored, and is not part of the package.

## Layout

```
config/                    configuration files (YAML), no secrets
  grids/                   named grids: CRS, pixel size, alignment policy
data/
  raw/                     data as delivered or downloaded
  interim/                 reshaped intermediates
  derived/                 analysis outputs
  aoi/                     area-of-interest GeoJSON, one per site
  inventory/               catalogue of datasets in use, tracked by git
src/soberana_toolkit/
  grid.py                  load and validate named grids
  verify.py                alignment gate for NetCDF products
  openeo/                  backend connections, collections, process graphs (S1, S2, ...)
  cube/                    cube assembly, merging, alignment, temporal and spatial reduction
  io/                      readers and writers for NetCDF, Zarr, COG, GeoJSON
  analysis/                indices, masking, statistics
  ml/                      feature preparation, training, inference, evaluation
  serve/                   web app and API layer
scripts/                   thin command-line entry points over the library
tests/                     tests
notebooks/                 exploration; not a deliverable
```

`config/` must never hold credentials. Keep secrets in the environment or in a local,
git-ignored file.

## Grids and alignment

Every product must sit on a named grid from `config/grids/`. The grid is the only place a CRS or a
pixel size is set; datatype configs name the grid and inherit the rest, so nothing can contradict
it.

`resolution_policy: native_or_grid` means the effective pixel size is the larger of the grid
resolution and the source's native resolution. Data finer than the grid is coarsened onto it;
data coarser than the grid keeps its native resolution. Pixel size never shrinks, so no product
claims detail it does not have.

The inventory at `data/inventory/datasets.yaml` declares what each dataset is, where it lives, its
native resolution and the period it covers. The verifier reads that file and checks each product
for CRS, pixel size against the policy, evenly spaced and monotonic axes, and a sane time axis:

```sh
./soberana-toolkit-venv/bin/python -m soberana_toolkit.verify
./soberana-toolkit-venv/bin/python -m soberana_toolkit.verify --grid utm28_30m some.nc
```

Output is a list of findings plus a count. Errors exit nonzero; warnings do not unless `--strict`.
The pipeline writes the grid name and a hash of the grid file into `run_metadata.json`, so an
output can always be traced to the definition that produced it.

## Data and dates

Data lives under `data/` and is not tracked, but the folder tree is. Any description of a dataset
carries its date: the acquisition window for imagery, the epoch for model output, the retrieval
date for a catalogue record. A dataset with no date is a claim with no time.

## Working rules

Work lands one piece at a time: one small module, one entry point, one test. Large speculative
builds are out of scope. See `AGENTS.md` for the working agreement.

## Development

The package is declared in `pyproject.toml` (setuptools, `src/` layout) and installed in editable
mode into a local environment. Nothing is installed into the system Python.

```sh
python3.12 -m venv soberana-toolkit-venv
./soberana-toolkit-venv/bin/python -m pip install -e ".[dev]"
./soberana-toolkit-venv/bin/python -m pytest
```

Dependencies are declared in `pyproject.toml` only; there is no `requirements.txt`. A new
dependency is added there, with a lower bound, at the moment the code that needs it is written.

## Contributing

`main` is protected. Changes arrive through a pull request and are merged there.
