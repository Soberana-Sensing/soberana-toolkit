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

Early, as of 2026-11-27, but installable from a local checkout. The implemented pieces are the
openEO Sentinel-2 pipeline, named-grid and NetCDF verification, and a local browser for derived
NetCDF products. Cube assembly, analysis, ML, and API modules are not implemented yet; they are
future layout, not package features. Code in `archive/` is earlier work kept on disk for reuse, is
git-ignored, and is not part of the package.

The package contains reusable code, not project data or configuration. Keep a project's
`config/` and `data/` in that project, install this package into the project's environment, and
run toolkit commands from the project root. Config and data paths are resolved relative to the
current working directory unless a setting explicitly allows an override.

## Layout

The toolkit repository contains reusable code and its tests:

```
src/soberana_toolkit/
  grid.py                  load and validate named grids
  verify.py                alignment gate for NetCDF products
  openeo/                  openEO processing
  serve/                   local browser for derived NetCDF products
tests/                     tests
```

Each consuming project keeps its own inputs and outputs:

```
config/
  grids/                   named grids: CRS, pixel size, alignment policy
  datasets/                dataset source, processing, and output configuration
data/
  raw/                     data as delivered or downloaded
  interim/                 reshaped intermediates
  derived/                 analysis outputs
  aoi/                     area-of-interest GeoJSON, one per site
```

`config/` must never hold credentials. Keep secrets in the environment or in a local,
git-ignored file.

## Grids and alignment

Every product in a consuming project must sit on a named grid from its `config/grids/`. The grid
is the only place a CRS or a pixel size is set; dataset configs name the grid and inherit the rest,
so nothing can contradict it.

`resolution_policy: native_or_grid` means the effective pixel size is the larger of the grid
resolution and the source's native resolution. Data finer than the grid is coarsened onto it;
data coarser than the grid keeps its native resolution. Pixel size never shrinks, so no product
claims detail it does not have.

Each dataset configuration in the consuming project's `config/datasets/` names its grid and
output directory. Set `source.native_resolution` there so the verifier can check the grid's
resolution policy. Run these commands from the consuming project root, with the toolkit installed
in that project's environment:

```sh
python -m soberana_toolkit.verify
python -m soberana_toolkit.verify --dataset-config config/datasets/<dataset>.yaml
python -m soberana_toolkit.verify --grid <grid-name> path/to/product.nc
```

Output is a list of findings plus a count. Errors exit nonzero; warnings do not unless `--strict`.
The pipeline writes the grid name and a hash of the grid file into `run_metadata.json`, so an
output can always be traced to the definition that produced it. The dataset YAML is the source
of truth; there is no separate inventory to keep in sync.

Temporal composites produce two count outputs by default: `n_images.nc` records the number of
source observations selected for each period before per-pixel masking, and `n_valid.nc` records how
many unmasked observations contributed at each pixel. Set `output.n_valid: false` to omit the
per-pixel count; the monthly source-observation count remains available.

## Data and dates

Consuming projects keep data under `data/`; large data files are not tracked. Any dataset config
carries its acquisition window or epoch. A dataset with no date is a claim with no time.

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

## Browsing derived products

Install the optional serving dependencies in the consuming project's environment and run the app
from the consuming project root:

```sh
python -m pip install -e "/path/to/soberana-toolkit[serve]"
python -m streamlit run "/path/to/soberana-toolkit/src/soberana_toolkit/serve/app.py"
```

The browser groups `.nc` files under `data/derived/` by their parent dataset directory. Choose the
dataset (for example, `s2_l2a_monthly`) as the product, then choose a raster variable. It provides
a time selector for time-series rasters; static rasters need no time selection. For temporal
products, it shows the selected month's source-observation count and a map of valid observations
per pixel when sibling `n_images.nc` and `n_valid.nc` files exist. It also shows a variable
histogram for the selected month and a monthly time series split by water, soil, and vegetation
SCL fractions. Pixels contribute fractionally to each class instead of being assigned a hard class.
The time series uses fraction-weighted medians to reduce sensitivity to isolated extreme values.
Histograms show the 1st-to-99th-percentile range, with more extreme values retained in the outer
bins so they do not stretch the entire plot. Dataset discovery reads NetCDF metadata, not raster
arrays. The map and histogram read the selected month's slices; the time series streams the
selected variable and its SCL fractions month by month, then caches its medians. Changing month
reuses that time series and computes only the selected month's histogram. It reads existing
products; it does not run processing.

## Contributing

`main` is protected. Changes arrive through a pull request and are merged there. GitHub Actions
runs the test suite on every push and pull request.
