# soberana-toolkit

Reusable tooling for Soberana Sensing Earth-observation work.

This is a shared library, not a single product. It holds the pieces we expect to use more than
once: openEO helpers for Sentinel-1 and Sentinel-2, datacube assembly and merging, analysis and
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

Scaffolding only, as of 2026-11-27. Folders and conventions exist; most modules are empty. Code
in `archive/` is earlier work that is kept on disk for reuse, is git-ignored, and is not part of
the package.

## Layout

```
config/                    configuration files (YAML), no secrets
data/
  raw/                     data as delivered or downloaded
  interim/                 reshaped intermediates
  derived/                 analysis outputs
  inventory/               catalogue records of what data exists, with dates
src/soberana_toolkit/
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
