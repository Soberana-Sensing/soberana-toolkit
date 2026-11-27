# Inventory

Catalogue of the datasets this toolkit consumes and produces: where each file lives, which grid
it sits on, its native resolution, and the period it covers.

`datasets.yaml` is tracked by git because it is metadata, not data. The data files it points to
stay under `data/` and are ignored.

Run the alignment gate:

```sh
./soberana-toolkit-venv/bin/python -m soberana_toolkit.verify
```

Every row is checked against its grid file in `config/grids/`. A row whose file is missing warns;
a row whose file is present is checked for CRS, pixel size, axis alignment and a sane time axis.
