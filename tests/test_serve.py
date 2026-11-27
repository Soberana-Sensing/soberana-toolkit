import numpy as np
import xarray as xr

from soberana_toolkit.serve.catalog import discover_products, spatial_variables


def test_discover_products_recursively_and_in_order(tmp_path):
    root = tmp_path / "derived"
    (root / "nested").mkdir(parents=True)
    (root / "z.nc").touch()
    (root / "nested" / "a.nc").touch()
    (root / "nested" / "ignore.json").touch()

    products = discover_products(root)

    assert [path.relative_to(root).as_posix() for path in products] == [
        "nested/a.nc",
        "z.nc",
    ]


def test_spatial_variables_include_static_and_time_series_rasters():
    dataset = xr.Dataset(
        {
            "ndvi": (("t", "y", "x"), np.ones((2, 2, 3))),
            "dem": (("y", "x"), np.ones((2, 3))),
            "metadata": (("band",), np.ones(1)),
            "crs": ((), 0),
        },
        coords={
            "t": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"),
            "y": [2, 1],
            "x": [1, 2, 3],
        },
    )

    assert spatial_variables(dataset) == ["ndvi", "dem"]
