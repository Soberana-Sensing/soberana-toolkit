import numpy as np
import xarray as xr

from soberana_toolkit.serve.catalog import (
    discover_products,
    find_valid_count_product,
    matching_valid_observation_slice,
    spatial_variables,
    summarize_valid_observations,
)


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


def test_find_valid_count_product_in_same_directory(tmp_path):
    product = tmp_path / "NDVI.nc"
    count_product = tmp_path / "n_valid.nc"
    product.touch()
    count_product.touch()

    assert find_valid_count_product(product) == count_product
    assert find_valid_count_product(count_product) is None


def test_summarize_valid_observations_ignores_nodata():
    summary = summarize_valid_observations(np.array([[2, 4], [6, np.nan]]))

    assert summary == (4.0, 2.0, 6.0)
    assert summarize_valid_observations(np.array([np.nan])) is None


def test_matching_valid_observation_slice_uses_the_selected_time():
    times = np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")
    dataset = xr.Dataset(
        {"n_valid": (("t", "y", "x"), np.array([[[1, 2]], [[3, 4]]]))},
        coords={"t": times, "y": [1], "x": [1, 2]},
    )

    selected = matching_valid_observation_slice(dataset, times[1])

    assert selected is not None
    np.testing.assert_array_equal(selected.values, [[3, 4]])
    assert matching_valid_observation_slice(
        dataset, np.datetime64("2024-03-01")
    ) is None
