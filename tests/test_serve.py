import numpy as np
import pytest
import xarray as xr

from soberana_toolkit.serve.catalog import (
    discover_dataset_variables,
    discover_datasets,
    discover_products,
    find_scene_count_product,
    find_valid_count_product,
    matching_count_slice,
    single_count_value,
    spatial_variables,
)
from soberana_toolkit.serve.statistics import (
    _histogram_edges,
    fractional_class_statistics,
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


def test_discover_datasets_groups_products_by_parent_directory(tmp_path):
    root = tmp_path / "derived"
    (root / "s2_l2a_monthly").mkdir(parents=True)
    (root / "dem").mkdir()
    (root / "s2_l2a_monthly" / "BSI.nc").touch()
    (root / "s2_l2a_monthly" / "NDVI.nc").touch()
    (root / "dem" / "elevation.nc").touch()

    assert [path.name for path in discover_datasets(root)] == ["dem", "s2_l2a_monthly"]


def test_discover_dataset_variables_maps_names_to_files(tmp_path):
    dataset_dir = tmp_path / "s2_l2a_monthly"
    dataset_dir.mkdir()
    for name in ("BSI", "NDVI"):
        xr.Dataset(
            {name: (("y", "x"), np.ones((2, 2)))},
            coords={"y": [2, 1], "x": [1, 2]},
        ).to_netcdf(dataset_dir / f"{name}.nc")

    assert discover_dataset_variables(dataset_dir) == {
        "BSI": dataset_dir / "BSI.nc",
        "NDVI": dataset_dir / "NDVI.nc",
    }


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


def test_find_scene_count_product_in_same_directory(tmp_path):
    product = tmp_path / "NDVI.nc"
    count_product = tmp_path / "n_images.nc"
    product.touch()
    count_product.touch()

    assert find_scene_count_product(product) == count_product
    assert find_scene_count_product(count_product) is None


def test_matching_count_slice_uses_the_selected_time():
    times = np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")
    dataset = xr.Dataset(
        {"n_images": (("t", "y", "x"), np.array([[[1, 2]], [[3, 4]]]))},
        coords={"t": times, "y": [1], "x": [1, 2]},
    )

    selected = matching_count_slice(dataset, "n_images", times[1])

    assert selected is not None
    np.testing.assert_array_equal(selected.values, [[3, 4]])
    assert matching_count_slice(dataset, "n_images", np.datetime64("2024-03-01")) is None


def test_single_count_value_requires_spatially_constant_counts():
    assert single_count_value(np.array([[4, 4], [4, np.nan]])) == 4.0
    assert single_count_value(np.array([np.nan])) is None
    with pytest.raises(ValueError, match="varies across pixels"):
        single_count_value(np.array([[2, 4]]))


def test_fractional_class_statistics_use_soft_membership_weights():
    times = np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")
    coords = {"t": times, "y": [1], "x": [1, 2]}
    values = xr.DataArray(
        [[[0.0, 10.0]], [[2.0, 8.0]]],
        dims=("t", "y", "x"),
        coords=coords,
        name="NDVI",
    )
    fractions = {
        "scl_water": xr.DataArray(
            [[[0.25, 0.75]], [[0.5, 0.5]]],
            dims=("t", "y", "x"),
            coords=coords,
            name="scl_water",
        ),
        "scl_soil": xr.DataArray(
            [[[0.75, 0.25]], [[0.5, 0.5]]],
            dims=("t", "y", "x"),
            coords=coords,
            name="scl_soil",
        ),
        "scl_vegetation": xr.DataArray(
            np.zeros((2, 1, 2)),
            dims=("t", "y", "x"),
            coords=coords,
            name="scl_vegetation",
        ),
    }

    stats = fractional_class_statistics(values, fractions)

    assert stats["classes"]["scl_water"]["medians"] == [7.5, 5.0]
    assert stats["classes"]["scl_soil"]["medians"] == [2.5, 5.0]
    assert stats["classes"]["scl_vegetation"]["medians"] == [None, None]
    assert sum(stats["classes"]["scl_water"]["histograms"][0]["probability"]) == pytest.approx(1)

    selected_month = fractional_class_statistics(values, fractions, time_index=1)
    assert selected_month["times"] == ["2024-02-01"]
    assert selected_month["classes"]["scl_water"]["medians"] == [5.0]
    assert selected_month["classes"]["scl_water"]["histograms"][0] is not None

    time_series = fractional_class_statistics(
        values, fractions, include_histograms=False
    )
    assert time_series["classes"]["scl_water"]["medians"] == [7.5, 5.0]
    assert time_series["classes"]["scl_water"]["histograms"] == [None, None]


def test_fractional_histogram_groups_outliers_and_median_resists_them():
    times = np.array(["2024-01-01"], dtype="datetime64[ns]")
    coords = {"t": times, "y": [1], "x": np.arange(101)}
    data = np.append(np.arange(100, dtype=float), 1e9).reshape(1, 1, 101)
    values = xr.DataArray(data, dims=("t", "y", "x"), coords=coords, name="BSI")
    fractions = xr.DataArray(
        np.ones_like(data),
        dims=("t", "y", "x"),
        coords=coords,
        name="scl_water",
    )

    stats = fractional_class_statistics(values, {"scl_water": fractions})
    water = stats["classes"]["scl_water"]
    histogram = water["histograms"][0]

    assert water["medians"] == [50]
    assert histogram["lower_tail_fraction"] == pytest.approx(1 / 101)
    assert histogram["upper_tail_fraction"] == pytest.approx(1 / 101)
    assert histogram["probability"][0] > 0
    assert histogram["probability"][-1] > 0
    assert sum(histogram["probability"]) == pytest.approx(1)


def test_fractional_class_statistics_reject_values_outside_zero_to_one():
    times = np.array(["2024-01-01"], dtype="datetime64[ns]")
    values = xr.DataArray(
        [[[1.0]]], dims=("t", "y", "x"), coords={"t": times, "y": [1], "x": [1]}
    )
    fraction = xr.DataArray(
        [[[1.5]]], dims=("t", "y", "x"), coords={"t": times, "y": [1], "x": [1]}
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        fractional_class_statistics(values, {"scl_water": fraction})


def test_histogram_bins_are_capped_before_allocation_for_outlier_heavy_values():
    edges = _histogram_edges(np.array([0.0, 1e-9, 2e-9, 3e-9, 1e9]))

    assert edges.size <= 51
    assert edges[0] == 0
    assert edges[-1] == 1e9
