import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from soberana_toolkit.openeo.s2_l2a_processing import split_netcdf_by_band


def test_split_netcdf_by_band_writes_only_requested_bands(tmp_path):
    source = tmp_path / "openEO.nc"
    out_dir = tmp_path / "outputs"
    data = np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    cube = xr.DataArray(
        data,
        dims=("t", "bands", "y", "x"),
        coords={
            "t": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"),
            "bands": ["NDVI", "SI1", "unused"],
            "y": [15, -15],
            "x": [15, 45],
        },
        name="openEO",
    )
    cube.attrs["grid_mapping"] = "crs"
    cube.to_dataset().to_netcdf(source)

    outputs = split_netcdf_by_band(
        source, ["NDVI", "SI1"], out_dir, 32628, {"NDVI": "vegetation index"}
    )

    assert [path.name for path in outputs] == ["NDVI.nc", "SI1.nc"]
    with xr.open_dataset(out_dir / "NDVI.nc", decode_coords="all") as ndvi:
        assert list(ndvi.data_vars) == ["NDVI"]
        assert "bands" not in ndvi.dims
        assert "crs" in ndvi.coords
        assert ndvi.attrs["description"] == "vegetation index"
        assert ndvi["NDVI"].attrs["description"] == "vegetation index"
        np.testing.assert_array_equal(ndvi["NDVI"].values, data[:, 0])
        assert str(ndvi["NDVI"].rio.crs) == "EPSG:32628"
    with xr.open_dataset(out_dir / "SI1.nc") as si1:
        assert "description" not in si1["SI1"].attrs
