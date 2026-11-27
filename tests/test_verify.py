import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine

from soberana_toolkit.grid import load_grid
from soberana_toolkit.verify import check_dataset, check_entry, verify_file

CRS = "EPSG:32628"


@pytest.fixture
def grid(tmp_path):
    path = tmp_path / "utm28_30m.yaml"
    path.write_text(
        "name: utm28_30m\ncrs: EPSG:32628\nresolution: 30\n"
        "resolution_policy: native_or_grid\nalign: upper-left\n"
    )
    return load_grid(path)


def write_cube(path, res=30, crs: str | None = CRS, nx=8, ny=8, times=None):
    """Write a small CF NetCDF. crs=None writes a file without a CRS."""
    x0, y0 = 500000, 1000000
    x = x0 + res / 2 + np.arange(nx) * res
    y = y0 - res / 2 - np.arange(ny) * res
    data = np.ones((ny, nx), dtype="float32")
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x}, name="ndvi")
    if crs is not None:
        da = da.rio.write_crs(crs)
        da = da.rio.write_transform(Affine(res, 0, x0, 0, -res, y0))
    if times is None:
        ds = da.to_dataset()
    else:
        ds = da.expand_dims(t=times).to_dataset()
    ds.to_netcdf(path)
    return path


def levels(findings, level="error"):
    return [f for f in findings if f.level == level]


def test_aligned_cube_passes(grid, tmp_path):
    path = write_cube(tmp_path / "ok.nc")
    findings = verify_file(path, grid, native_resolution=10)
    assert levels(findings) == []


def test_wrong_pixel_size_fails(grid, tmp_path):
    path = write_cube(tmp_path / "res20.nc", res=20)
    findings = verify_file(path, grid, native_resolution=10)
    assert any(f.check == "pixel_size" for f in levels(findings))


def test_upsampling_rejected(grid, tmp_path):
    path = write_cube(tmp_path / "fine.nc", res=15)
    findings = verify_file(path, grid, native_resolution=20)
    assert any(f.check == "resolution_policy" for f in levels(findings))


def test_wrong_crs_fails(grid, tmp_path):
    path = write_cube(tmp_path / "crs.nc", crs="EPSG:4326")
    findings = verify_file(path, grid, native_resolution=10)
    assert any(f.check == "crs" for f in levels(findings))


def test_missing_crs_fails(grid, tmp_path):
    path = write_cube(tmp_path / "nocrs.nc", crs=None)
    findings = verify_file(path, grid, native_resolution=10)
    assert any(f.check == "crs" for f in levels(findings))


def test_time_axis_checked(grid, tmp_path):
    good = write_cube(tmp_path / "t.nc", times=np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"))
    assert levels(verify_file(good, grid, 10)) == []

    bad = write_cube(tmp_path / "tbad.nc",
                     times=np.array(["2024-02-01", "2024-01-01"], dtype="datetime64[ns]"))
    assert any(f.check == "time" for f in levels(verify_file(bad, grid, 10)))


def test_missing_file_is_error(grid, tmp_path):
    findings = verify_file(tmp_path / "nope.nc", grid, 10)
    assert any(f.check == "file" for f in levels(findings))


def test_check_dataset_detects_non_uniform_axis(grid, tmp_path):
    path = write_cube(tmp_path / "irregular.nc")
    with xr.open_dataset(path, decode_coords="all") as ds:
        ds = ds.assign_coords(x=ds["x"].values + np.arange(ds.sizes["x"]) ** 2)
        findings = check_dataset(ds, grid, native_resolution=10)
    assert any(f.check == "axes" for f in levels(findings))


def test_inventory_declarations_checked(tmp_path, grid, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config" / "grids").mkdir(parents=True)
    (tmp_path / "config" / "grids" / "utm28_30m.yaml").write_text(grid.path.read_text())

    entry = {
        "name": "s2", "path": "data/derived/s2.nc", "role": "predictor",
        "source": "SENTINEL2_L2A", "grid": "utm28_30m", "crs": "EPSG:32628",
        "native_resolution": 10, "resolution": 30,
        "acquisition": "2016-01-01/2025-12-31", "retrieved": "2026-11-27",
    }
    findings = check_entry(entry)
    assert levels(findings) == []
    assert any(f.check == "file" for f in findings)  # warnings only when the file is absent

    bad = dict(entry, resolution=20)
    assert any(f.check == "inventory" for f in levels(check_entry(bad)))


def test_inventory_entry_requires_dates(tmp_path, grid, monkeypatch):
    monkeypatch.chdir(tmp_path)
    finding = check_entry({"name": "x", "path": "p", "grid": "utm28_30m",
                           "source": "s", "native_resolution": 10})
    assert any(f.check == "inventory" and "acquisition" in f.message for f in finding)
