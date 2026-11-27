import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine

from soberana_toolkit.grid import load_grid
from soberana_toolkit.verify import check_dataset, check_dataset_config, main, verify_file

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


def create_dataset_config(tmp_path, grid, output_dir="data/derived/s2"):
    grid_dir = tmp_path / "config" / "grids"
    grid_dir.mkdir(parents=True)
    (grid_dir / "utm28_30m.yaml").write_text(grid.path.read_text())
    config_dir = tmp_path / "config" / "datasets"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "s2.yaml"
    config_path.write_text(
        "name: s2\n"
        "grid: utm28_30m\n"
        "source:\n"
        "  native_resolution: 10\n"
        "output:\n"
        f"  dir: {output_dir}\n"
    )
    return config_path


def test_dataset_config_checks_each_netcdf(tmp_path, grid, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = create_dataset_config(tmp_path, grid)
    output_dir = tmp_path / "data" / "derived" / "s2"
    output_dir.mkdir(parents=True)
    write_cube(output_dir / "NDVI.nc")
    write_cube(output_dir / "SI1.nc", res=20)

    findings = check_dataset_config(config)

    assert any(f.check == "pixel_size" and "SI1.nc" in f.message for f in levels(findings))
    assert not any("NDVI.nc" in f.message for f in levels(findings))


@pytest.mark.parametrize(
    ("config_text", "missing_field"),
    [
        ("name: incomplete\noutput: {}\n", "grid"),
        ("name: incomplete\ngrid: utm28_30m\noutput: {}\n", "source.native_resolution"),
        (
            (
                "name: incomplete\ngrid: utm28_30m\n"
                "source:\n  native_resolution: 10\noutput: {}\n"
            ),
            "output.dir",
        ),
    ],
)
def test_dataset_config_requires_verification_fields(
    tmp_path, grid, monkeypatch, config_text, missing_field
):
    monkeypatch.chdir(tmp_path)
    config = create_dataset_config(tmp_path, grid)
    config.write_text(config_text)

    findings = check_dataset_config(config)

    assert any(missing_field in f.message for f in findings)


def test_cli_scans_dataset_configs_by_default(tmp_path, grid, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = create_dataset_config(tmp_path, grid)
    (tmp_path / "data" / "derived" / "s2").mkdir(parents=True)

    assert main([]) == 0
    assert "no NetCDF files found" in capsys.readouterr().out

    assert main(["--dataset-config", str(config)]) == 0
    assert "no NetCDF files found" in capsys.readouterr().out
