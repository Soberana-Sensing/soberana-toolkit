import pytest

from soberana_toolkit.grid import load_grid, parse_crs


def test_load_grid_reads_epsg_as_int(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(
        "name: g\ncrs: EPSG:32628\nresolution: 30\n"
        "resolution_policy: native_or_grid\nalign: upper-left\n"
    )
    grid = load_grid(path)
    assert grid.crs == 32628
    assert grid.crs_label == "EPSG:32628"
    assert grid.resolution == 30
    assert grid.expected_pixel_size(None) == 30


def test_expected_pixel_size_never_shrinks(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(
        "name: g\ncrs: EPSG:32628\nresolution: 30\n"
        "resolution_policy: native_or_grid\nalign: upper-left\n"
    )
    grid = load_grid(path)
    assert grid.expected_pixel_size(10) == 30   # finer source is coarsened to the grid
    assert grid.expected_pixel_size(20) == 30
    assert grid.expected_pixel_size(100) == 100  # coarser source keeps native resolution


def test_locked_grid_file_hashes(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(
        "name: g\ncrs: EPSG:32628\nresolution: 30\n"
        "resolution_policy: native_or_grid\nalign: upper-left\n"
    )
    assert len(load_grid(path).sha256()) == 64


@pytest.mark.parametrize("crs,expected", [("EPSG:4326", 4326), (32628, 32628), ("PROJCS[...]", "PROJCS[...]")])
def test_parse_crs(crs, expected):
    assert parse_crs(crs) == expected


def test_bad_grid_rejected(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("name: g\ncrs: EPSG:32628\nresolution: 0\n")
    with pytest.raises(ValueError):
        load_grid(path)


def test_unknown_grid_name_raises():
    with pytest.raises(FileNotFoundError):
        load_grid("no_such_grid")
