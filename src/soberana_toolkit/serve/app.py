"""Streamlit browser for NetCDF products under ``data/derived``."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import rioxarray  # noqa: F401
import streamlit as st
import xarray as xr

from soberana_toolkit.serve.catalog import (
    discover_dataset_variables,
    discover_datasets,
    find_scene_count_product,
    find_valid_count_product,
    matching_count_slice,
    single_count_value,
    spatial_coordinates,
    spatial_variables,
)
from soberana_toolkit.serve.statistics import (
    SCL_CLASS_STYLES,
    fractional_class_statistics,
)


@st.cache_data(show_spinner="Calculating SCL-class statistics...")
def _cached_scl_class_statistics(
    product_path: str,
    variable: str,
    product_signature: tuple[int, int],
    class_signatures: tuple[tuple[str, str, int, int], ...],
    time_index: int | None,
    include_histograms: bool,
) -> dict:
    with ExitStack() as stack:
        product = stack.enter_context(xr.open_dataset(product_path, decode_coords="all"))
        class_data = {}
        for class_name, class_path, _, _ in class_signatures:
            class_dataset = stack.enter_context(
                xr.open_dataset(class_path, decode_coords="all")
            )
            if class_name not in class_dataset.data_vars:
                raise ValueError(f"{class_path}: missing {class_name!r} data variable")
            class_data[class_name] = class_dataset[class_name]
        if variable not in product.data_vars:
            raise ValueError(f"{product_path}: missing {variable!r} data variable")
        return fractional_class_statistics(
            product[variable],
            class_data,
            time_index=time_index,
            include_histograms=include_histograms,
        )

st.set_page_config(page_title="Derived data browser", layout="wide")
st.title("Derived data browser")

data_root = Path("data/derived")
if not data_root.is_dir():
    st.error(f"Derived-data directory not found: {data_root}. Run this app from the repository root.")
    st.stop()

datasets = discover_datasets(data_root)
if not datasets:
    st.info(f"No NetCDF products found under {data_root}.")
    st.stop()

selected_dataset = st.sidebar.selectbox(
    "Product",
    datasets,
    format_func=lambda path: path.name,
)
variable_files = discover_dataset_variables(selected_dataset)
if not variable_files:
    st.error(f"No spatial variables found in {selected_dataset}.")
    st.stop()
variables = sorted(variable_files)
variable = st.sidebar.selectbox("Variable", variables)
selected_path = variable_files[variable]
class_names = tuple(SCL_CLASS_STYLES)
class_files = {
    name: variable_files[name]
    for name in class_names
    if name in variable_files
}
missing_class_names = sorted(set(class_names) - class_files.keys())

with xr.open_dataset(selected_path, decode_coords="all") as dataset:
    if variable not in spatial_variables(dataset):
        st.error(f"{selected_path}: selected variable {variable!r} is not a spatial raster.")
        st.stop()

    data = dataset[variable]
    axes = spatial_coordinates(data)
    if axes is None:
        st.error(f"Could not identify spatial coordinates for {variable}.")
        st.stop()
    x_name, y_name = axes

    time_dim = next((name for name in ("t", "time") if name in data.dims), None)
    time_label = "Static raster"
    selected_time = None
    if time_dim is not None:
        time_values = data[time_dim].values
        if len(time_values) == 0:
            st.error(f"{variable} has an empty {time_dim} dimension.")
            st.stop()
        time_labels = [str(value)[:10] for value in time_values]
        if len(time_values) == 1:
            time_index = 0
            st.sidebar.caption(f"Time: {time_labels[time_index]}")
        else:
            time_index = st.sidebar.select_slider(
                "Time",
                options=list(range(len(time_values))),
                format_func=lambda index: time_labels[index],
            )
        selected_time = time_values[time_index]
        data = data.isel({time_dim: time_index})
        time_label = time_labels[time_index]

    count_path = find_valid_count_product(selected_path)
    scene_count_path = find_scene_count_product(selected_path)
    valid_count_data = None
    scene_count = None
    if count_path is not None and selected_time is not None:
        with xr.open_dataset(count_path, decode_coords="all") as count_dataset:
            valid_count_data = matching_count_slice(
                count_dataset, "n_valid", selected_time
            )
    if scene_count_path is not None and selected_time is not None:
        with xr.open_dataset(scene_count_path, decode_coords="all") as count_dataset:
            scene_count_data = matching_count_slice(
                count_dataset, "n_images", selected_time
            )
            if scene_count_data is not None:
                scene_count = single_count_value(scene_count_data.values)
    elif selected_path.name == "n_images.nc" and selected_time is not None:
        scene_count = single_count_value(data.values)

    extra_dims = [dim for dim in data.dims if dim not in (x_name, y_name)]
    for dim in extra_dims:
        if data.sizes[dim] != 1:
            st.error(f"Cannot display {variable}: unsupported non-spatial dimension {dim!r}.")
            st.stop()
        data = data.isel({dim: 0}, drop=True)

    data = data.transpose(y_name, x_name).load()
    values = np.asarray(data.values)
    x_values = data[x_name].values
    y_values = data[y_name].values
    description = data.attrs.get("description") or dataset.attrs.get("description")
    units = data.attrs.get("units")
    crs = dataset.rio.crs

class_stats = None
class_histograms = None
if time_dim is not None and class_files:
    product_stat = selected_path.stat()
    class_signatures = tuple(
        (
            name,
            str(path),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for name, path in class_files.items()
    )
    statistics_args = (
        str(selected_path),
        variable,
        (product_stat.st_mtime_ns, product_stat.st_size),
        class_signatures,
    )
    class_stats = _cached_scl_class_statistics(
        *statistics_args,
        time_index=None,
        include_histograms=False,
    )
    class_histograms = _cached_scl_class_statistics(
        *statistics_args,
        time_index=time_index,
        include_histograms=True,
    )

st.subheader(variable)
st.caption(f"{selected_dataset.name} · {time_label}")
if description:
    with st.expander("Description"):
        st.text(description)
if scene_count_path is not None or selected_path.name == "n_images.nc":
    if scene_count is None:
        st.warning(f"No matching monthly source-image count found for {time_label}.")
    else:
        st.info(f"Source images selected for {time_label}: {scene_count:g}.")

finite_values = values[np.isfinite(values)]
if finite_values.size == 0:
    st.warning("This raster slice contains no finite values.")
    st.stop()

lower, upper = np.percentile(finite_values, (2, 98))
if lower == upper:
    lower, upper = float(finite_values.min()), float(finite_values.max())
    if lower == upper:
        lower -= 0.5
        upper += 0.5

figure = go.Figure(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=values,
        zmin=float(lower),
        zmax=float(upper),
        colorscale="Viridis",
        colorbar={"title": units or variable},
        hovertemplate=f"{x_name}=%{{x}}<br>{y_name}=%{{y}}<br>{variable}=%{{z}}<extra></extra>",
    )
)
figure.update_layout(
    title=f"{variable} — {time_label}",
    xaxis_title=x_name,
    yaxis_title=y_name,
    margin={"l": 20, "r": 20, "t": 50, "b": 20},
)
figure.update_yaxes(scaleanchor="x", scaleratio=1)
st.plotly_chart(figure, width="stretch")

if time_dim is not None:
    st.subheader(f"{variable} by SCL class")
    st.caption(
        "Pixels contribute fractionally according to each monthly SCL fraction—no hard "
        "class threshold. Histograms are normalized within each class; values beyond the "
        "1st and 99th percentiles are included in the outer bins. Time-series lines show "
        "fraction-weighted spatial medians, which are less sensitive to isolated outliers. "
        "Legend support is in fractional pixels."
    )
    if missing_class_names:
        st.warning(
            "Missing SCL fraction rasters: "
            + ", ".join(SCL_CLASS_STYLES[name][0] for name in missing_class_names)
        )
    if class_stats is None:
        st.info("Add SCL class-fraction rasters to this dataset to see class comparisons.")
    else:
        histogram_column, series_column = st.columns(2)
        with histogram_column:
            histogram = go.Figure()
            histogram_trace_count = 0
            for class_name, stats in class_stats["classes"].items():
                monthly_histogram = class_histograms["classes"][class_name]["histograms"][0]
                if monthly_histogram is None:
                    continue
                edges = np.asarray(monthly_histogram["edges"])
                histogram.add_trace(
                    go.Bar(
                        x=(edges[:-1] + edges[1:]) / 2,
                        y=monthly_histogram["probability"],
                        width=np.diff(edges),
                        name=(
                            f"{stats['label']} "
                            f"(support {monthly_histogram['support']:.1f} px)"
                        ),
                        marker_color=stats["color"],
                        opacity=0.55,
                        hovertemplate=(
                            f"{variable}=%{{x}}<br>weighted probability=%{{y:.3f}}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
                histogram_trace_count += 1
            if histogram_trace_count:
                histogram.update_layout(
                    title=f"Distribution · {time_label}",
                    xaxis_title=f"{variable}{f' ({units})' if units else ''}",
                    yaxis_title="Weighted probability",
                    barmode="overlay",
                    bargap=0,
                    margin={"l": 20, "r": 20, "t": 50, "b": 20},
                )
                st.plotly_chart(histogram, width="stretch")
            else:
                st.info(f"No class-weighted values are available for {time_label}.")

        with series_column:
            series = go.Figure()
            for stats in class_stats["classes"].values():
                series.add_trace(
                    go.Scatter(
                        x=class_stats["times"],
                        y=stats["medians"],
                        mode="lines+markers",
                        name=stats["label"],
                        line={"color": stats["color"]},
                        customdata=stats["support"],
                        hovertemplate=(
                            "%{x}<br>fraction-weighted median=%{y:.4g}"
                            "<br>fractional pixel support=%{customdata:.1f}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
            series.update_layout(
                title="Time series · fraction-weighted median",
                xaxis_title="Month",
                yaxis_title=f"{variable}{f' ({units})' if units else ''}",
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
            )
            series.update_xaxes(type="date")
            st.plotly_chart(series, width="stretch")

if valid_count_data is not None:
    count_axes = spatial_coordinates(valid_count_data)
    if count_axes is None:
        st.warning("The valid-observation count has no identifiable spatial coordinates.")
    else:
        count_x, count_y = count_axes
        count_data = valid_count_data.transpose(count_y, count_x)
        count_values = np.asarray(count_data.values)
        if np.any(np.isfinite(count_values)):
            count_figure = go.Figure(
                go.Heatmap(
                    x=count_data[count_x].values,
                    y=count_data[count_y].values,
                    z=count_values,
                    colorscale="Viridis",
                    colorbar={"title": "Valid images"},
                    hovertemplate=(
                        f"{count_x}=%{{x}}<br>{count_y}=%{{y}}"
                        "<br>valid images=%{z}<extra></extra>"
                    ),
                )
            )
            count_figure.update_layout(
                title=f"Valid input images per pixel — {time_label}",
                xaxis_title=count_x,
                yaxis_title=count_y,
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
            )
            count_figure.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(count_figure, width="stretch")
        else:
            st.warning(f"No valid-observation counts are available for {time_label}.")

with st.expander("Metadata"):
    st.text(f"Dataset: {selected_dataset.name}")
    st.text(f"Dimensions: {', '.join(data.dims)}")
    st.text(f"CRS: {crs or 'not recorded'}")
    st.text(f"Units: {units or 'not recorded'}")
