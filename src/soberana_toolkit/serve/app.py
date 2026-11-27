"""Streamlit browser for NetCDF products under ``data/derived``."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import rioxarray  # noqa: F401
import streamlit as st
import xarray as xr

from soberana_toolkit.serve.catalog import (
    discover_products,
    find_scene_count_product,
    find_valid_count_product,
    matching_count_slice,
    single_count_value,
    spatial_coordinates,
    spatial_variables,
)

st.set_page_config(page_title="Derived data browser", layout="wide")
st.title("Derived data browser")

data_root = Path("data/derived")
if not data_root.is_dir():
    st.error(f"Derived-data directory not found: {data_root}. Run this app from the repository root.")
    st.stop()

products = discover_products(data_root)
if not products:
    st.info(f"No NetCDF products found under {data_root}.")
    st.stop()

selected_path = st.sidebar.selectbox(
    "Product",
    products,
    format_func=lambda path: str(path.relative_to(data_root)),
)

with xr.open_dataset(selected_path, decode_coords="all") as dataset:
    variables = spatial_variables(dataset)
    if not variables:
        st.error("This file has no raster variables with one-dimensional x/y coordinates.")
        st.stop()

    variable = st.sidebar.selectbox("Variable", variables)
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

st.subheader(variable)
st.caption(f"{selected_path.relative_to(data_root)} · {time_label}")
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
    st.text(f"File: {selected_path}")
    st.text(f"Dimensions: {', '.join(data.dims)}")
    st.text(f"CRS: {crs or 'not recorded'}")
    st.text(f"Units: {units or 'not recorded'}")
