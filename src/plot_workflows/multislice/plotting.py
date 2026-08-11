# Multislice plotting helpers.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable

from .data import (
    SliceDataset,
    TransformFn,
    inspect_slice_series,
    stack_field,
)


# =============================================================================
# 1. Slice-axis mapping
# =============================================================================

@dataclass
class SliceAxisMapping:
    """
    Mapping from physical slice values to plotting positions.

    original_values:
        Real values, for example tri_factor = [0.05, 0.10, 0.15].

    plot_values:
        Positions used in the 3D axes. These may be normalized.

    mode:
        'data', 'index', or 'normalized'.
    """

    original_values: np.ndarray
    plot_values: np.ndarray
    mode: str

    def tick_positions(self) -> np.ndarray:
        return self.plot_values

    def tick_labels(self, fmt: str = "{:.4g}") -> List[str]:
        labels: List[str] = []

        for value in self.original_values:
            try:
                labels.append(fmt.format(float(value)))
            except Exception:
                labels.append(str(value))

        return labels


def _as_float_array_or_none(values: Sequence[Any]) -> Optional[np.ndarray]:
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        return None


def make_slice_axis_mapping(
    slice_values: Sequence[Any],
    *,
    x_values: Optional[Sequence[float]] = None,
    y_values: Optional[Sequence[float]] = None,
    mode: str = "normalized",
    normalized_span_ratio: float = 1,
) -> SliceAxisMapping:
    """
    Convert real slice values to 3D plotting positions.

    mode='data':
        Use physical values directly.

    mode='index':
        Use 0, 1, 2, ...

    mode='normalized':
        Linearly map slice values to a display span comparable to the in-plane
        x/y ranges. This is the recommended default for 3D slice plots because
        raw z values can easily make the visual scale misleading.
    """

    if len(slice_values) == 0:
        raise ValueError("slice_values is empty.")

    original = np.asarray(slice_values, dtype=object)
    mode = mode.lower()

    if mode == "index":
        plot_values = np.arange(len(slice_values), dtype=float)
        return SliceAxisMapping(original, plot_values, mode)

    numeric_values = _as_float_array_or_none(slice_values)

    if numeric_values is None:
        if mode == "data":
            raise TypeError(
                "slice_position_mode='data' requires numeric slice values."
            )

        warnings.warn(
            "Non-numeric slice values detected. Falling back to index mode.",
            RuntimeWarning,
        )
        plot_values = np.arange(len(slice_values), dtype=float)
        return SliceAxisMapping(original, plot_values, "index")

    if mode == "data":
        return SliceAxisMapping(original, numeric_values, mode)

    if mode != "normalized":
        raise ValueError("slice_position_mode must be 'data', 'index', or 'normalized'.")

    x_span = _safe_span(x_values)
    y_span = _safe_span(y_values)
    base_span = max(x_span, y_span, 1.0)
    target_span = base_span * float(normalized_span_ratio)

    s_min = float(np.nanmin(numeric_values))
    s_max = float(np.nanmax(numeric_values))

    if np.isclose(s_min, s_max):
        plot_values = np.zeros_like(numeric_values, dtype=float)
    else:
        plot_values = (numeric_values - s_min) / (s_max - s_min) * target_span

    return SliceAxisMapping(original, plot_values, mode)


def _safe_span(values: Optional[Sequence[float]]) -> float:
    if values is None:
        return 1.0

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size <= 1:
        return 1.0

    span = float(np.nanmax(arr) - np.nanmin(arr))

    if span <= 0 or not np.isfinite(span):
        return 1.0

    return span


def _collect_global_xy_values(
    slice_datasets: Sequence[SliceDataset],
    *,
    x_key: str,
    y_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for ds in slice_datasets:
        xs.append(np.asarray(ds.get_coord(x_key), dtype=float))
        ys.append(np.asarray(ds.get_coord(y_key), dtype=float))

    return np.concatenate(xs), np.concatenate(ys)


# =============================================================================
# 2. Coordinate projection
# =============================================================================

def _validate_slice_direction(slice_direction: str) -> str:
    direction = slice_direction.lower()

    if direction not in {"x", "y", "z"}:
        raise ValueError("slice_direction must be 'x', 'y', or 'z'.")

    return direction


def _project_surface_to_3d(
    X: np.ndarray,
    Y: np.ndarray,
    S: np.ndarray,
    *,
    slice_direction: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert in-plane coordinates X/Y and slice coordinate S to 3D axes.

    slice_direction='z':
        3D axes = (X, Y, S)

    slice_direction='x':
        3D axes = (S, X, Y)

    slice_direction='y':
        3D axes = (X, S, Y)
    """

    direction = _validate_slice_direction(slice_direction)

    if direction == "z":
        return X, Y, S

    if direction == "x":
        return S, X, Y

    return X, S, Y


def _project_points_to_3d(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    *,
    slice_direction: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = _validate_slice_direction(slice_direction)

    if direction == "z":
        return x, y, s

    if direction == "x":
        return s, x, y

    return x, s, y


def _axis_labels_for_direction(
    *,
    x_key: str,
    y_key: str,
    slice_key: str,
    slice_direction: str,
    axis_labels: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str, str]:
    labels = dict(axis_labels or {})

    x_label = labels.get(x_key, x_key)
    y_label = labels.get(y_key, y_key)
    s_label = labels.get(slice_key, slice_key)

    direction = _validate_slice_direction(slice_direction)

    if direction == "z":
        return x_label, y_label, s_label

    if direction == "x":
        return s_label, x_label, y_label

    return x_label, s_label, y_label


def _set_slice_axis_ticks(
    ax: Any,
    *,
    slice_direction: str,
    mapping: SliceAxisMapping,
    max_ticks: int = 8,
    tick_label_format: str = "{:.4g}",
) -> None:
    direction = _validate_slice_direction(slice_direction)

    positions = mapping.tick_positions()
    labels = mapping.tick_labels(tick_label_format)

    if len(positions) > max_ticks:
        idx = np.linspace(0, len(positions) - 1, max_ticks).round().astype(int)
        positions = positions[idx]
        labels = [labels[i] for i in idx]

    if direction == "z":
        ax.set_zticks(positions)
        ax.set_zticklabels(labels)
    elif direction == "x":
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)


def _apply_box_aspect(
    ax: Any,
    *,
    x_values_3d: Sequence[float],
    y_values_3d: Sequence[float],
    z_values_3d: Sequence[float],
    box_aspect: Union[str, Tuple[float, float, float], None],
) -> None:
    """
    box_aspect='data':
        Use actual plotted coordinate spans.

    This avoids visually over-expanded z/slice axes.
    """

    if box_aspect is None:
        return

    if isinstance(box_aspect, str):
        if box_aspect != "data":
            raise ValueError("box_aspect must be None, 'data', or a 3-tuple.")

        spans = [
            _safe_span(x_values_3d),
            _safe_span(y_values_3d),
            _safe_span(z_values_3d),
        ]
        ax.set_box_aspect(spans)
        return

    ax.set_box_aspect(box_aspect)


def _finite_limits(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return -0.5, 0.5

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -0.5, 0.5

    return vmin, vmax


def _pad_limits(
    vmin: float,
    vmax: float,
    *,
    padding_ratio: float = 0.03,
) -> Tuple[float, float]:
    if np.isclose(vmin, vmax):
        base = max(abs(vmin), abs(vmax), 1.0)
        pad = base * padding_ratio
        return vmin - pad, vmax + pad

    span = vmax - vmin
    pad = span * padding_ratio
    return vmin - pad, vmax + pad


def _set_axis_limits_from_projected_data(
    ax: Any,
    *,
    x_values_3d: Sequence[float],
    y_values_3d: Sequence[float],
    z_values_3d: Sequence[float],
    padding_ratio: float = 0.03,
) -> None:
    """
    Force 3D axis limits from actual geometric coordinates.

    This is important for 3D contour projections. Matplotlib may otherwise
    include the contour field values in autoscaling, which can make the slice
    axis much too large.
    """

    xlim = _pad_limits(
        *_finite_limits(x_values_3d),
        padding_ratio=padding_ratio,
    )
    ylim = _pad_limits(
        *_finite_limits(y_values_3d),
        padding_ratio=padding_ratio,
    )
    zlim = _pad_limits(
        *_finite_limits(z_values_3d),
        padding_ratio=padding_ratio,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    try:
        ax.autoscale(False)
    except Exception:
        pass



# =============================================================================
# 3. Color utilities
# =============================================================================

def _make_color_values(
    raw_values: np.ndarray,
    color_transform: Optional[TransformFn],
) -> np.ndarray:
    raw_values = np.asarray(raw_values)

    if color_transform is not None:
        values = np.asarray(color_transform(raw_values))
    else:
        values = raw_values

    if np.iscomplexobj(values):
        values = np.abs(values)

    try:
        return values.astype(float)
    except Exception as exc:
        raise TypeError(
            "Color values cannot be converted to float. "
            "Please provide color_transform."
        ) from exc


def infer_color_norm(
    slice_datasets: Sequence[SliceDataset],
    *,
    field_key: str,
    x_key: str,
    y_key: str,
    color_transform: Optional[TransformFn] = None,
    norm: Optional[Normalize] = None,
    norm_type: str = "linear",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile_clip: Optional[Tuple[float, float]] = None,
    linthresh: float = 1e-3,
) -> Normalize:
    """
    Infer one shared color normalization over all slices.
    """

    if norm is not None:
        return norm

    report = inspect_slice_series(
        slice_datasets,
        x_key=x_key,
        y_key=y_key,
        field_keys=field_key,
        check_xy_values=False,
    )

    values_list: List[np.ndarray] = []

    if report.consistent_shape:
        field_stack = stack_field(
            slice_datasets,
            field_key=field_key,
            x_key=x_key,
            y_key=y_key,
        )
        values_list.append(
            _make_color_values(field_stack, color_transform).ravel()
        )
    else:
        for ds in slice_datasets:
            values_list.append(
                _make_color_values(ds.get_field(field_key), color_transform).ravel()
            )

    values = np.concatenate(values_list)
    values = values[np.isfinite(values)]

    norm_type = norm_type.lower()

    if norm_type == "log":
        values = values[values > 0]

    if values.size == 0:
        raise ValueError("No finite color values found.")

    if percentile_clip is not None:
        p_low, p_high = percentile_clip
        inferred_vmin = float(np.nanpercentile(values, p_low))
        inferred_vmax = float(np.nanpercentile(values, p_high))
    else:
        inferred_vmin = float(np.nanmin(values))
        inferred_vmax = float(np.nanmax(values))

    if vmin is None:
        vmin = inferred_vmin
    if vmax is None:
        vmax = inferred_vmax

    if vmin == vmax:
        delta = abs(vmin) * 1e-6 + 1e-12
        vmin -= delta
        vmax += delta

    if norm_type == "linear":
        return Normalize(vmin=vmin, vmax=vmax)

    if norm_type == "log":
        if vmin <= 0:
            raise ValueError(
                f"LogNorm requires vmin > 0, got vmin={vmin}."
            )
        return LogNorm(vmin=vmin, vmax=vmax)

    if norm_type == "symlog":
        return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

    raise ValueError("norm_type must be 'linear', 'log', or 'symlog'.")


# =============================================================================
# 4. Plot 1: extrema trajectory
# =============================================================================

def plot_extrema_trajectory_3d(
    trajectory_df: pd.DataFrame,
    *,
    x_key: str,
    y_key: str,
    slice_key: str,
    slice_direction: str = "z",
    slice_position_mode: str = "normalized",
    normalized_slice_span_ratio: float = 0.75,
    color_key: Optional[str] = "score_value",
    cmap: str = "viridis",
    norm: Optional[Normalize] = None,
    connect: bool = True,
    scatter: bool = True,
    sort_by_slice: bool = True,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (4.0, 3.5),
    axis_labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    line_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    add_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    view: Optional[Tuple[float, float]] = None,
    box_aspect: Union[str, Tuple[float, float, float], None] = "data",
    slice_tick_label_format: str = "{:.4g}",
) -> Tuple[Any, Any, Optional[Any], SliceAxisMapping]:
    """
    Plot extrema trajectory as a 3D curve.

    slice_direction:
        Direction of the slice axis: 'x', 'y', or 'z'.

    slice_position_mode:
        'normalized' is recommended for visualization.
        'data' uses true slice values directly.
        'index' uses 0, 1, 2, ...
    """

    if trajectory_df.empty:
        raise ValueError("trajectory_df is empty.")

    for key in [x_key, y_key, slice_key]:
        if key not in trajectory_df.columns:
            raise KeyError(f"{key!r} not found in trajectory_df.")

    df = trajectory_df.copy()

    if "valid" in df.columns:
        df = df[df["valid"].astype(bool)]

    if df.empty:
        raise ValueError("trajectory_df has no valid points.")

    if sort_by_slice:
        df = df.sort_values(by=slice_key)

    x = np.asarray(df[x_key], dtype=float)
    y = np.asarray(df[y_key], dtype=float)
    slice_values = list(df[slice_key])

    mapping = make_slice_axis_mapping(
        slice_values,
        x_values=x,
        y_values=y,
        mode=slice_position_mode,
        normalized_span_ratio=normalized_slice_span_ratio,
    )

    s_plot = mapping.plot_values

    X3, Y3, Z3 = _project_points_to_3d(
        x,
        y,
        s_plot,
        slice_direction=slice_direction,
    )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    line_kwargs_final: Dict[str, Any] = {
        "linewidth": 1.5,
        "color": "k",
        "alpha": 0.9,
    }
    if line_kwargs:
        line_kwargs_final.update(line_kwargs)

    scatter_kwargs_final: Dict[str, Any] = {
        "s": 18,
        "depthshade": False,
    }
    if scatter_kwargs:
        scatter_kwargs_final.update(scatter_kwargs)

    artist = None

    if connect:
        ax.plot(X3, Y3, Z3, **line_kwargs_final)

    if scatter:
        if color_key is not None and color_key in df.columns:
            c = np.asarray(df[color_key])

            if np.iscomplexobj(c):
                c = np.abs(c)

            c = c.astype(float)

            artist = ax.scatter(
                X3,
                Y3,
                Z3,
                c=c,
                cmap=cmap,
                norm=norm,
                **scatter_kwargs_final,
            )

            if add_colorbar:
                cbar = fig.colorbar(artist, ax=ax, shrink=0.65, pad=0.08)
                cbar.set_label(colorbar_label or color_key)
        else:
            scatter_kwargs_final.setdefault("color", "k")
            artist = ax.scatter(X3, Y3, Z3, **scatter_kwargs_final)

    xlabel, ylabel, zlabel = _axis_labels_for_direction(
        x_key=x_key,
        y_key=y_key,
        slice_key=slice_key,
        slice_direction=slice_direction,
        axis_labels=axis_labels,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    _set_slice_axis_ticks(
        ax,
        slice_direction=slice_direction,
        mapping=mapping,
        tick_label_format=slice_tick_label_format,
    )

    if title:
        ax.set_title(title)

    if view is not None:
        elev, azim = view
        ax.view_init(elev=elev, azim=azim)

    _apply_box_aspect(
        ax,
        x_values_3d=X3,
        y_values_3d=Y3,
        z_values_3d=Z3,
        box_aspect=box_aspect,
    )

    return fig, ax, artist, mapping


# =============================================================================
# 5. Plot 2: 3D heatmap slices
# =============================================================================

def _make_mesh(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(x_values, y_values, indexing="ij")

def _valid_contour_levels(
    color_values: np.ndarray,
    contour_levels: Sequence[float],
) -> List[float]:
    values = np.asarray(color_values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return []

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    levels = []
    for level in contour_levels:
        try:
            level_float = float(level)
        except Exception:
            continue

        if vmin <= level_float <= vmax:
            levels.append(level_float)

    return levels


def _convert_contour_kwargs_to_line_kwargs(
    contour_kwargs: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Convert matplotlib contour-style kwargs to ax.plot line kwargs.

    contour uses:
        colors, linewidths

    ax.plot uses:
        color, linewidth
    """

    line_kwargs: Dict[str, Any] = {
        "color": "k",
        "linewidth": 0.45,
        "alpha": 0.85,
        "zorder": 20,
    }

    if contour_kwargs is None:
        return line_kwargs

    for key, value in contour_kwargs.items():
        if key == "colors":
            if isinstance(value, str):
                line_kwargs["color"] = value
            else:
                try:
                    line_kwargs["color"] = list(value)[0]
                except Exception:
                    line_kwargs["color"] = value

        elif key == "linewidths":
            if np.isscalar(value):
                line_kwargs["linewidth"] = value
            else:
                try:
                    line_kwargs["linewidth"] = list(value)[0]
                except Exception:
                    line_kwargs["linewidth"] = value

        elif key in {"levels", "zdir", "offset"}:
            continue

        else:
            line_kwargs[key] = value

    return line_kwargs


def _extract_contour_segments_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    color_values: np.ndarray,
    *,
    contour_levels: Sequence[float],
) -> List[Tuple[float, np.ndarray]]:
    """
    Compute 2D contour line segments.

    Returns
    -------
    segments:
        List of (level, segment) pairs.

        Each segment has shape (n_points, 2), where columns are:

            segment[:, 0] -> x coordinate
            segment[:, 1] -> y coordinate
    """

    color_values = np.asarray(color_values, dtype=float)

    valid_levels = _valid_contour_levels(
        color_values,
        contour_levels,
    )

    if len(valid_levels) == 0:
        return []

    X, Y = np.meshgrid(x_values, y_values, indexing="ij")

    # Use a temporary invisible 2D figure only to compute contour geometry.
    # We do not draw this figure to the final output.
    fig_tmp = plt.figure(figsize=(1, 1))
    ax_tmp = fig_tmp.add_subplot(111)

    try:
        cs = ax_tmp.contour(
            X,
            Y,
            color_values,
            levels=valid_levels,
        )

        segments: List[Tuple[float, np.ndarray]] = []

        for level, level_segments in zip(cs.levels, cs.allsegs):
            for seg in level_segments:
                seg = np.asarray(seg)

                if seg.ndim != 2 or seg.shape[0] < 2 or seg.shape[1] != 2:
                    continue

                segments.append((float(level), seg.copy()))

    finally:
        plt.close(fig_tmp)

    return segments


def _plot_contours_on_slice_plane(
    ax: Any,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color_values: np.ndarray,
    slice_position: float,
    slice_direction: str,
    contour_levels: Sequence[float],
    contour_kwargs: Optional[Mapping[str, Any]] = None,
    contour_offset: float = 0.0,
) -> List[Any]:
    """
    Draw contour lines manually on a 3D slice plane.

    This avoids using ax.contour(..., zdir=..., offset=...), which can be
    unstable when combined with plot_surface.
    """

    segments = _extract_contour_segments_2d(
        x_values,
        y_values,
        color_values,
        contour_levels=contour_levels,
    )

    line_kwargs = _convert_contour_kwargs_to_line_kwargs(contour_kwargs)

    artists: List[Any] = []

    s_value = float(slice_position) + float(contour_offset)

    for _, seg in segments:
        x_line = seg[:, 0]
        y_line = seg[:, 1]
        s_line = np.full_like(x_line, s_value, dtype=float)

        X3, Y3, Z3 = _project_points_to_3d(
            x_line,
            y_line,
            s_line,
            slice_direction=slice_direction,
        )

        artist = ax.plot(
            X3,
            Y3,
            Z3,
            **line_kwargs,
        )

        artists.extend(artist)

    return artists


def plot_heatmap_slices_3d(
    slice_datasets: Sequence[SliceDataset],
    *,
    x_key: str,
    y_key: str,
    slice_key: str,
    field_key: str,
    slice_direction: str = "z",
    slice_position_mode: str = "normalized",
    normalized_slice_span_ratio: float = 0.75,
    cmap: str = "RdBu",
    norm: Optional[Normalize] = None,
    norm_type: str = "linear",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile_clip: Optional[Tuple[float, float]] = None,
    color_transform: Optional[TransformFn] = None,
    alpha: float = 0.95,
    stride: int = 1,
    shade: bool = False,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    antialiased: bool = False,
    sort_slices: bool = True,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (4.5, 3.8),
    add_colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    contour_levels: Optional[Sequence[float]] = None,
    contour_kwargs: Optional[Mapping[str, Any]] = None,
    contour_offset: Optional[float] = None,
    contour_offset_ratio: float = 1e-4,
    warn_empty_contours: bool = False,
    axis_labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    view: Optional[Tuple[float, float]] = (25, -55),
    box_aspect: Union[str, Tuple[float, float, float], None] = "data",
    vectorized: Union[str, bool] = "auto",
    slice_tick_label_format: str = "{:.4g}",
    axis_limit_padding_ratio: float = 0.03,
) -> Tuple[Any, Any, List[Any], ScalarMappable, SliceAxisMapping]:
    """
    Plot multiple 2D heat maps as slices in a 3D coordinate system.

    slice_direction:
        'z':
            3D axes = x_key, y_key, slice_key

        'x':
            3D axes = slice_key, x_key, y_key

        'y':
            3D axes = x_key, slice_key, y_key

    slice_position_mode:
        'normalized':
            recommended default. Prevents the slice axis from becoming visually
            too large or too small.

        'data':
            use the true slice values as the plotted coordinates.

        'index':
            use 0, 1, 2, ...
    """

    if len(slice_datasets) == 0:
        raise ValueError("slice_datasets is empty.")

    datasets = sorted(slice_datasets, key=lambda ds: float(ds.slice_value)) if sort_slices else list(slice_datasets)

    report = inspect_slice_series(
        datasets,
        x_key=x_key,
        y_key=y_key,
        field_keys=field_key,
        check_xy_values=False,
    )

    if vectorized is True and not report.consistent_shape:
        raise ValueError(
            "vectorized=True was requested, but slice shapes are inconsistent.\n"
            + report.summary()
        )

    global_x, global_y = _collect_global_xy_values(
        datasets,
        x_key=x_key,
        y_key=y_key,
    )

    mapping = make_slice_axis_mapping(
        [ds.slice_value for ds in datasets],
        x_values=global_x,
        y_values=global_y,
        mode=slice_position_mode,
        normalized_span_ratio=normalized_slice_span_ratio,
    )

    slice_axis_span = _safe_span(mapping.plot_values)

    if contour_offset is None:
        contour_offset_value = slice_axis_span * float(contour_offset_ratio)
    else:
        contour_offset_value = float(contour_offset)

    norm_final = infer_color_norm(
        datasets,
        field_key=field_key,
        x_key=x_key,
        y_key=y_key,
        color_transform=color_transform,
        norm=norm,
        norm_type=norm_type,
        vmin=vmin,
        vmax=vmax,
        percentile_clip=percentile_clip,
    )

    cmap_obj = cm.get_cmap(cmap)
    mappable = ScalarMappable(norm=norm_final, cmap=cmap_obj)
    mappable.set_array([])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    use_vectorized_color = report.consistent_shape and vectorized != False
    color_stack = None

    if use_vectorized_color:
        try:
            field_stack = stack_field(
                datasets,
                field_key=field_key,
                x_key=x_key,
                y_key=y_key,
            )
            color_stack = _make_color_values(field_stack, color_transform)
        except Exception as exc:
            if vectorized is True:
                raise
            warnings.warn(
                "Vectorized color preparation failed. Falling back to loop mode. "
                f"Original error: {exc}",
                RuntimeWarning,
            )

    surfaces: List[Any] = []

    all_X3: List[np.ndarray] = []
    all_Y3: List[np.ndarray] = []
    all_Z3: List[np.ndarray] = []

    for s_idx, ds in enumerate(datasets):
        x_values = ds.get_coord(x_key)
        y_values = ds.get_coord(y_key)
        field = ds.get_field(field_key)

        X, Y = _make_mesh(x_values, y_values)
        S = np.full_like(X, mapping.plot_values[s_idx], dtype=float)

        X3, Y3, Z3 = _project_surface_to_3d(
            X,
            Y,
            S,
            slice_direction=slice_direction,
        )

        all_X3.append(X3.ravel())
        all_Y3.append(Y3.ravel())
        all_Z3.append(Z3.ravel())

        if color_stack is not None:
            color_values = color_stack[s_idx]
        else:
            color_values = _make_color_values(field, color_transform)

        rgba = cmap_obj(norm_final(color_values))
        rgba[..., -1] = alpha

        surf = ax.plot_surface(
            X3,
            Y3,
            Z3,
            rstride=stride,
            cstride=stride,
            facecolors=rgba,
            shade=shade,
            edgecolor=edgecolor,
            linewidth=linewidth,
            antialiased=antialiased,
        )
        surfaces.append(surf)

        if contour_levels is not None:
            contour_kwargs_final: Dict[str, Any] = {
                "colors": "k",
                "linewidths": 0.45,
                "alpha": 0.9,
                "zorder": 20,
            }

            if contour_kwargs:
                contour_kwargs_final.update(contour_kwargs)

            contour_artists = _plot_contours_on_slice_plane(
                ax,
                x_values=x_values,
                y_values=y_values,
                color_values=color_values,
                slice_position=mapping.plot_values[s_idx],
                slice_direction=slice_direction,
                contour_levels=contour_levels,
                contour_kwargs=contour_kwargs_final,
                contour_offset=contour_offset_value,
            )

            if warn_empty_contours and len(contour_artists) == 0:
                warnings.warn(
                    f"No contour was drawn for slice {ds.slice_key}={ds.slice_value}. "
                    "Possible reason: contour_levels are outside the field range.",
                    RuntimeWarning,
                )

    xlabel, ylabel, zlabel = _axis_labels_for_direction(
        x_key=x_key,
        y_key=y_key,
        slice_key=slice_key,
        slice_direction=slice_direction,
        axis_labels=axis_labels,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    _set_slice_axis_ticks(
        ax,
        slice_direction=slice_direction,
        mapping=mapping,
        tick_label_format=slice_tick_label_format,
    )

    if title:
        ax.set_title(title)

    if view is not None:
        elev, azim = view
        ax.view_init(elev=elev, azim=azim)

    X3_all = np.concatenate(all_X3)
    Y3_all = np.concatenate(all_Y3)
    Z3_all = np.concatenate(all_Z3)

    # Critical fix:
    # Matplotlib 3D contour may pollute autoscale with field values.
    # Therefore, after all surfaces and contours are drawn, force axis limits
    # using only the real projected geometric coordinates.
    _set_axis_limits_from_projected_data(
        ax,
        x_values_3d=X3_all,
        y_values_3d=Y3_all,
        z_values_3d=Z3_all,
        padding_ratio=axis_limit_padding_ratio,
    )

    _apply_box_aspect(
        ax,
        x_values_3d=X3_all,
        y_values_3d=Y3_all,
        z_values_3d=Z3_all,
        box_aspect=box_aspect,
    )

    if add_colorbar:
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label(colorbar_label or field_key)

    return fig, ax, surfaces, mappable, mapping


def _match_slice_dataset(
    slice_datasets: Sequence[SliceDataset],
    slice_value: Any,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> SliceDataset:
    """
    Find the SliceDataset whose slice_value matches the given value.
    """

    try:
        target = float(slice_value)

        best_ds = None
        best_dist = np.inf

        for ds in slice_datasets:
            try:
                value = float(ds.slice_value)
            except Exception:
                continue

            dist = abs(value - target)

            if dist < best_dist:
                best_dist = dist
                best_ds = ds

        if best_ds is not None and np.isclose(
            float(best_ds.slice_value),
            target,
            atol=atol,
            rtol=rtol,
        ):
            return best_ds

    except Exception:
        pass

    for ds in slice_datasets:
        if ds.slice_value == slice_value:
            return ds

    raise KeyError(f"No SliceDataset matches slice_value={slice_value!r}.")


def _sample_field_on_trajectory_indices(
    trajectory_df: pd.DataFrame,
    slice_datasets: Sequence[SliceDataset],
    *,
    slice_key: str,
    field_key: str,
    ix_key: str = "ix",
    iy_key: str = "iy",
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> np.ndarray:
    """
    Sample one field, for example qfactor or qlog, at the extrema indices
    stored in trajectory_df.

    trajectory_df must contain:
        slice_key, ix, iy
    """

    required = [slice_key, ix_key, iy_key]
    for key in required:
        if key not in trajectory_df.columns:
            raise KeyError(
                f"{key!r} not found in trajectory_df. "
                "Cannot sample field from slice_datasets."
            )

    values = []

    for _, row in trajectory_df.iterrows():
        if not np.isfinite(row[ix_key]) or not np.isfinite(row[iy_key]):
            values.append(np.nan)
            continue

        ds = _match_slice_dataset(
            slice_datasets,
            row[slice_key],
            atol=atol,
            rtol=rtol,
        )

        field = ds.get_field(field_key)

        ix = int(row[ix_key])
        iy = int(row[iy_key])

        if ix < 0 or ix >= field.shape[0] or iy < 0 or iy >= field.shape[1]:
            values.append(np.nan)
            continue

        values.append(field[ix, iy])

    return np.asarray(values)


def _prepare_logq_for_projected_trajectory(
    trajectory_df: pd.DataFrame,
    *,
    slice_datasets: Optional[Sequence[SliceDataset]],
    slice_key: str,
    q_key: str = "qfactor",
    qlog_key: str = "qlog",
    q_field_key: str = "qfactor",
    q_is_log: Optional[bool] = None,
) -> np.ndarray:
    """
    Get log10(Q) values for the trajectory.

    Priority:
        1. If qlog_key exists in trajectory_df, use it directly.
        2. Else if q_key exists in trajectory_df, use log10(q_key).
        3. Else sample q_field_key from slice_datasets using ix/iy.
    """

    if qlog_key in trajectory_df.columns:
        logq = np.asarray(trajectory_df[qlog_key], dtype=float)
        return logq

    if q_key in trajectory_df.columns:
        q = np.asarray(trajectory_df[q_key])

        if np.iscomplexobj(q):
            q = np.abs(q)

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(q.astype(float))

    if slice_datasets is None:
        raise ValueError(
            f"trajectory_df contains neither {qlog_key!r} nor {q_key!r}. "
            "Please provide slice_datasets so q_field_key can be sampled."
        )

    q_values = _sample_field_on_trajectory_indices(
        trajectory_df,
        slice_datasets,
        slice_key=slice_key,
        field_key=q_field_key,
    )

    if q_is_log is None:
        q_is_log = "log" in q_field_key.lower()

    q_values = np.asarray(q_values)

    if np.iscomplexobj(q_values):
        q_values = np.abs(q_values)

    q_values = q_values.astype(float)

    if q_is_log:
        return q_values

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(q_values)


def _collapse_duplicate_parameter_values(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse duplicate interpolation parameters by averaging x, y, and c.

    This is useful if two trajectory rows accidentally share the same slice value.
    """

    df = pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": y,
            "c": c,
        }
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["t", "x", "y", "c"])

    if df.empty:
        raise ValueError("No finite trajectory data left after cleaning.")

    df = (
        df.groupby("t", as_index=False)
        .mean(numeric_only=True)
        .sort_values("t")
    )

    return (
        df["t"].to_numpy(dtype=float),
        df["x"].to_numpy(dtype=float),
        df["y"].to_numpy(dtype=float),
        df["c"].to_numpy(dtype=float),
    )


def _interpolate_trajectory_1d(
    t: np.ndarray,
    values: np.ndarray,
    t_new: np.ndarray,
    *,
    method: str = "pchip",
) -> np.ndarray:
    """
    Interpolate one trajectory quantity.

    method:
        'linear'
        'pchip'   shape-preserving cubic, requires scipy
        'cubic'   cubic spline, requires scipy

    If scipy is unavailable, falls back to linear interpolation.
    """

    method = method.lower()

    if len(t) == 1:
        return np.full_like(t_new, values[0], dtype=float)

    if method == "linear":
        return np.interp(t_new, t, values)

    if method == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator

            interpolator = PchipInterpolator(t, values, extrapolate=False)
            return np.asarray(interpolator(t_new), dtype=float)
        except Exception:
            warnings.warn(
                "PCHIP interpolation failed or scipy is unavailable. "
                "Falling back to linear interpolation.",
                RuntimeWarning,
            )
            return np.interp(t_new, t, values)

    if method == "cubic":
        try:
            from scipy.interpolate import CubicSpline

            interpolator = CubicSpline(t, values, extrapolate=False)
            return np.asarray(interpolator(t_new), dtype=float)
        except Exception:
            warnings.warn(
                "Cubic interpolation failed or scipy is unavailable. "
                "Falling back to linear interpolation.",
                RuntimeWarning,
            )
            return np.interp(t_new, t, values)

    raise ValueError("interpolation method must be 'linear', 'pchip', or 'cubic'.")


def _make_gradient_line_collection_2d(
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray,
    *,
    cmap: str,
    norm: Normalize,
    linewidth: float,
    alpha: float,
) -> LineCollection:
    """
    Build a 2D gradient line from interpolated x/y/color arrays.
    """

    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)

    segment_colors = 0.5 * (color_values[:-1] + color_values[1:])

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    )

    lc.set_array(segment_colors)

    return lc


def plot_projected_extrema_trajectory_2d(
    trajectory_df: pd.DataFrame,
    *,
    x_key: str,
    y_key: str,
    slice_key: str,
    slice_datasets: Optional[Sequence[SliceDataset]] = None,
    q_key: str = "qfactor",
    qlog_key: str = "qlog",
    q_field_key: str = "qfactor",
    q_is_log: Optional[bool] = None,
    interpolation: str = "pchip",
    n_interp: int = 400,
    cmap: str = "hot",
    logq_vmin: float = 2.0,
    logq_vmax: float = 8.0,
    clip_logq: bool = True,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (2.0, 2.0),
    linewidth: float = 1,
    alpha: float = 1.0,
    show_original_points: bool = True,
    original_point_kwargs: Optional[Mapping[str, Any]] = None,
    show_start_end: bool = True,
    start_marker_kwargs: Optional[Mapping[str, Any]] = None,
    end_marker_kwargs: Optional[Mapping[str, Any]] = None,
    add_colorbar: bool = False,
    colorbar_label: str = r"$\log_{10}(Q)$",
    axis_labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    equal_aspect: bool = False,
    sort_by_slice: bool = True,
    return_interpolated: bool = True,
) -> Tuple[Any, Any, LineCollection, Optional[pd.DataFrame]]:
    """
    Project the 3D extrema trajectory onto the 2D (x_key, y_key) plane.

    This function interpolates:
        1. x coordinate as x(slice_key)
        2. y coordinate as y(slice_key)
        3. log10(Q) as logQ(slice_key)

    Then it draws a 2D gradient curve colored by log10(Q).

    Typical usage
    -------------
    plot_projected_extrema_trajectory_2d(
        trajectory_df,
        slice_datasets=slice_datasets,
        x_key="t_slab_factor",
        y_key="fill",
        slice_key="tri_factor",
        q_field_key="qfactor",
        interpolation="pchip",
        cmap="hot",
        logq_vmin=2,
        logq_vmax=8,
    )
    """

    if trajectory_df.empty:
        raise ValueError("trajectory_df is empty.")

    for key in [x_key, y_key, slice_key]:
        if key not in trajectory_df.columns:
            raise KeyError(f"{key!r} not found in trajectory_df.")

    df = trajectory_df.copy()

    if "valid" in df.columns:
        df = df[df["valid"].astype(bool)]

    if df.empty:
        raise ValueError("trajectory_df has no valid points.")

    if sort_by_slice:
        df = df.sort_values(slice_key)

    t = np.asarray(df[slice_key], dtype=float)
    x = np.asarray(df[x_key], dtype=float)
    y = np.asarray(df[y_key], dtype=float)

    logq = _prepare_logq_for_projected_trajectory(
        df,
        slice_datasets=slice_datasets,
        slice_key=slice_key,
        q_key=q_key,
        qlog_key=qlog_key,
        q_field_key=q_field_key,
        q_is_log=q_is_log,
    )

    t, x, y, logq = _collapse_duplicate_parameter_values(t, x, y, logq)

    if len(t) < 2:
        raise ValueError("At least two valid trajectory points are required.")

    n_interp = int(max(n_interp, len(t)))
    t_new = np.linspace(float(np.min(t)), float(np.max(t)), n_interp)

    x_new = _interpolate_trajectory_1d(
        t,
        x,
        t_new,
        method=interpolation,
    )

    y_new = _interpolate_trajectory_1d(
        t,
        y,
        t_new,
        method=interpolation,
    )

    logq_new = _interpolate_trajectory_1d(
        t,
        logq,
        t_new,
        method=interpolation,
    )

    finite = (
        np.isfinite(t_new)
        & np.isfinite(x_new)
        & np.isfinite(y_new)
        & np.isfinite(logq_new)
    )

    t_new = t_new[finite]
    x_new = x_new[finite]
    y_new = y_new[finite]
    logq_new = logq_new[finite]

    if len(t_new) < 2:
        raise ValueError("Interpolation produced fewer than two finite points.")

    if clip_logq:
        logq_for_color = np.clip(logq_new, logq_vmin, logq_vmax)
    else:
        logq_for_color = logq_new

    norm = Normalize(vmin=logq_vmin, vmax=logq_vmax)

    lc = _make_gradient_line_collection_2d(
        x_new,
        y_new,
        logq_for_color,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.add_collection(lc)
    ax.autoscale_view()

    if show_original_points:
        point_kwargs = {
            "s": 12,
            "facecolors": "none",
            "edgecolors": "k",
            "linewidths": 0.5,
            "zorder": 5,
        }

        if original_point_kwargs:
            point_kwargs.update(original_point_kwargs)

        ax.scatter(x, y, **point_kwargs)

    if show_start_end:
        start_kwargs = {
            "s": 24,
            "marker": "o",
            "color": "k",
            "zorder": 6,
            "label": "start",
        }

        end_kwargs = {
            "s": 28,
            "marker": "s",
            "color": "k",
            "zorder": 6,
            "label": "end",
        }

        if start_marker_kwargs:
            start_kwargs.update(start_marker_kwargs)

        if end_marker_kwargs:
            end_kwargs.update(end_marker_kwargs)

        ax.scatter([x_new[0]], [y_new[0]], **start_kwargs)
        ax.scatter([x_new[-1]], [y_new[-1]], **end_kwargs)

    if add_colorbar:
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label(colorbar_label)

    labels = dict(axis_labels or {})
    ax.set_xlabel(labels.get(x_key, x_key))
    ax.set_ylabel(labels.get(y_key, y_key))

    if title:
        ax.set_title(title)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    interp_df = None

    if return_interpolated:
        interp_df = pd.DataFrame(
            {
                slice_key: t_new,
                x_key: x_new,
                y_key: y_new,
                "logq": logq_new,
                "logq_color": logq_for_color,
            }
        )

    return fig, ax, lc, interp_df


def _get_quantity_from_trajectory_or_slice(
    trajectory_df: pd.DataFrame,
    *,
    quantity_key: str,
    slice_datasets: Optional[Sequence[SliceDataset]],
    slice_key: str,
    field_key: Optional[str] = None,
    ix_key: str = "ix",
    iy_key: str = "iy",
) -> np.ndarray:
    """
    Get a quantity along the extrema trajectory.

    Priority:
        1. Use trajectory_df[quantity_key] if it exists.
        2. Otherwise sample field_key from slice_datasets using ix/iy.
    """

    if quantity_key in trajectory_df.columns:
        values = np.asarray(trajectory_df[quantity_key])

        if np.iscomplexobj(values):
            values = np.abs(values)

        return values.astype(float)

    if field_key is None:
        field_key = quantity_key

    if slice_datasets is None:
        raise ValueError(
            f"{quantity_key!r} is not in trajectory_df, and slice_datasets is None. "
            f"Cannot sample field {field_key!r}."
        )

    return _sample_field_on_trajectory_indices(
        trajectory_df,
        slice_datasets,
        slice_key=slice_key,
        field_key=field_key,
        ix_key=ix_key,
        iy_key=iy_key,
    ).astype(float)


def plot_logq_and_ueff_vs_alpha(
    trajectory_df: pd.DataFrame,
    *,
    alpha_key: str,
    slice_datasets: Optional[Sequence[SliceDataset]] = None,
    qlog_key: str = "qlog",
    qfactor_key: str = "qfactor",
    qfactor_field_key: str = "qfactor",
    ueff_key: str = "u_eff",
    ueff_field_key: str = "u_eff",
    figsize: Tuple[float, float] = (2.2, 1.8),
    ax: Optional[Any] = None,
    alpha_label: str = r"$\alpha$",
    qlog_label: str = r"$\log_{10}(Q)$",
    ueff_label: str = r"$U_{\mathrm{eff}}$",
    qlog_ylim: Optional[Tuple[float, float]] = None,
    ueff_ylim: Optional[Tuple[float, float]] = (-1.0, 1.0),
    qlog_line_kwargs: Optional[Mapping[str, Any]] = None,
    ueff_line_kwargs: Optional[Mapping[str, Any]] = None,
    marker: str = ".",
    linewidth: float = 1,
    markersize: float = 3.0,
    sort_by_alpha: bool = True,
    grid: bool = False,
    title: Optional[str] = None,
    return_plot_data: bool = True,
) -> Tuple[Any, Any, Any, Optional[pd.DataFrame]]:
    """
    Plot log10(Q) and U_eff as functions of alpha with two y axes.

    Left axis:
        log10(Q)

    Right axis:
        U_eff

    The function first tries to read qlog and u_eff from trajectory_df.
    If unavailable, it samples qfactor and u_eff from slice_datasets using ix/iy.

    Typical usage
    -------------
    fig, ax_l, ax_r, plot_df = plot_logq_and_ueff_vs_alpha(
        trajectory_df,
        slice_datasets=slice_datasets,
        alpha_key="tri_factor",
    )
    """

    if trajectory_df.empty:
        raise ValueError("trajectory_df is empty.")

    if alpha_key not in trajectory_df.columns:
        raise KeyError(f"{alpha_key!r} not found in trajectory_df.")

    df = trajectory_df.copy()

    if "valid" in df.columns:
        df = df[df["valid"].astype(bool)]

    if df.empty:
        raise ValueError("trajectory_df has no valid rows.")

    if sort_by_alpha:
        df = df.sort_values(alpha_key)

    alpha = np.asarray(df[alpha_key], dtype=float)

    # ---- logQ ----
    qlog: Optional[np.ndarray] = None

    if qlog_key in df.columns:
        qlog = np.asarray(df[qlog_key], dtype=float)
    elif qfactor_key in df.columns:
        qfactor = np.asarray(df[qfactor_key])

        if np.iscomplexobj(qfactor):
            qfactor = np.abs(qfactor)

        with np.errstate(divide="ignore", invalid="ignore"):
            qlog = np.log10(qfactor.astype(float))
    else:
        qfactor = _get_quantity_from_trajectory_or_slice(
            df,
            quantity_key=qfactor_key,
            slice_datasets=slice_datasets,
            slice_key=alpha_key,
            field_key=qfactor_field_key,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            qlog = np.log10(qfactor)

    assert qlog is not None

    # ---- U_eff ----
    ueff = _get_quantity_from_trajectory_or_slice(
        df,
        quantity_key=ueff_key,
        slice_datasets=slice_datasets,
        slice_key=alpha_key,
        field_key=ueff_field_key,
    )

    finite = (
        np.isfinite(alpha)
        & np.isfinite(qlog)
        & np.isfinite(ueff)
    )

    alpha = alpha[finite]
    qlog = qlog[finite]
    ueff = ueff[finite]

    if len(alpha) == 0:
        raise ValueError("No finite data points available for plotting.")

    if ax is None:
        fig, ax_left = plt.subplots(figsize=figsize)
    else:
        ax_left = ax
        fig = ax_left.figure

    ax_right = ax_left.twinx()

    qlog_kwargs: Dict[str, Any] = {
        "marker": marker,
        "linewidth": linewidth,
        "markersize": markersize,
        "label": qlog_label,
    }

    if qlog_line_kwargs:
        qlog_kwargs.update(qlog_line_kwargs)

    ueff_kwargs: Dict[str, Any] = {
        "marker": marker,
        "linewidth": linewidth,
        "markersize": markersize,
        "label": ueff_label,
    }

    if ueff_line_kwargs:
        ueff_kwargs.update(ueff_line_kwargs)

    line_q, = ax_left.plot(alpha, qlog, **qlog_kwargs, color='darkred')
    line_u, = ax_right.plot(alpha, ueff, **ueff_kwargs, color='darkblue')

    ax_left.set_xlabel(alpha_label)
    ax_left.set_ylabel(qlog_label)
    ax_right.set_ylabel(ueff_label)

    if qlog_ylim is not None:
        ax_left.set_ylim(*qlog_ylim)

    if ueff_ylim is not None:
        ax_right.set_ylim(*ueff_ylim)

    if grid:
        ax_left.grid(True, linewidth=0.4, alpha=0.35)

    if title:
        ax_left.set_title(title)

    # # Combined legend
    # ax_left.legend(
    #     [line_q, line_u],
    #     [qlog_label, ueff_label],
    #     loc="best",
    #     frameon=False,
    # )

    plot_df = None

    if return_plot_data:
        plot_df = pd.DataFrame(
            {
                alpha_key: alpha,
                qlog_key: qlog,
                ueff_key: ueff,
            }
        )

    return fig, ax_left, ax_right, plot_df




def save_figure(
    fig: Any,
    path: str,
    *,
    dpi: int = 600,
    transparent: bool = True,
    bbox_inches: str = "tight",
) -> None:
    fig.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
    )
