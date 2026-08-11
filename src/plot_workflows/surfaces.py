"""Schema-neutral surface plot spec composition for Plot Foundation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from plot_foundation import (
    HeatmapPlotSpec,
    HeatmapStyle,
    MultiSurfacePlotSpec,
    SurfaceLayer,
    SurfaceStyle,
)

from .style_profiles import axes_spec_for_profile, figure_spec_for_profile, resolve_plot_profile


def _surface_arrays(payload: Any) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """Extract x/y and a y-first field from a Workbench-like payload."""

    if hasattr(payload, "x_grid"):
        x = getattr(payload, "x_grid")
        y = getattr(payload, "y_grid")
        values = getattr(payload, "z_values")
        names = getattr(payload, "quantity_names", None) or ("value",)
    elif isinstance(payload, Mapping):
        x = payload.get("x_grid", payload.get("x"))
        y = payload.get("y_grid", payload.get("y"))
        values = payload.get("z_values", payload.get("values"))
        names = payload.get("quantity_names") or ("value",)
    else:
        raise TypeError("surface payload must expose x_grid/y_grid/z_values")
    if x is None or y is None or values is None:
        raise ValueError("surface payload is missing x, y, or values")
    quantity = str(names[0]) if not isinstance(names, str) else names
    if isinstance(values, Mapping):
        values = values.get(quantity)
    if values is None:
        raise ValueError(f"surface payload is missing quantity {quantity!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    expected = (len(y), len(x))
    if x.ndim != 1 or y.ndim != 1 or values.shape != expected:
        raise ValueError(f"surface values must have shape {expected}, got {values.shape}")
    return x, y, quantity, values.T


def surface_heatmap_spec_from_payload(
    payload: Any,
    *,
    title: str = "",
    cmap: str = "viridis",
    contour: bool = True,
    figsize: tuple[float, float] | None = (8, 6),
    style_profile: str = "diagnostic",
    figure_preset: str | None = None,
) -> HeatmapPlotSpec:
    """Compose a standard heatmap spec without owning figure lifecycle."""

    x, y, quantity, values = _surface_arrays(payload)
    finite = values[np.isfinite(values)]
    levels = None
    if contour and finite.size >= 2 and np.ptp(finite) > 0:
        levels = tuple(np.linspace(float(finite.min()), float(finite.max()), 8))
    return HeatmapPlotSpec(
        x=x,
        y=y,
        values=values,
        axes=axes_spec_for_profile(
            style_profile,
            xlabel="kx",
            ylabel="ky",
            title=title or f"{quantity}(kx, ky)",
            aspect="equal",
        ),
        figure=figure_spec_for_profile(
            style_profile,
            figsize=figsize,
            preset=figure_preset,
        ),
        style=HeatmapStyle(cmap=cmap, contour_levels=levels, contour_linewidth=0.3),
        colorbar=resolve_plot_profile(style_profile).policy.show_colorbar,
        colorbar_label=(
            quantity
            if resolve_plot_profile(style_profile).policy.show_axis_labels
            else ""
        ),
    )


def multi_surface_spec_from_payloads(
    height_payloads: Sequence[Any],
    *,
    color_payloads: Sequence[Any] | None = None,
    alpha_payloads: Sequence[Any] | None = None,
    title: str = "",
    cmap: str = "plasma",
    vmin=None,
    vmax=None,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (12, 9),
    style_profile: str = "diagnostic",
    figure_preset: str | None = None,
) -> tuple[MultiSurfacePlotSpec, str, str]:
    """Compose a multi-layer 3D surface spec and return its color quantity."""

    heights = tuple(height_payloads)
    if not heights:
        raise ValueError("height_payloads cannot be empty")
    colors = tuple(color_payloads) if color_payloads is not None else (None,) * len(heights)
    alphas = tuple(alpha_payloads) if alpha_payloads is not None else (None,) * len(heights)
    if len(colors) != len(heights) or len(alphas) != len(heights):
        raise ValueError("surface payload layer lengths must match")
    x, y, height_name, _ = _surface_arrays(heights[0])
    layers = []
    color_name = height_name
    for height_payload, color_payload, alpha_payload in zip(heights, colors, alphas):
        layer_x, layer_y, _, z = _surface_arrays(height_payload)
        if not np.array_equal(layer_x, x) or not np.array_equal(layer_y, y):
            raise ValueError("all surface payloads must share identical x/y grids")
        color_values = alpha_values = None
        if color_payload is not None:
            color_x, color_y, color_name, color_values = _surface_arrays(color_payload)
            if not np.array_equal(color_x, x) or not np.array_equal(color_y, y):
                raise ValueError("color payload grid must match its height payload")
        if alpha_payload is not None:
            alpha_x, alpha_y, _, alpha_values = _surface_arrays(alpha_payload)
            if not np.array_equal(alpha_x, x) or not np.array_equal(alpha_y, y):
                raise ValueError("alpha payload grid must match its height payload")
        layers.append(SurfaceLayer(z=z, color_values=color_values, alpha_values=alpha_values, alpha=alpha))
    spec = MultiSurfacePlotSpec(
        x=x,
        y=y,
        layers=tuple(layers),
        figure=figure_spec_for_profile(
            style_profile,
            figsize=figsize,
            preset=figure_preset,
        ),
        style=SurfaceStyle(cmap=cmap, vmin=vmin, vmax=vmax),
        xlabel="kx" if resolve_plot_profile(style_profile).policy.show_axis_labels else "",
        ylabel="ky" if resolve_plot_profile(style_profile).policy.show_axis_labels else "",
        zlabel=height_name if resolve_plot_profile(style_profile).policy.show_axis_labels else "",
    )
    return spec, title, color_name


__all__ = ["surface_heatmap_spec_from_payload", "multi_surface_spec_from_payloads"]
