"""High-level band plotting specs shared by A and Workbench.

Inputs are deliberately duck-typed: consumers can pass their own payload
objects as long as they expose ``x_values`` and ``y_values`` or provide
``x`` and a quantity mapping.  The package owns styles and spec composition;
consumers still own figure creation, file paths, and physical labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

import numpy as np

from plot_foundation import (
    LinePlotSpec,
    LineSeries,
    SeriesStyle,
    render_line_plot,
)

from .style_profiles import axes_spec_for_profile, figure_spec_for_profile


def _payload_arrays(payload: Any, quantity_name: str) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(payload, "x_values") and hasattr(payload, "y_values"):
        x = getattr(payload, "x_values")
        values = getattr(payload, "y_values")
    elif isinstance(payload, Mapping):
        x = payload.get("x_values", payload.get("x"))
        values = payload.get("y_values", payload.get("y"))
    else:
        raise TypeError("band payload must expose x_values/y_values or x/y")
    if x is None or values is None:
        raise ValueError("band payload is missing x or y values")
    if isinstance(values, Mapping):
        y = values.get(quantity_name)
    else:
        y = values
    if y is None:
        raise ValueError(f"band payload is missing quantity {quantity_name!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("band payload x/y arrays must be one-dimensional and equal")
    return x, y


def band_plot_spec_from_payloads(
    band_payloads: Mapping[str, Any],
    *,
    title: str,
    ylabel: str,
    quantity_name: str = "f_thz",
    is_tracked: bool,
    target_band: str | None = None,
    use_semilogy: bool = False,
    diffraction_thresholds=None,
    xlabel: str = "k",
    show_legend: bool = True,
    figsize: tuple[float, float] | None = (10, 6),
    style_profile: str = "diagnostic",
    figure_preset: str | None = None,
) -> LinePlotSpec:
    """Compose the standard raw/tracked band plot contract."""

    series = []
    has_target = target_band is not None
    for label, payload in band_payloads.items():
        x, y = _payload_arrays(payload, quantity_name)
        n_inf = int(np.sum(np.isinf(y)))
        if n_inf and use_semilogy:
            warnings.warn(
                f"Band {label!r}: {n_inf} infinite values replaced with 1e15 "
                "for log-plot display",
                stacklevel=2,
            )
            y = np.where(np.isinf(y), 1e15, y)
        is_target = has_target and label == target_band
        show_label = label if (is_target or not has_target) else ""
        if not is_tracked:
            style = SeriesStyle(
                kind="scatter", color="black", scatter_size=8,
                alpha=0.5, zorder=1,
            )
            show_label = ""
        elif is_target:
            style = SeriesStyle(
                color="red", linewidth=2.0, marker="o",
                marker_size=6, zorder=10,
            )
        elif has_target:
            style = SeriesStyle(
                color="gray", linewidth=0.5, marker="o", marker_size=3,
                zorder=1,
            )
        else:
            style = SeriesStyle(linewidth=0.8, marker="o", marker_size=4, zorder=1)
        series.append(LineSeries(x, y, label=show_label, style=style))

    if diffraction_thresholds is not None:
        threshold_x, thresholds = diffraction_thresholds
        n_air = thresholds.get("n_air", 1.0)
        n_sub = thresholds.get("n_sub", 1.45)
        if "air" in thresholds:
            series.append(
                LineSeries(
                    threshold_x,
                    thresholds["air"],
                    label=f"1st diff (cladding, n={n_air:g})",
                    role="guide",
                    style=SeriesStyle(
                        color="red", linestyle="--", linewidth=1.2,
                        alpha=0.7, zorder=5,
                    ),
                )
            )
        if "substrate" in thresholds:
            series.append(
                LineSeries(
                    threshold_x,
                    thresholds["substrate"],
                    label=f"1st diff (substrate, n={n_sub:g})",
                    role="guide",
                    style=SeriesStyle(
                        color="blue", linestyle="-.", linewidth=1.2,
                        alpha=0.7, zorder=5,
                    ),
                )
            )

    return LinePlotSpec(
        series=tuple(series),
        axes=axes_spec_for_profile(
            style_profile,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            yscale="log" if use_semilogy else "linear",
            grid_alpha=0.3,
            legend=show_legend,
            legend_loc="upper right",
            legend_ncol=2,
            legend_fontsize=7,
        ),
        figure=figure_spec_for_profile(
            style_profile,
            figsize=figsize,
            preset=figure_preset,
        ),
        preserve_data_y_limits=diffraction_thresholds is not None,
    )


def render_band_payloads(ax, band_payloads: Mapping[str, Any], **options):
    """Render a standard band spec onto caller-owned axes."""

    spec = band_plot_spec_from_payloads(band_payloads, **options)
    return render_line_plot(spec, ax=ax)


__all__ = ["band_plot_spec_from_payloads", "render_band_payloads"]
