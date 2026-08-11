"""Adapters from caller payloads to Plot Foundation specifications."""

from __future__ import annotations

import numpy as np


def shared_plotting_provenance():
    """Return the exact Plot Foundation source used by plotting adapters."""

    from plot_foundation.runtime import runtime_info

    return runtime_info().to_dict()


def plot_visualizer_line_shared(
    plotter,
    index,
    x_key,
    z1_key,
    z2_key=None,
    z3_key=None,
    **plot_params,
):
    """Map one dataset to domain-neutral line primitives."""

    from plot_foundation import (
        AxesSpec,
        ColorMappedLineSeries,
        ColorMappedLineStyle,
        FigureSpec,
        LinePlotSpec,
        LineSeries,
        RibbonSeries,
        RibbonStyle,
        SeriesStyle,
        MatplotlibStyle,
        render_line_plot,
        style_context,
    )

    params = {**(plotter.config.plot_params or {}), **plot_params}
    if params.get("gradient_fill"):
        raise NotImplementedError(
            "plot-foundation adapter does not yet support gradient_fill"
        )

    dataset = plotter.raw_datasets["data_list"][index]
    x = np.squeeze(plotter.coordinates[x_key])
    y = np.squeeze(dataset[z1_key])
    width = np.squeeze(dataset[z2_key]) if z2_key is not None else None
    color_values = np.squeeze(dataset[z3_key]) if z3_key is not None else None
    color = params.get("default_line_color", False)
    if color is False:
        from matplotlib import colormaps

        color = colormaps["tab10"](index % 10)

    enable_dynamic_color = bool(params.get("enable_dynamic_color", False))
    enable_fill = bool(params.get("enable_fill", False))
    if enable_dynamic_color and color_values is None:
        raise ValueError("enable_dynamic_color requires z3_key")
    if enable_fill and width is None:
        raise ValueError("enable_fill requires z2_key")

    series = []
    # Preserve the legacy branch order: dynamic color wins over fill when both
    # flags are supplied in one call. Callers that need both draw the ribbon
    # Draw the reference line first and the colored line second.
    if enable_dynamic_color:
        series.append(
            ColorMappedLineSeries(
                x,
                y,
                color_values,
                label=str(index),
                style=ColorMappedLineStyle(
                    cmap=params.get("cmap", "viridis"),
                    vmin=params.get("global_color_vmin"),
                    vmax=params.get("global_color_vmax"),
                    linewidth=float(params.get("linewidth_base", 1)),
                    alpha=float(params.get("alpha_line", 1)),
                    zorder=2,
                ),
            )
        )
    elif enable_fill:
        series.append(
            RibbonSeries(
                x,
                y,
                np.abs(width) * float(params.get("scale", 0.5)),
                label="Fill Width",
                style=RibbonStyle(
                    color=params.get("default_fill_color", "gray"),
                    alpha=float(params.get("alpha_fill", 0.3)),
                    edge_color=params.get("edge_color", "none"),
                    zorder=0,
                ),
            )
        )
    else:
        series.append(
            LineSeries(
                x,
                y,
                label=str(index),
                style=SeriesStyle(
                    color=color,
                    linewidth=float(params.get("linewidth_base", 1)),
                    linestyle=params.get("default_linestyle", "-"),
                    alpha=float(params.get("alpha_line", 1)),
                    zorder=2,
                ),
            )
        )

    ax = plotter.twin_plot_ax(params.get("twinx", False), params.get("twiny", False))
    scoped_style = MatplotlibStyle(
        font_size=float(plotter.config.fs),
        font_family=str(plotter.config.font),
        sans_serif=(str(plotter.config.font), "DejaVu Sans"),
        xtick_direction=str(plotter.config.tick_direction),
        ytick_direction=str(plotter.config.tick_direction),
    )
    with style_context(scoped_style):
        result = render_line_plot(
            LinePlotSpec(
                series=tuple(series),
                axes=AxesSpec(),
                figure=FigureSpec(figsize=tuple(plotter.config.figsize)),
            ),
            ax=ax,
        )
    plotter.ax = result.axes
    plotter.plot_xlims.append((np.nanmin(x), np.nanmax(x)))
    plotter.plot_zlims.append((np.nanmin(y), np.nanmax(y)))
    return result


def plot_visualizer_heatmap_shared(
    plotter,
    index,
    x_key,
    y_key,
    field_key,
    **plot_params,
):
    """Map one scalar grid to Plot Foundation's xy-grid contract."""

    from plot_foundation import (
        AxesSpec,
        FigureSpec,
        HeatmapPlotSpec,
        HeatmapStyle,
        MatplotlibStyle,
        render_heatmap,
        style_context,
    )

    allowed = {"cmap", "vmin", "vmax", "alpha", "aspect"}
    unknown = sorted(set(plot_params) - allowed)
    if unknown:
        raise NotImplementedError(
            f"plot-foundation heatmap adapter does not support options: {unknown}"
        )
    x, y, values = plotter._get_coord_field(index, field_key, x_key, y_key)
    scoped_style = MatplotlibStyle(
        font_size=float(plotter.config.fs),
        font_family=str(plotter.config.font),
        sans_serif=(str(plotter.config.font), "DejaVu Sans"),
        xtick_direction=str(plotter.config.tick_direction),
        ytick_direction=str(plotter.config.tick_direction),
    )
    spec = HeatmapPlotSpec(
        x=np.asarray(x),
        y=np.asarray(y),
        values=np.asarray(values),
        axes=AxesSpec(aspect=plot_params.get("aspect")),
        figure=FigureSpec(figsize=tuple(plotter.config.figsize)),
        style=HeatmapStyle(
            cmap=plot_params.get("cmap", "viridis"),
            vmin=plot_params.get("vmin"),
            vmax=plot_params.get("vmax"),
            alpha=float(plot_params.get("alpha", 1.0)),
        ),
    )
    with style_context(scoped_style):
        result = render_heatmap(spec, ax=plotter.ax)
    plotter.ax = result.axes
    return result.artists[0]
