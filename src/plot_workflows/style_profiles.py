"""Profile-aware figure and axes defaults shared by workflow composers."""

from __future__ import annotations

from plot_foundation import AxesSpec, FigureSpec, PlotProfile, profile


def resolve_plot_profile(value: str | PlotProfile) -> PlotProfile:
    """Resolve a stable Plot Foundation profile without consumer policy."""

    return profile(value) if isinstance(value, str) else value


def axes_spec_for_profile(
    value: str | PlotProfile,
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    legend: bool = False,
    grid: bool | None = None,
    **options,
) -> AxesSpec:
    """Create axes defaults while preserving explicit physical options."""

    selected = resolve_plot_profile(value)
    policy = selected.policy
    return AxesSpec(
        xlabel=xlabel if policy.show_axis_labels else "",
        ylabel=ylabel if policy.show_axis_labels else "",
        title=title if policy.show_title else "",
        legend=legend and policy.show_legend,
        grid=policy.grid if grid is None else grid and policy.grid,
        **options,
    )


def figure_spec_for_profile(
    value: str | PlotProfile,
    *,
    figsize: tuple[float, float] | None = None,
    preset: str | None = None,
    rows: int = 1,
    cols: int = 1,
) -> FigureSpec:
    """Create figure geometry with profile-controlled layout defaults."""

    selected = resolve_plot_profile(value)
    if preset is not None:
        figure = FigureSpec.from_preset(preset, rows=rows, cols=cols)
        if figsize is not None:
            figure = FigureSpec(
                figsize=figsize,
                tight_layout=figure.tight_layout,
                preset=figure.preset,
            )
    else:
        figure = FigureSpec(figsize=figsize or (6.0, 4.0))
    return FigureSpec(
        figsize=figure.figsize,
        tight_layout=selected.policy.tight_layout,
        preset=figure.preset,
    )


__all__ = [
    "axes_spec_for_profile",
    "figure_spec_for_profile",
    "resolve_plot_profile",
]
