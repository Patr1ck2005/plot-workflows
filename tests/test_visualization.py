from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_workflows import LinePlotter, PlotConfig


class DemoPlotter(LinePlotter):
    def prepare_data(self, **kwargs):
        del kwargs

    def plot(self, **kwargs):
        del kwargs


def test_shared_plot_config_and_line_lifecycle_are_headless():
    plotter = DemoPlotter(PlotConfig(figsize=(3, 2), font="DejaVu Sans", fs=8, show=False, plot_params={}))
    plotter.new_2d_fig()
    plotter.plot_line(np.arange(3), np.array([1.0, 2.0, 1.5]), default_line_color="black")
    assert plotter.ax.has_data()
    plotter.save_and_show(save=False)
    plt.close(plotter.fig)


def test_plot_config_resolves_shared_profile():
    config = PlotConfig(style_profile="publication_minimal", show=False)
    assert config.resolved_profile.name == "publication_minimal"
    assert config.figure_policy.show_title is False
