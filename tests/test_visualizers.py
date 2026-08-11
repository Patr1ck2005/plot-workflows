import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_workflows import OneDimFieldVisualizer, PlotConfig, TwoDimFieldVisualizer


def _line_plotter():
    x = np.linspace(0.0, 1.0, 5)
    return OneDimFieldVisualizer(
        config=PlotConfig(show=False, figsize=(2, 2)),
        data_path=None,
    ), x


def test_shared_line_visualizer_renders_payload():
    plotter, x = _line_plotter()
    plotter.raw_datasets = {"coords": {"k": x}, "data_list": [{"f": x**2}]}
    plotter.coordinates = plotter.raw_datasets["coords"]
    plotter.data_num = 1
    plotter.new_2d_fig()
    plotter.plot(0, x_key="k", z1_key="f")
    assert len(plotter.ax.lines) == 1
    plt.close(plotter.fig)

def test_shared_grid_visualizer_renders_heatmap():
    x = np.linspace(0.0, 1.0, 3)
    y = np.linspace(-1.0, 1.0, 4)
    plotter = TwoDimFieldVisualizer(config=PlotConfig(show=False, figsize=(2, 2)))
    plotter.raw_datasets = {"coords": {"x": x, "y": y}, "data_list": [{"f": np.ones((3, 4))}]}
    plotter.coordinates = plotter.raw_datasets["coords"]
    plotter.data_num = 1
    plotter.new_2d_fig()
    artist = plotter.imshow_field_shared(0, "x", "y", "f", cmap="viridis")
    assert artist is not None
    plt.close(plotter.fig)
