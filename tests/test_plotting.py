from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_workflows.plotting import band_plot_spec_from_payloads, render_band_payloads


class Payload:
    def __init__(self, x, y):
        self.x_values = x
        self.y_values = y


def test_band_spec_accepts_workbench_like_payloads():
    payloads = {
        "B00": Payload(np.arange(3), {"f_thz": np.array([1.0, 1.1, 1.2])}),
        "B01": Payload(np.arange(3), {"f_thz": np.array([2.0, 2.1, 2.2])}),
    }
    spec = band_plot_spec_from_payloads(
        payloads,
        title="test",
        ylabel="f",
        is_tracked=True,
    )
    assert len(spec.series) == 2
    fig, ax = plt.subplots()
    result = render_band_payloads(
        ax,
        payloads,
        title="test",
        ylabel="f",
        is_tracked=True,
    )
    assert result.artists
    assert ax.has_data()
    plt.close(fig)


def test_publication_profile_suppresses_diagnostic_chrome():
    payloads = {
        "B00": Payload(np.arange(3), {"f_thz": np.array([1.0, 1.1, 1.2])}),
    }
    spec = band_plot_spec_from_payloads(
        payloads,
        title="diagnostic title",
        ylabel="Frequency",
        is_tracked=True,
        style_profile="publication_minimal",
        figure_preset="single",
        figsize=None,
    )
    assert spec.axes.title == ""
    assert spec.axes.xlabel == ""
    assert spec.axes.ylabel == ""
    assert spec.axes.grid is False
    assert spec.axes.legend is False
    assert spec.figure.figsize == (1.5, 1.5)
    assert spec.figure.tight_layout is False
