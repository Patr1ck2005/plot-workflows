from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_workflows import multi_surface_spec_from_payloads, surface_heatmap_spec_from_payload


class Payload:
    quantity_names = ["f_thz"]

    def __init__(self, values):
        self.x_grid = np.array([0.0, 1.0, 2.0])
        self.y_grid = np.array([-1.0, 1.0])
        self.z_values = {"f_thz": values}


def test_surface_spec_transposes_workbench_yx_values():
    payload = Payload(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    spec = surface_heatmap_spec_from_payload(payload)
    assert spec.values.shape == (3, 2)
    np.testing.assert_array_equal(spec.values[:, 0], [1.0, 2.0, 3.0])


def test_publication_surface_profile_uses_minimal_policy():
    payload = Payload(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    spec = surface_heatmap_spec_from_payload(
        payload,
        style_profile="publication_minimal",
        figure_preset="single",
        figsize=None,
    )
    assert spec.axes.xlabel == ""
    assert spec.axes.ylabel == ""
    assert spec.axes.title == ""
    assert spec.colorbar is False
    assert spec.figure.figsize == (1.5, 1.5)


def test_multi_surface_spec_is_renderable_without_opening_window():
    payload = Payload(np.ones((2, 3)))
    spec, title, color_name = multi_surface_spec_from_payloads([payload])
    assert len(spec.layers) == 1
    assert title == ""
    assert color_name == "f_thz"
    plt.close("all")
