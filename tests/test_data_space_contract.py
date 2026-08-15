from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
CANONICAL_SPEC = ROOT / "docs" / "data-space-and-visualization.md"


def test_canonical_spec_keeps_core_data_space_terms_and_agent_record():
    text = CANONICAL_SPEC.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for term in (
        "Rendering dimension must never be used to infer data dimension",
        "### 4.1 0D / point observable",
        "### 4.2 1D",
        "### 4.3 2D",
        "### 4.4 Dense 3D",
        "### 4.5 Quasi-2D",
        "### 4.6 Quasi-3D",
        "## 2. Analysis-space transformations",
        "Raw data space",
        "Lift/expansion",
        "Peak/ridge extraction",
        "Slice-wise aggregation",
        "resulting dimension",
        "`quasi-1D` is intentionally not part",
        "heatmap and a 3D surface",
        "Intrinsic data dimension",
        "Sampling topology",
        "Parameter-space type",
        "Physical quantity",
        "Compatible view candidates",
        "Selected views and rationale",
        "Transformation status",
        "direct raw-data view",
        "transformed derived-data view",
    ):
        assert term in normalized


def test_plot_workflows_agent_entry_links_the_canonical_spec():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    normalized = " ".join(agents.split())

    assert "docs/data-space-and-visualization.md" in agents
    assert "seven-field Agent decision record" in normalized
    assert "Rendering dimension does not determine data dimension" in normalized


def test_angle_frequency_peak_ridge_is_a_derived_1d_space():
    """A 2D field may be reduced to a 1D physical ridge before rendering."""
    theta = np.asarray([-1.0, 0.0, 1.0])
    frequency = np.asarray([1.0, 2.0, 3.0, 4.0])
    peak_indices = np.asarray([0, 2, 1])
    intensity = np.zeros((len(theta), len(frequency)), dtype=float)
    intensity[np.arange(len(theta)), peak_indices] = 10.0

    peak_frequency = frequency[np.argmax(intensity, axis=1)]
    derived_record = {
        "raw_intrinsic_dimension": "2D",
        "raw_coordinates": ("theta", "frequency"),
        "operator": "peak/ridge extraction",
        "derived_coordinates": ("theta",),
        "derived_quantity": "f_peak(theta)",
        "derived_dimension": "1D",
    }

    assert intensity.shape == (len(theta), len(frequency))
    assert peak_frequency.tolist() == [1.0, 3.0, 2.0]
    assert derived_record["derived_dimension"] == "1D"
    assert len(derived_record["derived_coordinates"]) == 1


def test_eigenfrequency_can_be_lifted_to_a_frequency_response():
    """A point eigen-observable can seed a sampled 1D response space."""
    eigenfrequency = 2.0
    frequency = np.linspace(1.0, 3.0, 9)
    response = 1.0 / (1.0 + ((frequency - eigenfrequency) / 0.1) ** 2)
    response_channels = {name: response.copy() for name in ("R", "T", "A")}

    assert np.asarray(eigenfrequency).ndim == 0
    assert frequency.ndim == 1
    assert all(values.shape == frequency.shape for values in response_channels.values())
    assert int(np.argmax(response)) == 4


def test_quasi_3d_slice_extrema_is_a_derived_1d_space():
    slices = _quasi_3d_slices()
    from plot_workflows.multislice import find_extrema_trajectory

    trajectory = find_extrema_trajectory(
        slices,
        x_key="kx",
        y_key="ky",
        slice_key="alpha",
        field_key="signal",
        mode="max",
    )

    assert len(slices) == 3
    assert len(trajectory) == len(slices)
    assert list(trajectory["alpha"]) == [0.0, 1.0, 2.0]
    # The extrema trajectory is parameterized by one slice coordinate, even
    # though consumers may render its (kx, ky) locations in 3D.
    assert trajectory["alpha"].nunique() == 3


def _quasi_3d_slices():
    pandas = pytest.importorskip("pandas")
    del pandas
    from plot_workflows.multislice import SliceDataset

    x = np.asarray([0.0, 1.0, 2.0])
    y = np.asarray([-1.0, 1.0])
    maxima = ((0, 0), (1, 1), (2, 0))
    slices = []
    for slice_value, (ix, iy) in enumerate(maxima):
        signal = np.zeros((len(x), len(y)), dtype=float)
        signal[ix, iy] = 3.0 + slice_value
        slices.append(
            SliceDataset(
                slice_key="alpha",
                slice_value=float(slice_value),
                coords={"kx": x, "ky": y},
                fields={"signal": signal},
            )
        )
    return slices


def test_quasi_3d_slice_family_supports_four_view_families():
    pytest.importorskip("scipy")
    from plot_workflows import surface_heatmap_spec_from_payload
    from plot_workflows.multislice import (
        find_extrema_trajectory,
        plot_extrema_trajectory_3d,
        plot_heatmap_slices_3d,
        plot_projected_extrema_trajectory_2d,
    )

    slices = _quasi_3d_slices()

    per_slice_specs = [
        surface_heatmap_spec_from_payload(
            {
                "x_grid": ds.get_coord("kx"),
                "y_grid": ds.get_coord("ky"),
                "quantity_names": ["signal"],
                "z_values": {"signal": ds.get_field("signal").T},
            }
        )
        for ds in slices
    ]
    assert len(per_slice_specs) == len(slices)
    assert all(spec.values.shape == (3, 2) for spec in per_slice_specs)

    stacked_fig, stacked_ax, stacked_artists, _, mapping = plot_heatmap_slices_3d(
        slices,
        x_key="kx",
        y_key="ky",
        slice_key="alpha",
        field_key="signal",
        add_colorbar=False,
    )
    assert stacked_fig is stacked_ax.figure
    assert len(stacked_artists) == len(slices)
    assert len(mapping.plot_values) == len(slices)

    trajectory = find_extrema_trajectory(
        slices,
        x_key="kx",
        y_key="ky",
        slice_key="alpha",
        field_key="signal",
        mode="max",
    )
    assert trajectory[["kx", "ky"]].to_records(index=False).tolist() == [
        (0.0, -1.0),
        (1.0, 1.0),
        (2.0, -1.0),
    ]

    trajectory_fig, trajectory_ax, _, trajectory_mapping = plot_extrema_trajectory_3d(
        trajectory,
        x_key="kx",
        y_key="ky",
        slice_key="alpha",
        add_colorbar=False,
    )
    assert trajectory_fig is trajectory_ax.figure
    assert len(trajectory_mapping.plot_values) == len(slices)

    projection_fig, projection_ax, projection_artist, interpolated = (
        plot_projected_extrema_trajectory_2d(
            trajectory,
            x_key="kx",
            y_key="ky",
            slice_key="alpha",
            qlog_key="score_value",
            q_is_log=True,
            interpolation="linear",
            n_interp=9,
            add_colorbar=False,
        )
    )
    assert projection_fig is projection_ax.figure
    assert projection_artist.get_segments()
    assert len(interpolated) == 9
