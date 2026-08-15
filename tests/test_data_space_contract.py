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
        "### 3.1 1D",
        "### 3.2 2D",
        "### 3.3 Dense 3D",
        "### 3.4 Quasi-2D",
        "### 3.5 Quasi-3D",
        "`quasi-1D` is intentionally not part",
        "heatmap and a 3D surface",
        "Intrinsic data dimension",
        "Sampling topology",
        "Parameter-space type",
        "Physical quantity",
        "Compatible view candidates",
        "Selected views and rationale",
    ):
        assert term in normalized


def test_plot_workflows_agent_entry_links_the_canonical_spec():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    normalized = " ".join(agents.split())

    assert "docs/data-space-and-visualization.md" in agents
    assert "six-field Agent decision record" in normalized
    assert "Rendering dimension does not determine data dimension" in normalized


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
