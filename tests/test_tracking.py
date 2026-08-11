from __future__ import annotations

import numpy as np

from plot_workflows.tracking import (
    CandidateSet,
    TrackingWorkflowOptions,
    run_path_workflow,
    run_surface_workflow,
)


def _point(values):
    return CandidateSet(list(range(len(values))), np.asarray(values, dtype=float)[:, None])


def test_path_workflow_returns_maps_and_provenance_for_missing_candidate():
    result = run_path_workflow(
        [_point([1.0, 2.0]), _point([1.1]), _point([1.2, 2.2])],
        [0.0, 1.0, 2.0],
        options=TrackingWorkflowOptions(n_tracks=2),
    )
    assert result.algorithm == "explicit-anchor-bidirectional-path"
    assert result.labels[0][0] == 0
    assert result.labels[2][1] == 1
    assert 1 not in result.tracked_index[1]
    assert result.runtime["distribution"] == "eigenmode-analysis"


def test_surface_workflow_uses_center_anchor_and_rectangular_contract():
    points = {
        (i, j): _point([float(i + j), float(i + j + 10)])
        for i in range(3)
        for j in range(3)
    }
    result = run_surface_workflow(
        points,
        (3, 3),
        options=TrackingWorkflowOptions(n_tracks=2),
    )
    assert result.anchor_id == (1, 1)
    assert result.node_ids[0] == (1, 1)
    assert result.algorithm == "center-anchor-bfs-grid"
    assert result.coverage == 1.0
