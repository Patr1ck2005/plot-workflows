"""Schema-neutral tracking workflows built on EigenmodeAnalysis.

This module owns the repeated orchestration around the lower-level tracking
algorithms: option normalization, path/grid traversal, provenance, and the
legacy-compatible label maps consumed by A and B adapters.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import sys
from typing import Any

import numpy as np

from eigenmode_analysis.tracking import (
    CandidateSet,
    GraphTrackingResult,
    TrackingConfig,
    TrackingResult,
    TraversalEdge,
    group_vectors_one_sided_hungarian,
    infer_candidate_selector,
    pick_candidate_data_stable,
    rectilinear_traversal,
    split_disconnected_grid_tracks,
    track_graph,
    track_path,
)


@dataclass(frozen=True)
class TrackingWorkflowOptions:
    """Numerical policy shared by path and rectangular-grid workflows."""

    n_tracks: int | None = None
    value_weights: float | Sequence[float] = 1.0
    derivative_weights: float | Sequence[float] = 0.1
    nan_cost_penalty: float = 10.0
    anchor_feature: int = 0
    anchor_window: float | None = None
    anchor_slope: float | None = None
    rank_penalty: float = 0.0


@dataclass(frozen=True)
class TrackingWorkflowResult:
    """Tracking result plus consumer-neutral maps and runtime provenance."""

    result: TrackingResult | GraphTrackingResult
    node_ids: tuple[Hashable, ...]
    anchor_id: Hashable
    algorithm: str
    runtime: Mapping[str, Any]
    labels: Mapping[Hashable, Mapping[Hashable, int]]
    tracked_index: Mapping[int, Mapping[Hashable, Hashable]]

    @property
    def candidate_ids(self):
        return self.result.candidate_ids

    @property
    def diagnostics(self):
        return self.result.diagnostics

    @property
    def coverage(self) -> float:
        if isinstance(self.result, TrackingResult):
            return self.result.coverage
        if not self.result.candidate_ids or not self.result.candidate_ids[0]:
            return 0.0
        assigned = sum(
            candidate_id is not None
            for row in self.result.candidate_ids
            for candidate_id in row
        )
        return assigned / (len(self.result.candidate_ids) * len(self.result.candidate_ids[0]))


def _runtime() -> dict[str, Any]:
    import eigenmode_analysis

    try:
        from eigenmode_analysis.runtime import runtime_info
    except ImportError:
        runtime = {
            "distribution": "eigenmode-analysis",
            "version": eigenmode_analysis.__version__,
            "install_mode": "legacy-package",
            "module_file": getattr(eigenmode_analysis, "__file__", None),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
        }
    else:
        runtime = runtime_info().to_dict()
    import plot_workflows

    runtime.update(
        {
            "workflow_distribution": "plot-workflows",
            "workflow_version": plot_workflows.__version__,
            "workflow_module_file": getattr(plot_workflows, "__file__", None),
        }
    )
    return runtime


def _as_config(options: TrackingWorkflowOptions | None) -> TrackingConfig:
    options = options or TrackingWorkflowOptions()
    return TrackingConfig(
        n_tracks=options.n_tracks,
        value_weights=options.value_weights,
        derivative_weights=options.derivative_weights,
        nan_cost_penalty=options.nan_cost_penalty,
        anchor_feature=options.anchor_feature,
        anchor_window=options.anchor_window,
        anchor_slope=options.anchor_slope,
        rank_penalty=options.rank_penalty,
    )


def tracking_maps(
    result: TrackingResult | GraphTrackingResult,
    node_ids: Sequence[Hashable] | None = None,
) -> tuple[dict[Hashable, dict[Hashable, int]], dict[int, dict[Hashable, Hashable]]]:
    """Build raw-candidate-to-track and track-to-raw maps from a result."""

    if node_ids is None:
        node_ids = (
            result.node_ids
            if isinstance(result, GraphTrackingResult)
            else tuple(range(len(result.candidate_ids)))
        )
    node_ids = tuple(node_ids)
    if len(node_ids) != len(result.candidate_ids):
        raise ValueError("node_ids length must match the tracking result")
    n_tracks = len(result.candidate_ids[0]) if result.candidate_ids else 0
    labels: dict[Hashable, dict[Hashable, int]] = {}
    tracked_index: dict[int, dict[Hashable, Hashable]] = {
        track_id: {} for track_id in range(n_tracks)
    }
    for node_id, row in zip(node_ids, result.candidate_ids):
        labels[node_id] = {}
        for track_id, candidate_id in enumerate(row):
            if candidate_id is None:
                continue
            labels[node_id][candidate_id] = track_id
            tracked_index[track_id][node_id] = candidate_id
    return labels, tracked_index


def _wrap_result(
    result: TrackingResult | GraphTrackingResult,
    *,
    node_ids: Sequence[Hashable],
    anchor_id: Hashable,
    algorithm: str,
) -> TrackingWorkflowResult:
    node_ids = tuple(node_ids)
    labels, tracked_index = tracking_maps(result, node_ids)
    return TrackingWorkflowResult(
        result=result,
        node_ids=node_ids,
        anchor_id=anchor_id,
        algorithm=algorithm,
        runtime=_runtime(),
        labels=labels,
        tracked_index=tracked_index,
    )


def run_path_workflow(
    points: Sequence[CandidateSet],
    coordinates: Sequence[float],
    *,
    anchor_index: int = 0,
    node_ids: Sequence[Hashable] | None = None,
    options: TrackingWorkflowOptions | None = None,
) -> TrackingWorkflowResult:
    """Track an ordered path and return compatibility maps plus provenance."""

    points = tuple(points)
    coordinates = np.asarray(coordinates, dtype=float)
    if not points:
        raise ValueError("points cannot be empty")
    if coordinates.ndim != 1 or len(coordinates) != len(points):
        raise ValueError("coordinates must be one-dimensional and match points")
    if not 0 <= anchor_index < len(points):
        raise ValueError("anchor_index is outside points")
    result = track_path(
        points,
        coordinates=coordinates,
        anchor_index=anchor_index,
        config=_as_config(options),
    )
    if node_ids is None:
        node_ids = tuple(range(len(points)))
    if len(node_ids) != len(points):
        raise ValueError("node_ids length must match points")
    return _wrap_result(
        result,
        node_ids=node_ids,
        anchor_id=node_ids[anchor_index],
        algorithm="explicit-anchor-bidirectional-path",
    )


def _centered_traversal(
    shape: tuple[int, int], anchor: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], tuple[TraversalEdge, ...]]:
    """Build a deterministic tree grown from the selected grid anchor."""

    if len(shape) != 2 or any(size <= 0 for size in shape):
        raise ValueError("shape must contain two positive dimensions")
    if any(index < 0 or index >= shape[axis] for axis, index in enumerate(anchor)):
        raise ValueError("anchor is outside the grid")
    order: list[tuple[int, int]] = []
    edges: list[TraversalEdge] = []
    visited = {anchor}
    queue = deque([anchor])
    while queue:
        source = queue.popleft()
        order.append(source)
        for axis, direction in ((0, -1), (0, 1), (1, -1), (1, 1)):
            target = list(source)
            target[axis] += direction
            target = tuple(target)
            if (
                0 <= target[0] < shape[0]
                and 0 <= target[1] < shape[1]
                and target not in visited
            ):
                visited.add(target)
                queue.append(target)
                edges.append(TraversalEdge(source, target, axis=axis, step=1.0))
    return tuple(order), tuple(edges)


def run_surface_workflow(
    points: Mapping[tuple[int, int], CandidateSet],
    shape: Sequence[int],
    *,
    anchor: tuple[int, int] | None = None,
    deltas: Sequence[float] = (1.0, 1.0),
    options: TrackingWorkflowOptions | None = None,
) -> TrackingWorkflowResult:
    """Track a rectangular candidate grid from its center anchor by default."""

    shape = tuple(int(value) for value in shape)
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError("shape must contain two positive dimensions")
    if len(deltas) != 2:
        raise ValueError("deltas must contain two values")
    anchor = anchor or (shape[0] // 2, shape[1] // 2)
    node_order, edges = _centered_traversal(shape, anchor)
    expected = set(np.ndindex(*shape))
    if set(points) != expected:
        raise ValueError("points keys must cover the complete rectangular grid")
    result = track_graph(
        points,
        node_order=node_order,
        edges=tuple(
            TraversalEdge(
                edge.source,
                edge.target,
                axis=edge.axis,
                step=abs(float(deltas[int(edge.axis)])),
            )
            for edge in edges
        ),
        anchor_id=anchor,
        config=_as_config(options),
    )
    return _wrap_result(
        result,
        node_ids=node_order,
        anchor_id=anchor,
        algorithm="center-anchor-bfs-grid",
    )


def run_graph_workflow(
    points: Mapping[Hashable, CandidateSet],
    *,
    node_order: Sequence[Hashable],
    edges: Sequence[TraversalEdge],
    anchor_id: Hashable,
    options: TrackingWorkflowOptions | None = None,
    algorithm: str = "graph",
) -> TrackingWorkflowResult:
    """Track an explicitly supplied graph while owning result/provenance maps.

    This is the escape hatch for callers whose compatibility contract already
    defines a traversal order (for example a corner-anchored
    rectangular traversal).  The consumer still owns conversion into
    :class:`CandidateSet`; the shared package owns graph orchestration.
    """

    node_order = tuple(node_order)
    if not node_order:
        raise ValueError("node_order cannot be empty")
    if anchor_id not in points:
        raise ValueError("anchor_id must identify a supplied point")
    if set(node_order) != set(points):
        raise ValueError("node_order must cover points exactly once")
    if len(set(node_order)) != len(node_order):
        raise ValueError("node_order cannot contain duplicate nodes")
    result = track_graph(
        points,
        node_order=node_order,
        edges=tuple(edges),
        anchor_id=anchor_id,
        config=_as_config(options),
    )
    return _wrap_result(
        result,
        node_ids=node_order,
        anchor_id=anchor_id,
        algorithm=algorithm,
    )


__all__ = [
    "CandidateSet",
    "TrackingWorkflowOptions",
    "TrackingWorkflowResult",
    "group_vectors_one_sided_hungarian",
    "infer_candidate_selector",
    "pick_candidate_data_stable",
    "rectilinear_traversal",
    "run_path_workflow",
    "run_surface_workflow",
    "run_graph_workflow",
    "split_disconnected_grid_tracks",
    "tracking_maps",
]
