"""Object-grid compatibility adapter.

The adapter is schema-neutral: it knows object-cell arrays and legacy return
shapes, but no Chinese columns, DataFrames, manifests, or project policy.
"""

from __future__ import annotations

from dataclasses import replace
import sys

import numpy as np

from .tracking import (
    CandidateSet,
    TrackingWorkflowOptions,
    run_graph_workflow,
    run_path_workflow,
    tracking_maps,
)
from eigenmode_analysis.tracking import (
    TraversalEdge,
    infer_candidate_selector,
    pick_candidate_data_stable,
    rectilinear_traversal,
    split_disconnected_grid_tracks,
)


def shared_tracking_provenance():
    import eigenmode_analysis

    try:
        from eigenmode_analysis.runtime import runtime_info
    except ImportError:
        return {
            "distribution": "eigenmode-analysis",
            "version": eigenmode_analysis.__version__,
            "install_mode": "legacy-package",
            "module_file": getattr(eigenmode_analysis, "__file__", None),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "direct_url": None,
            "editable_root": None,
        }
    return runtime_info().to_dict()


def _candidate_grid(components):
    dims = components[0].shape
    points = {}
    for index in np.ndindex(*dims):
        values = [component[index] for component in components]
        lengths = [len(value) if value is not None else 0 for value in values]
        if len(set(lengths)) != 1:
            raise ValueError(f"candidate component lengths differ at {index}: {lengths}")
        n_candidates = lengths[0]
        features = np.empty((n_candidates, len(values)), dtype=complex)
        for feature, value in enumerate(values):
            if n_candidates:
                features[:, feature] = value
        if not any(np.iscomplexobj(value) for value in values):
            features = features.real
        points[index] = CandidateSet(range(n_candidates), features)
    return points


def _candidate_sets(components):
    points = _candidate_grid(components)
    return [points[(index,)] for index in range(len(points))]


def _resolve_selector(additional_data, components, explicit, resolve):
    if explicit is not None or additional_data is None:
        return tuple(explicit) if explicit is not None else None
    for index in np.ndindex(*components[0].shape):
        values = components[0][index]
        n_candidates = len(values) if values is not None else 0
        if n_candidates:
            return infer_candidate_selector(additional_data[index], n_candidates, resolve=resolve)
    return None


def _validate_grid_inputs(components, deltas):
    if not components:
        raise ValueError("Z_vector_components cannot be empty")
    dims = components[0].shape
    if any(component.shape != dims for component in components):
        raise ValueError("all feature-component grids must have the same shape")
    if len(deltas) != len(dims) or any(delta == 0 for delta in deltas):
        raise ValueError("tracking requires one non-zero delta per grid axis")
    return dims


def _map_graph_result(result, dims, additional_data, components, selector, resolve):
    output = np.empty(dims, dtype=object)
    for position, point in enumerate(result.node_ids):
        output[point] = result.features[position].copy()
    if additional_data is None:
        return output
    if additional_data.shape[: len(dims)] != dims:
        raise ValueError("additional_data leading dimensions must match the candidate grid")
    selector = _resolve_selector(additional_data, components, selector, resolve)
    grouped = np.empty(dims, dtype=object)
    for position, point in enumerate(result.node_ids):
        cell = additional_data[point]
        grouped[point] = [
            pick_candidate_data_stable(cell, candidate_id, selector)
            if candidate_id is not None else None
            for candidate_id in result.candidate_ids[position]
        ]
    return output, grouped


def _apply_auto_split(result, dims, *, min_segment_points, mad_multiplier):
    node_ids = getattr(result, "node_ids", None)
    if node_ids is None:
        node_ids = tuple((index,) for index in range(len(result.candidate_ids)))
    split = split_disconnected_grid_tracks(
        node_ids, result.candidate_ids, result.features,
        shape=dims, min_segment_points=min_segment_points, mad_multiplier=mad_multiplier,
    )
    return replace(result, candidate_ids=split.candidate_ids, features=split.features)


def group_vectors_one_sided_hungarian_shared(
    Z_vector_components, deltas, value_weights, deriv_weights, max_m=None,
    initial_derivatives=None, nan_cost_penalty=1e9, additional_data=None,
    additional_selector=None, selector_resolve="deepest", auto_split_streams=False,
    min_segment_points=3, mad_multiplier=5.0, freq_rank_penalty=0.0, freq_window=0.0,
):
    del initial_derivatives
    dims = _validate_grid_inputs(Z_vector_components, deltas)
    if len(dims) != 1:
        raise ValueError("the shared compatibility adapter currently supports 1D grids only")
    points = _candidate_sets(Z_vector_components)
    n_tracks = max_m or max((point.n_candidates for point in points), default=0)
    if n_tracks <= 0:
        raise ValueError("candidate grid contains no modes")
    value_weights = np.asarray(value_weights, dtype=float)
    deriv_weights = np.asarray(deriv_weights, dtype=float)
    n_features = len(Z_vector_components)
    if value_weights.shape != (1, n_features):
        raise ValueError(f"value_weights must have shape (1, {n_features}) for the 1D adapter")
    if deriv_weights.shape != (1, 1):
        raise ValueError("deriv_weights must have shape (1, 1) for the 1D adapter")
    workflow = run_path_workflow(
        points,
        coordinates=np.arange(dims[0], dtype=float) * float(deltas[0]),
        options=TrackingWorkflowOptions(
            n_tracks=n_tracks,
            value_weights=np.full(n_features, value_weights.sum(axis=1)[0]),
            derivative_weights=np.full(n_features, deriv_weights[0, 0]),
            nan_cost_penalty=nan_cost_penalty,
            anchor_window=freq_window if freq_window > 0 else None,
            rank_penalty=freq_rank_penalty,
        ),
    )
    result = workflow.result
    if auto_split_streams:
        result = _apply_auto_split(result, dims, min_segment_points=min_segment_points, mad_multiplier=mad_multiplier)
    output = np.empty(dims, dtype=object)
    for index in range(dims[0]):
        output[index] = result.features[index].copy()
    if additional_data is None:
        return output
    selector = _resolve_selector(additional_data, Z_vector_components, additional_selector, selector_resolve)
    grouped = np.empty(dims, dtype=object)
    for index, candidate_ids in enumerate(result.candidate_ids):
        grouped[index] = [
            pick_candidate_data_stable(additional_data[index], candidate_id, selector)
            if candidate_id is not None else None
            for candidate_id in candidate_ids
        ]
    return output, grouped


def track_vectors_nd_hungarian_shared(
    Z_vector_components, deltas, value_weights, deriv_weights, max_m=None,
    nan_cost_penalty=1e9, freq_rank_penalty=0.0, freq_window=0.0,
):
    dims = _validate_grid_inputs(Z_vector_components, deltas)
    if len(dims) < 2:
        raise ValueError("use group_vectors_one_sided_hungarian_shared for 1D grids")
    n_dims = len(dims)
    n_features = len(Z_vector_components)
    value_weights = np.asarray(value_weights, dtype=float)
    deriv_weights = np.asarray(deriv_weights, dtype=float)
    if value_weights.shape != (n_dims, n_dims) or deriv_weights.shape != (n_dims, n_dims):
        raise ValueError(f"weights must have shape ({n_dims}, {n_dims})")
    if not np.allclose(deriv_weights, np.diag(np.diag(deriv_weights))):
        raise NotImplementedError("cross-axis derivative weights are not defined by the shared adapter")
    points = _candidate_grid(Z_vector_components)
    n_tracks = max_m or max((point.n_candidates for point in points.values()), default=0)
    if n_tracks <= 0:
        raise ValueError("candidate grid contains no modes")
    node_order, base_edges = rectilinear_traversal(dims, deltas)
    scales = value_weights.sum(axis=1)
    edges = tuple(
        TraversalEdge(edge.source, edge.target, axis=edge.axis, step=edge.step,
                      value_scale=float(scales[edge.axis]),
                      derivative_scale=float(deriv_weights[edge.axis, edge.axis]))
        for edge in base_edges
    )
    workflow = run_graph_workflow(
        points, node_order=node_order, edges=edges,
        anchor_id=tuple(0 for _ in dims),
        options=TrackingWorkflowOptions(
            n_tracks=n_tracks, value_weights=np.ones(n_features),
            derivative_weights=np.ones(n_features), nan_cost_penalty=nan_cost_penalty,
            anchor_window=freq_window if freq_window > 0 else None,
            rank_penalty=freq_rank_penalty,
        ),
    )
    return workflow.result


def group_vectors_nd_hungarian_shared(
    Z_vector_components, deltas, value_weights, deriv_weights, max_m=None,
    initial_derivatives=None, nan_cost_penalty=1e9, additional_data=None,
    additional_selector=None, selector_resolve="deepest", auto_split_streams=False,
    min_segment_points=3, mad_multiplier=5.0, freq_rank_penalty=0.0, freq_window=0.0,
):
    del initial_derivatives
    result = track_vectors_nd_hungarian_shared(
        Z_vector_components, deltas, value_weights, deriv_weights, max_m=max_m,
        nan_cost_penalty=nan_cost_penalty, freq_rank_penalty=freq_rank_penalty,
        freq_window=freq_window,
    )
    if auto_split_streams:
        result = _apply_auto_split(result, Z_vector_components[0].shape,
                                   min_segment_points=min_segment_points,
                                   mad_multiplier=mad_multiplier)
    return _map_graph_result(
        result, Z_vector_components[0].shape, additional_data,
        Z_vector_components, additional_selector, selector_resolve,
    )


__all__ = [
    "shared_tracking_provenance",
    "group_vectors_one_sided_hungarian_shared",
    "track_vectors_nd_hungarian_shared",
    "group_vectors_nd_hungarian_shared",
]
