"""A-derived 1D/2D research pipeline orchestration.

The workflow owns grid preparation, filtering, grouping, and target extraction.
Consumers provide a tracker callback that maps their candidate cells to the
shared tracking result and may keep their own schema-specific policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .filtering import advanced_filter_eigensolution
from .grid import create_data_grid, extract_cells, group_solution


def _require_tracker(tracker, dimension: int):
    if tracker is None:
        raise ValueError(
            f"a {dimension}D tracker callback is required; provide the consumer adapter"
        )
    return tracker


def _group_1d(
    group_cells,
    filtered,
    *,
    max_num,
    deltas,
    value_weight,
    deriv_weight,
    nan_cost_penalty,
    auto_split_streams,
    mad_multiplier,
    freq_window,
    freq_rank_penalty,
    method,
    tracker,
):
    tracker = _require_tracker(tracker, 1)
    n_features = len(group_cells)
    return tracker(
        group_cells,
        deltas,
        additional_data=filtered,
        value_weights=np.full((1, n_features), value_weight),
        deriv_weights=np.full((1, n_features), deriv_weight),
        max_m=max_num,
        nan_cost_penalty=nan_cost_penalty,
        auto_split_streams=auto_split_streams,
        mad_multiplier=mad_multiplier,
        freq_window=freq_window,
        freq_rank_penalty=freq_rank_penalty,
    )


def _finish_targets(coords, grouped, max_num):
    targets = []
    for frequency_index in range(max_num):
        coords, target = group_solution(coords, grouped, freq_index=frequency_index)
        targets.append(target)
    return coords, targets


def run_pipeline_1d(
    df: Any,
    param_keys: Sequence[str],
    z_keys: Sequence[str],
    fixed_params: Mapping[str, object],
    filter_conditions: Mapping[str, Mapping[str, object]],
    max_num: int,
    *,
    group_cols=(0,),
    deltas=(1e-3,),
    nan_cost_penalty=1e1,
    auto_split_streams=False,
    value_weight=1.0,
    deriv_weight=1e-1,
    mad_multiplier=5.0,
    freq_window=0.0,
    freq_rank_penalty=0.0,
    tracker: Callable | None = None,
):
    """Run the standard 1D grid/filter/group/target workflow."""

    coords, grid = create_data_grid(df, param_keys, z_keys, deduplication=False)
    return run_pipeline_1d_from_grid(
        coords,
        grid,
        z_keys,
        fixed_params,
        filter_conditions,
        max_num,
        group_cols=group_cols,
        deltas=deltas,
        nan_cost_penalty=nan_cost_penalty,
        auto_split_streams=auto_split_streams,
        value_weight=value_weight,
        deriv_weight=deriv_weight,
        mad_multiplier=mad_multiplier,
        freq_window=freq_window,
        freq_rank_penalty=freq_rank_penalty,
        tracker=tracker,
    )


def run_pipeline_1d_from_grid(
    grid_coords,
    grid,
    z_keys,
    fixed_params,
    filter_conditions,
    max_num,
    *,
    group_cols=(0,),
    deltas=(1e-3,),
    nan_cost_penalty=1e1,
    auto_split_streams=False,
    value_weight=1.0,
    deriv_weight=1e-1,
    mad_multiplier=5.0,
    freq_window=0.0,
    freq_rank_penalty=0.0,
    tracker: Callable | None = None,
):
    coords, filtered, _ = advanced_filter_eigensolution(
        grid_coords, grid, z_keys, fixed_params, filter_conditions
    )
    cols = [z_keys.index(col) if isinstance(col, str) else int(col) for col in group_cols]
    grouped, additional = _group_1d(
        extract_cells(filtered, cols),
        filtered,
        max_num=max_num,
        deltas=deltas,
        value_weight=value_weight,
        deriv_weight=deriv_weight,
        nan_cost_penalty=nan_cost_penalty,
        auto_split_streams=auto_split_streams,
        mad_multiplier=mad_multiplier,
        freq_window=freq_window,
        freq_rank_penalty=freq_rank_penalty,
        method="hungarian",
        tracker=tracker,
    )
    coords, targets = _finish_targets(coords, grouped, max_num)
    return coords, targets, additional, filtered


def run_pipeline_2d(
    df: Any,
    param_keys: Sequence[str],
    z_keys: Sequence[str],
    fixed_params: Mapping[str, object],
    filter_conditions: Mapping[str, Mapping[str, object]],
    max_num: int,
    *,
    group_cols=(0,),
    deltas=(1e-3, 1e-3),
    nan_cost_penalty=1e1,
    auto_split_streams=False,
    value_weight=1.0,
    deriv_weight=1e-1,
    mad_multiplier=5.0,
    freq_window=0.0,
    freq_rank_penalty=0.0,
    group_use_real=True,
    tracker: Callable | None = None,
    polynomial_tracker: Callable | None = None,
    poly_lambda=1.0,
):
    """Run the standard 2D grid/filter/group/target workflow."""

    coords, grid = create_data_grid(df, param_keys, z_keys, deduplication=False)
    return run_pipeline_2d_from_grid(
        coords,
        grid,
        z_keys,
        fixed_params,
        filter_conditions,
        max_num,
        group_cols=group_cols,
        deltas=deltas,
        nan_cost_penalty=nan_cost_penalty,
        auto_split_streams=auto_split_streams,
        value_weight=value_weight,
        deriv_weight=deriv_weight,
        mad_multiplier=mad_multiplier,
        freq_window=freq_window,
        freq_rank_penalty=freq_rank_penalty,
        group_use_real=group_use_real,
        tracker=tracker,
        polynomial_tracker=polynomial_tracker,
        poly_lambda=poly_lambda,
    )


def run_pipeline_2d_from_grid(
    grid_coords,
    grid,
    z_keys,
    fixed_params,
    filter_conditions,
    max_num,
    *,
    group_cols=(0,),
    deltas=(1e-3, 1e-3),
    nan_cost_penalty=1e1,
    auto_split_streams=False,
    value_weight=1.0,
    deriv_weight=1e-1,
    mad_multiplier=5.0,
    freq_window=0.0,
    freq_rank_penalty=0.0,
    group_use_real=True,
    tracker: Callable | None = None,
    polynomial_tracker: Callable | None = None,
    poly_lambda=1.0,
):
    coords, filtered, _ = advanced_filter_eigensolution(
        grid_coords, grid, z_keys, fixed_params, filter_conditions
    )
    cols = [z_keys.index(col) if isinstance(col, str) else int(col) for col in group_cols]
    if polynomial_tracker is not None:
        grouped, additional = polynomial_tracker(
            extract_cells(filtered, cols)[0], filtered, z_keys, max_num, poly_lambda
        )
    else:
        tracker = _require_tracker(tracker, 2)
        grouping_grid = filtered.real if group_use_real else filtered
        grouped, additional = tracker(
            extract_cells(grouping_grid, cols),
            deltas,
            additional_data=filtered,
            value_weights=np.eye(len(deltas)) * value_weight,
            deriv_weights=np.eye(len(deltas)) * deriv_weight,
            max_m=max_num,
            nan_cost_penalty=nan_cost_penalty,
            auto_split_streams=auto_split_streams,
            mad_multiplier=mad_multiplier,
            freq_window=freq_window,
            freq_rank_penalty=freq_rank_penalty,
        )
    coords, targets = _finish_targets(coords, grouped, max_num)
    return coords, targets, additional, filtered


__all__ = [
    "run_pipeline_1d",
    "run_pipeline_1d_from_grid",
    "run_pipeline_2d",
    "run_pipeline_2d_from_grid",
]
