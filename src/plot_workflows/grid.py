"""Object-grid preparation primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def create_data_grid(
    df: Any,
    param_keys: Sequence[str],
    z_key_list: Sequence[str],
    aggregator: Callable | None = None,
    deduplication: bool = False,
):
    """Build the historical object grid without depending on column names."""

    grid_coords: dict[str, np.ndarray] = {}
    grid_shape: list[int] = []
    value_to_index: dict[str, dict[Any, int]] = {}
    for key in param_keys:
        values = np.sort(np.asarray(df[key].unique()))
        grid_coords[key] = values
        grid_shape.append(len(values))
        value_to_index[key] = {value: index for index, value in enumerate(values)}

    z_keys = tuple(z_key_list)
    grid = np.empty(tuple(grid_shape), dtype=object)
    for index in np.ndindex(*grid_shape):
        grid[index] = [[] for _ in z_keys]

    columns = {name: index for index, name in enumerate(df.columns)}
    for row in df.itertuples(index=False):
        index = tuple(value_to_index[key][row[columns[key]]] for key in param_keys)
        cell = grid[index]
        for z_index, key in enumerate(z_keys):
            value = row[columns[key]]
            if deduplication and value in cell[z_index]:
                continue
            cell[z_index].append(value)

    if aggregator is not None:
        for index in np.ndindex(*grid_shape):
            grid[index] = aggregator(grid[index])
    return grid_coords, grid


def _resolve_cols(cols, z_keys):
    return [z_keys.index(col) if isinstance(col, str) else int(col) for col in cols]


def extract_cell(grid: np.ndarray, col, *, z_keys: Sequence[str] | None = None):
    (index,) = _resolve_cols([col], z_keys) if z_keys is not None else [int(col)]
    result = np.empty(grid.shape, dtype=object)
    for position, cell in enumerate(grid.flat):
        result.flat[position] = cell[index]
    return result


def extract_cells(grid: np.ndarray, cols, *, z_keys: Sequence[str] | None = None):
    indices = _resolve_cols(cols, z_keys) if z_keys is not None else [int(col) for col in cols]
    return [extract_cell(grid, index) for index in indices]


def group_solution(grid_coords: Mapping[str, np.ndarray], grouped: np.ndarray, freq_index: int = 1):
    """Extract one track from object cells, preserving A's missing-value policy."""

    if not isinstance(grouped, np.ndarray) or grouped.dtype != object:
        raise TypeError("grouped must be an object ndarray")
    if not isinstance(freq_index, int) or freq_index < 0:
        raise ValueError("freq_index must be a non-negative integer")
    result = np.zeros(grouped.shape, dtype=complex)
    for index in np.ndindex(*grouped.shape):
        values = grouped[index]
        if len(values) <= freq_index:
            result[index] = np.nan
        else:
            candidate = np.asarray(values[freq_index]).reshape(-1)
            result[index] = candidate[0] if candidate.size else np.nan
    return dict(grid_coords), result


def query_data_grid(grid_coords: Mapping[str, np.ndarray], grid: np.ndarray, query: Mapping[str, object]):
    """Return the cell selected by coordinate values."""

    indices = []
    for key, value in query.items():
        if key not in grid_coords:
            raise ValueError(f"parameter {key!r} is not in grid coordinates")
        matches = np.where(grid_coords[key] == value)[0]
        if not len(matches):
            raise ValueError(f"value {value!r} for parameter {key!r} was not found")
        indices.append(int(matches[0]))
    return grid[tuple(indices)]


def compress_data_axis(
    coords: Mapping[str, np.ndarray],
    data: np.ndarray,
    axis_key: str,
    aggregator=np.max,
    selection_range=None,
):
    """Reduce one coordinate axis and return coordinates/data without it."""

    keys = list(coords)
    if axis_key not in keys:
        raise KeyError(f"coordinate key {axis_key!r} is not present")
    axis = keys.index(axis_key)
    selected = data
    if selection_range is not None:
        slicers = [slice(None)] * data.ndim
        slicers[axis] = (
            slice(*selection_range) if isinstance(selection_range, tuple) else selection_range
        )
        selected = data[tuple(slicers)]
    return {key: value for key, value in coords.items() if key != axis_key}, aggregator(selected, axis=axis)


def extract_adjacent_fields(additional_grid: np.ndarray, band_index: int, z_keys: Sequence[str]):
    """Extract all fields for one grouped band as complex arrays."""

    if additional_grid.ndim == 1:
        additional_grid = additional_grid[:, np.newaxis]
    height, width = additional_grid.shape[:2]
    outputs = [np.full((height, width), np.nan, dtype=complex) for _ in z_keys]
    for i in range(height):
        for j in range(width):
            band = additional_grid[i, j][band_index]
            if band is None:
                continue
            for field_index in range(len(z_keys)):
                outputs[field_index][i, j] = band[field_index]
    return tuple(outputs)


__all__ = [
    "create_data_grid",
    "extract_cell",
    "extract_cells",
    "query_data_grid",
    "compress_data_axis",
    "group_solution",
    "extract_adjacent_fields",
]
