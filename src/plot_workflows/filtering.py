"""Grid slicing and candidate filtering extracted from the A workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def advanced_filter_eigensolution(
    grid_coords: Mapping[str, np.ndarray],
    grid: np.ndarray,
    z_keys: Sequence[str],
    fixed_params: Mapping[str, object] | None = None,
    filter_conditions: Mapping[str, Mapping[str, object]] | None = None,
):
    """Slice fixed axes and apply AND conditions to every candidate cell."""

    fixed_params = dict(fixed_params or {})
    filter_conditions = dict(filter_conditions or {})
    param_keys = list(grid_coords)
    slices = [slice(None)] * grid.ndim
    for key, value in fixed_params.items():
        if key not in param_keys:
            raise ValueError(f"fixed parameter {key!r} is not in grid coordinates")
        matches = np.where(np.isclose(grid_coords[key], value))[0]
        if not len(matches):
            raise ValueError(f"value {value!r} was not found for {key!r}")
        slices[param_keys.index(key)] = int(matches[0])

    sliced = grid[tuple(slices)]
    filtered = np.empty(sliced.shape, dtype=object)
    minimum = float("inf")
    keys = tuple(z_keys)
    for index in np.ndindex(*sliced.shape):
        elements = sliced[index]
        if not isinstance(elements, list) or not elements:
            filtered[index] = []
            minimum = min(minimum, 0)
            continue
        array = np.asarray(elements, dtype=object)
        if array.ndim != 2 or array.shape[1] == 0:
            mask = np.zeros(array.shape[1] if array.ndim == 2 else 0, dtype=bool)
        else:
            mask = np.ones(array.shape[1], dtype=bool)
            for key, conditions in filter_conditions.items():
                if key not in keys:
                    raise ValueError(f"filter key {key!r} is not in z_keys")
                row = np.asarray(
                    [np.real(item) if np.iscomplexobj(item) else item for item in array[keys.index(key)]],
                )
                for operator, threshold in conditions.items():
                    if operator == ">":
                        mask &= row > threshold
                    elif operator == "<":
                        mask &= row < threshold
                    elif operator == "==":
                        mask &= row == threshold
                    elif operator == ">=":
                        mask &= row >= threshold
                    elif operator == "<=":
                        mask &= row <= threshold
                    else:
                        raise ValueError(f"unsupported filter operator {operator!r}")
        filtered[index] = array[:, mask].tolist()
        minimum = min(minimum, int(mask.sum()))
    return (
        {key: values for key, values in grid_coords.items() if key not in fixed_params},
        filtered,
        0 if minimum == float("inf") else minimum,
    )


__all__ = ["advanced_filter_eigensolution"]
