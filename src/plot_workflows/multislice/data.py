from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import pickle
import warnings

import numpy as np
import pandas as pd

from ..filtering import advanced_filter_eigensolution
from ..grid import create_data_grid, extract_adjacent_fields, group_solution


_READ_DATA_HOOK = None


def configure_multislice_hooks(*, read_data_cached=None):
    global _READ_DATA_HOOK
    _READ_DATA_HOOK = read_data_cached


def read_data_cached(filepath, sep=None, **read_csv_kwargs):
    """Read a delimited source; consumers may wrap this with a cache hook."""

    if _READ_DATA_HOOK is not None:
        return _READ_DATA_HOOK(filepath, sep=sep, **read_csv_kwargs)
    return pd.read_csv(filepath, sep=sep, **read_csv_kwargs)


def convert_complex(value):
    return complex(value.replace("i", "j")) if isinstance(value, str) else value


def norm_freq(freq, period):
    return freq / (299792458.0 / period)


ArrayLike = Union[np.ndarray, Sequence[float]]
TransformFn = Callable[[np.ndarray], np.ndarray]
MaskFn = Callable[["SliceDataset", np.ndarray], np.ndarray]
FieldBuilderFn = Callable[..., Dict[str, np.ndarray]]


# =============================================================================
# 1. Standard quasi-3D data container
# =============================================================================

@dataclass
class SliceDataset:
    """
    Standard data container for one 2D slice in a quasi-3D parameter space.

    Requirement
    -----------
    Each field must satisfy:

        field.shape == (len(coords[x_key]), len(coords[y_key]))

    Different slices are allowed to have different shapes.
    """

    slice_key: str
    slice_value: Any
    coords: Mapping[str, ArrayLike]
    fields: Mapping[str, ArrayLike]
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def get_coord(self, key: str) -> np.ndarray:
        if key not in self.coords:
            raise KeyError(f"Coordinate key {key!r} not found.")
        return _as_1d_array(self.coords[key], name=f"coords[{key!r}]")

    def get_field(self, key: str) -> np.ndarray:
        if key not in self.fields:
            raise KeyError(f"Field key {key!r} not found.")
        return np.asarray(self.fields[key])

    def validate(
        self,
        x_key: str,
        y_key: str,
        field_keys: Union[str, Sequence[str]],
    ) -> None:
        validate_slice_dataset(
            self,
            x_key=x_key,
            y_key=y_key,
            field_keys=field_keys,
        )


@dataclass(frozen=True)
class SliceBuildSpec:
    """Compact per-slice build settings."""

    slice_value: Any
    band_index: int
    max_num: int


@dataclass
class SliceShapeReport:
    n_slices: int
    x_key: str
    y_key: str
    field_keys: List[str]

    x_lengths: List[int]
    y_lengths: List[int]
    field_shapes: Dict[str, List[Tuple[int, int]]]

    consistent_xy_lengths: bool
    consistent_field_shapes: bool
    consistent_shape: bool
    consistent_xy_values: bool

    common_shape: Optional[Tuple[int, int]] = None

    def summary(self) -> str:
        lines = [
            "Slice shape report:",
            f"  n_slices = {self.n_slices}",
            f"  x_key = {self.x_key}",
            f"  y_key = {self.y_key}",
            f"  field_keys = {self.field_keys}",
            f"  consistent_xy_lengths = {self.consistent_xy_lengths}",
            f"  consistent_field_shapes = {self.consistent_field_shapes}",
            f"  consistent_shape = {self.consistent_shape}",
            f"  consistent_xy_values = {self.consistent_xy_values}",
            f"  common_shape = {self.common_shape}",
        ]
        return "\n".join(lines)


# =============================================================================
# 2. Basic utilities
# =============================================================================

def _as_1d_array(values: ArrayLike, name: str = "array") -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, but got shape {arr.shape}.")
    return arr


def _as_2d_array(values: ArrayLike, name: str = "array") -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {arr.shape}.")
    return arr


def _to_python_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return value


def _normalize_field_keys(field_keys: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(field_keys, str):
        return [field_keys]
    return list(field_keys)


def _validate_per_slice_sequence(
    values: Sequence[Any],
    *,
    name: str,
    n_slices: int,
) -> List[Any]:
    """
    Require a per-slice sequence argument.

    No scalar backward compatibility is provided.
    """

    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence with length {n_slices}, not a string."
        )

    values_list: List[Any] = list(values)

    if len(values_list) != n_slices:
        raise ValueError(
            f"{name} must have the same length as selected_slice_values. "
            f"Expected length {n_slices}, got {len(values_list)}."
        )

    return values_list


def _normalize_slice_build_specs(
    selected_slice_values: Sequence[Any],
    band_index: Sequence[int],
    max_num: Sequence[int],
) -> List[SliceBuildSpec]:
    slice_values = list(selected_slice_values)

    band_index_list = _validate_per_slice_sequence(
        band_index,
        name="band_index",
        n_slices=len(slice_values),
    )

    max_num_list = _validate_per_slice_sequence(
        max_num,
        name="max_num",
        n_slices=len(slice_values),
    )

    return [
        SliceBuildSpec(
            slice_value=slice_value,
            band_index=int(band_index_list[i]),
            max_num=int(max_num_list[i]),
        )
        for i, slice_value in enumerate(slice_values)
    ]


def _sort_slices(slice_datasets: Sequence[SliceDataset]) -> List[SliceDataset]:
    try:
        return sorted(slice_datasets, key=lambda ds: float(ds.slice_value))
    except Exception:
        return list(slice_datasets)


def _default_score_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)

    if np.iscomplexobj(values):
        return np.abs(values).astype(float)

    try:
        return values.astype(float)
    except Exception as exc:
        raise TypeError(
            "Values cannot be converted to real float scores. "
            "Please provide value_transform."
        ) from exc


def _make_score_values(
    raw_values: np.ndarray,
    value_transform: Optional[TransformFn],
) -> np.ndarray:
    if value_transform is None:
        return _default_score_values(raw_values)

    score = np.asarray(value_transform(raw_values))

    if np.iscomplexobj(score):
        score = np.abs(score)

    try:
        return score.astype(float)
    except Exception as exc:
        raise TypeError(
            "value_transform must return real numeric values, or complex values "
            "that can be converted through abs()."
        ) from exc


def _make_objective_values(
    score_values: np.ndarray,
    mode: str,
    target_value: Optional[float],
) -> np.ndarray:
    mode = mode.lower()

    if mode == "max":
        return score_values

    if mode == "min":
        return score_values

    if mode in {"closest", "nearest"}:
        if target_value is None:
            raise ValueError("target_value must be provided when mode='closest'.")
        return np.abs(score_values - target_value)

    raise ValueError("mode must be 'max', 'min', or 'closest'.")


def _select_best_index_from_objective(
    objective: np.ndarray,
    mode: str,
) -> Optional[Tuple[int, int]]:
    objective = np.asarray(objective, dtype=float)
    finite = np.isfinite(objective)

    if not np.any(finite):
        return None

    mode = mode.lower()

    if mode == "max":
        safe = np.where(finite, objective, -np.inf)
        flat_index = int(np.argmax(safe))
    elif mode in {"min", "closest", "nearest"}:
        safe = np.where(finite, objective, np.inf)
        flat_index = int(np.argmin(safe))
    else:
        raise ValueError("mode must be 'max', 'min', or 'closest'.")

    ix, iy = np.unravel_index(flat_index, objective.shape)
    return int(ix), int(iy)


def _infer_delta(values: np.ndarray) -> float:
    values = np.asarray(values)

    try:
        values = values.astype(float)
    except Exception:
        return 1.0

    values = np.unique(values[np.isfinite(values)])

    if len(values) <= 1:
        return 1.0

    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 1.0

    return float(np.median(diffs))


def _infer_deltas_for_xy(
    coords: Mapping[str, ArrayLike],
    x_key: str,
    y_key: str,
) -> Tuple[float, float]:
    return (
        _infer_delta(np.asarray(coords[x_key])),
        _infer_delta(np.asarray(coords[y_key])),
    )


def _make_default_weights(ndim: int = 2) -> np.ndarray:
    return np.ones((ndim, ndim), dtype=float)


# =============================================================================
# 3. Save / load prepared plotting data
# =============================================================================

def save_slice_datasets_pkl(
    slice_datasets: Sequence[SliceDataset],
    path: str,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save prepared quasi-3D slice data.

    pkl is recommended because each slice may have different 2D array shapes.
    """

    payload = {
        "format": "multislice_slice_datasets",
        "version": 1,
        "slice_datasets": list(slice_datasets),
        "extra": dict(extra or {}),
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_slice_datasets_pkl(
    path: str,
    *,
    return_extra: bool = False,
) -> Union[List[SliceDataset], Tuple[List[SliceDataset], Dict[str, Any]]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, list):
        datasets = payload
        extra = {}
    else:
        datasets = payload["slice_datasets"]
        extra = payload.get("extra", {})

    if return_extra:
        return datasets, extra

    return datasets


def save_trajectory_csv(
    trajectory_df: pd.DataFrame,
    path: str,
    *,
    index: bool = False,
) -> None:
    trajectory_df.to_csv(path, index=index)


def load_trajectory_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# =============================================================================
# 4. Shape inspection and stacking
# =============================================================================

def validate_slice_dataset(
    dataset: SliceDataset,
    x_key: str,
    y_key: str,
    field_keys: Union[str, Sequence[str]],
) -> None:
    field_keys = _normalize_field_keys(field_keys)

    x = dataset.get_coord(x_key)
    y = dataset.get_coord(y_key)
    expected_shape = (len(x), len(y))

    for field_key in field_keys:
        field = _as_2d_array(
            dataset.get_field(field_key),
            name=f"fields[{field_key!r}]",
        )

        if field.shape != expected_shape:
            raise ValueError(
                f"Field {field_key!r} in slice {dataset.slice_value!r} has shape "
                f"{field.shape}, but expected {expected_shape}."
            )


def inspect_slice_series(
    slice_datasets: Sequence[SliceDataset],
    x_key: str,
    y_key: str,
    field_keys: Union[str, Sequence[str]],
    check_xy_values: bool = True,
) -> SliceShapeReport:
    """
    Check whether a slice series can be treated as a regular 3D array.

    consistent_shape=True means all selected fields in all slices have the same
    2D shape.

    consistent_xy_values=True additionally means all slices have exactly the same
    x and y coordinate arrays.
    """

    if len(slice_datasets) == 0:
        raise ValueError("slice_datasets is empty.")

    field_keys = _normalize_field_keys(field_keys)

    x_lengths: List[int] = []
    y_lengths: List[int] = []
    field_shapes: Dict[str, List[Tuple[int, int]]] = {key: [] for key in field_keys}

    first_x = None
    first_y = None
    xy_values_equal = True

    for ds in slice_datasets:
        ds.validate(x_key=x_key, y_key=y_key, field_keys=field_keys)

        x = ds.get_coord(x_key)
        y = ds.get_coord(y_key)

        x_lengths.append(len(x))
        y_lengths.append(len(y))

        if first_x is None:
            first_x = x
            first_y = y
        elif check_xy_values:
            if len(x) != len(first_x) or len(y) != len(first_y):
                xy_values_equal = False
            else:
                if not np.array_equal(x, first_x) or not np.array_equal(y, first_y):
                    xy_values_equal = False

        for field_key in field_keys:
            field_shapes[field_key].append(tuple(ds.get_field(field_key).shape))

    consistent_xy_lengths = len(set(zip(x_lengths, y_lengths))) == 1

    all_shapes: List[Tuple[int, int]] = []
    for key in field_keys:
        all_shapes.extend(field_shapes[key])

    consistent_field_shapes = len(set(all_shapes)) == 1
    consistent_shape = consistent_xy_lengths and consistent_field_shapes

    return SliceShapeReport(
        n_slices=len(slice_datasets),
        x_key=x_key,
        y_key=y_key,
        field_keys=field_keys,
        x_lengths=x_lengths,
        y_lengths=y_lengths,
        field_shapes=field_shapes,
        consistent_xy_lengths=consistent_xy_lengths,
        consistent_field_shapes=consistent_field_shapes,
        consistent_shape=consistent_shape,
        consistent_xy_values=xy_values_equal if check_xy_values else False,
        common_shape=all_shapes[0] if consistent_shape else None,
    )


def stack_field(
    slice_datasets: Sequence[SliceDataset],
    field_key: str,
    x_key: str,
    y_key: str,
) -> np.ndarray:
    """
    Stack one field into a regular 3D array.

    Return shape:

        (n_slices, n_x, n_y)
    """

    report = inspect_slice_series(
        slice_datasets,
        x_key=x_key,
        y_key=y_key,
        field_keys=field_key,
        check_xy_values=False,
    )

    if not report.consistent_shape:
        raise ValueError(
            "Cannot stack fields because slice shapes are inconsistent.\n"
            + report.summary()
        )

    return np.stack(
        [np.asarray(ds.get_field(field_key)) for ds in slice_datasets],
        axis=0,
    )


# =============================================================================
# 5. Extrema trajectory data generation
# =============================================================================

def find_extrema_2d(
    field: np.ndarray,
    x_values: ArrayLike,
    y_values: ArrayLike,
    *,
    mode: str = "max",
    value_transform: Optional[TransformFn] = np.abs,
    target_value: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find one extremum point in a 2D field.

    mode:
        'max', 'min', or 'closest'.

    value_transform:
        Transform raw field values before comparison.
    """

    field = _as_2d_array(field, name="field")
    x = _as_1d_array(x_values, name="x_values")
    y = _as_1d_array(y_values, name="y_values")

    expected_shape = (len(x), len(y))
    if field.shape != expected_shape:
        raise ValueError(
            f"field.shape must be {expected_shape}, got {field.shape}."
        )

    score_values = _make_score_values(field, value_transform=value_transform)
    objective = _make_objective_values(
        score_values,
        mode=mode,
        target_value=target_value,
    )

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != field.shape:
            raise ValueError(
                f"mask.shape must be {field.shape}, got {mask.shape}."
            )
        objective = np.where(mask, objective, np.nan)

    best = _select_best_index_from_objective(objective, mode=mode)
    if best is None:
        return None

    ix, iy = best

    return {
        "ix": int(ix),
        "iy": int(iy),
        "x": _to_python_scalar(x[ix]),
        "y": _to_python_scalar(y[iy]),
        "field_value": _to_python_scalar(field[ix, iy]),
        "score_value": float(score_values[ix, iy]),
        "objective_value": float(objective[ix, iy]),
        "mode": mode,
    }


def _prepare_mask_stack(
    mask: Optional[Union[np.ndarray, MaskFn]],
    field_stack: np.ndarray,
) -> Optional[np.ndarray]:
    if mask is None:
        return None

    if callable(mask):
        raise TypeError("Callable mask requires loop mode.")

    mask_arr = np.asarray(mask, dtype=bool)

    if mask_arr.ndim == 2:
        if mask_arr.shape != field_stack.shape[1:]:
            raise ValueError(
                f"2D mask must have shape {field_stack.shape[1:]}, "
                f"got {mask_arr.shape}."
            )
        return np.broadcast_to(mask_arr, field_stack.shape)

    if mask_arr.ndim == 3:
        if mask_arr.shape != field_stack.shape:
            raise ValueError(
                f"3D mask must have shape {field_stack.shape}, "
                f"got {mask_arr.shape}."
            )
        return mask_arr

    raise ValueError("mask must be None, 2D array, 3D array, or callable.")


def find_extrema_trajectory(
    slice_datasets: Sequence[SliceDataset],
    *,
    x_key: str,
    y_key: str,
    slice_key: Optional[str] = None,
    field_key: str,
    mode: str = "max",
    value_transform: Optional[TransformFn] = np.abs,
    target_value: Optional[float] = None,
    mask: Optional[Union[np.ndarray, MaskFn]] = None,
    vectorized: Union[str, bool] = "auto",
    keep_invalid: bool = False,
    sort_slices: bool = True,
    include_metadata_keys: Optional[Sequence[str]] = None,
    include_field_keys: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate trajectory data from prepared slice datasets.

    This function is part of data processing, not plotting.

    It automatically chooses:

    1. vectorized path:
        if all slices have the same 2D shape and mask is not callable.

    2. loop path:
        if slice shapes are inconsistent or mask is callable.
    """

    if len(slice_datasets) == 0:
        raise ValueError("slice_datasets is empty.")

    datasets = _sort_slices(slice_datasets) if sort_slices else list(slice_datasets)

    report = inspect_slice_series(
        datasets,
        x_key=x_key,
        y_key=y_key,
        field_keys=field_key,
        check_xy_values=False,
    )

    if verbose:
        print(report.summary())

    can_vectorize = report.consistent_shape and not callable(mask)

    if vectorized is True and not can_vectorize:
        raise ValueError(
            "vectorized=True was requested, but vectorized path is impossible.\n"
            + report.summary()
        )

    use_vectorized = can_vectorize and vectorized != False

    if use_vectorized:
        try:
            trajectory_df = _find_extrema_trajectory_vectorized(
                datasets,
                x_key=x_key,
                y_key=y_key,
                slice_key=slice_key,
                field_key=field_key,
                mode=mode,
                value_transform=value_transform,
                target_value=target_value,
                mask=mask,
                keep_invalid=keep_invalid,
                include_metadata_keys=include_metadata_keys,
            )
            if include_field_keys:
                trajectory_df = _sample_trajectory_fields(
                    trajectory_df,
                    datasets,
                    slice_key=slice_key or datasets[0].slice_key,
                    field_keys=include_field_keys,
                )
            return trajectory_df
        except Exception as exc:
            if vectorized is True:
                raise

            warnings.warn(
                "Vectorized extrema search failed. Falling back to loop mode. "
                f"Original error: {exc}",
                RuntimeWarning,
            )

    trajectory_df = _find_extrema_trajectory_loop(
        datasets,
        x_key=x_key,
        y_key=y_key,
        slice_key=slice_key,
        field_key=field_key,
        mode=mode,
        value_transform=value_transform,
        target_value=target_value,
        mask=mask,
        keep_invalid=keep_invalid,
        include_metadata_keys=include_metadata_keys,
    )

    if include_field_keys:
        trajectory_df = _sample_trajectory_fields(
            trajectory_df,
            datasets,
            slice_key=slice_key or datasets[0].slice_key,
            field_keys=include_field_keys,
        )

    return trajectory_df


def _find_extrema_trajectory_vectorized(
    slice_datasets: Sequence[SliceDataset],
    *,
    x_key: str,
    y_key: str,
    slice_key: Optional[str],
    field_key: str,
    mode: str,
    value_transform: Optional[TransformFn],
    target_value: Optional[float],
    mask: Optional[Union[np.ndarray, MaskFn]],
    keep_invalid: bool,
    include_metadata_keys: Optional[Sequence[str]],
) -> pd.DataFrame:
    field_stack = stack_field(
        slice_datasets,
        field_key=field_key,
        x_key=x_key,
        y_key=y_key,
    )

    score_stack = _make_score_values(field_stack, value_transform=value_transform)
    objective_stack = _make_objective_values(
        score_stack,
        mode=mode,
        target_value=target_value,
    )

    mask_stack = _prepare_mask_stack(mask, field_stack)
    if mask_stack is not None:
        objective_stack = np.where(mask_stack, objective_stack, np.nan)

    n_slices, nx, ny = field_stack.shape
    flat_objective = objective_stack.reshape(n_slices, nx * ny)

    finite = np.isfinite(flat_objective)
    row_valid = np.any(finite, axis=1)

    mode_lower = mode.lower()

    if mode_lower == "max":
        safe = np.where(finite, flat_objective, -np.inf)
        flat_best = np.argmax(safe, axis=1)
    elif mode_lower in {"min", "closest", "nearest"}:
        safe = np.where(finite, flat_objective, np.inf)
        flat_best = np.argmin(safe, axis=1)
    else:
        raise ValueError("mode must be 'max', 'min', or 'closest'.")

    rows: List[Dict[str, Any]] = []

    for s_idx, ds in enumerate(slice_datasets):
        actual_slice_key = slice_key or ds.slice_key

        if not row_valid[s_idx]:
            if keep_invalid:
                rows.append(
                    {
                        actual_slice_key: ds.slice_value,
                        x_key: np.nan,
                        y_key: np.nan,
                        "ix": np.nan,
                        "iy": np.nan,
                        "field_value": np.nan,
                        "score_value": np.nan,
                        "objective_value": np.nan,
                        "mode": mode,
                        "valid": False,
                    }
                )
            continue

        flat_idx = int(flat_best[s_idx])
        ix = flat_idx // ny
        iy = flat_idx % ny

        x_values = ds.get_coord(x_key)
        y_values = ds.get_coord(y_key)

        row = {
            actual_slice_key: ds.slice_value,
            x_key: _to_python_scalar(x_values[ix]),
            y_key: _to_python_scalar(y_values[iy]),
            "ix": int(ix),
            "iy": int(iy),
            "field_value": _to_python_scalar(field_stack[s_idx, ix, iy]),
            "score_value": float(score_stack[s_idx, ix, iy]),
            "objective_value": float(objective_stack[s_idx, ix, iy]),
            "mode": mode,
            "valid": True,
        }

        if include_metadata_keys:
            for key in include_metadata_keys:
                row[key] = ds.metadata.get(key, None)

        rows.append(row)

    return pd.DataFrame(rows)


def _find_extrema_trajectory_loop(
    slice_datasets: Sequence[SliceDataset],
    *,
    x_key: str,
    y_key: str,
    slice_key: Optional[str],
    field_key: str,
    mode: str,
    value_transform: Optional[TransformFn],
    target_value: Optional[float],
    mask: Optional[Union[np.ndarray, MaskFn]],
    keep_invalid: bool,
    include_metadata_keys: Optional[Sequence[str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for s_idx, ds in enumerate(slice_datasets):
        actual_slice_key = slice_key or ds.slice_key

        field = ds.get_field(field_key)
        x_values = ds.get_coord(x_key)
        y_values = ds.get_coord(y_key)

        if callable(mask):
            mask_i = np.asarray(mask(ds, field), dtype=bool)
        elif mask is None:
            mask_i = None
        else:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.ndim == 2:
                mask_i = mask_arr
            elif mask_arr.ndim == 3:
                mask_i = mask_arr[s_idx]
            else:
                raise ValueError("mask must be None, 2D array, 3D array, or callable.")

        result = find_extrema_2d(
            field,
            x_values,
            y_values,
            mode=mode,
            value_transform=value_transform,
            target_value=target_value,
            mask=mask_i,
        )

        if result is None:
            if keep_invalid:
                rows.append(
                    {
                        actual_slice_key: ds.slice_value,
                        x_key: np.nan,
                        y_key: np.nan,
                        "ix": np.nan,
                        "iy": np.nan,
                        "field_value": np.nan,
                        "score_value": np.nan,
                        "objective_value": np.nan,
                        "mode": mode,
                        "valid": False,
                    }
                )
            continue

        row = {
            actual_slice_key: ds.slice_value,
            x_key: result["x"],
            y_key: result["y"],
            "ix": result["ix"],
            "iy": result["iy"],
            "field_value": result["field_value"],
            "score_value": result["score_value"],
            "objective_value": result["objective_value"],
            "mode": result["mode"],
            "valid": True,
        }

        if include_metadata_keys:
            for key in include_metadata_keys:
                row[key] = ds.metadata.get(key, None)

        rows.append(row)

    return pd.DataFrame(rows)


def _match_slice_dataset(
    slice_datasets: Sequence[SliceDataset],
    slice_value: Any,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> SliceDataset:
    try:
        target = float(slice_value)

        best_ds: Optional[SliceDataset] = None
        best_dist = np.inf

        for ds in slice_datasets:
            try:
                value = float(ds.slice_value)
            except Exception:
                continue

            dist = abs(value - target)

            if dist < best_dist:
                best_dist = dist
                best_ds = ds

        if best_ds is not None and np.isclose(
            float(best_ds.slice_value),
            target,
            atol=atol,
            rtol=rtol,
        ):
            return best_ds

    except Exception:
        pass

    for ds in slice_datasets:
        if ds.slice_value == slice_value:
            return ds

    raise KeyError(f"No SliceDataset matches slice_value={slice_value!r}.")


def _sample_trajectory_fields(
    trajectory_df: pd.DataFrame,
    slice_datasets: Sequence[SliceDataset],
    *,
    slice_key: str,
    field_keys: Sequence[str],
    ix_key: str = "ix",
    iy_key: str = "iy",
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> pd.DataFrame:
    if len(field_keys) == 0:
        return trajectory_df

    required = [slice_key, ix_key, iy_key]
    column_names = set(trajectory_df.columns)
    missing = [key for key in required if key not in column_names]
    if missing:
        raise KeyError(
            "trajectory_df must contain slice and index columns before field sampling. "
            f"Missing: {missing}"
        )

    out = trajectory_df.copy()

    for field_key in field_keys:
        sampled: List[Any] = []

        for _, row in out.iterrows():
            if not np.isfinite(row[ix_key]) or not np.isfinite(row[iy_key]):
                sampled.append(np.nan)
                continue

            ds = _match_slice_dataset(
                slice_datasets,
                row[slice_key],
                atol=atol,
                rtol=rtol,
            )

            field = ds.get_field(field_key)
            ix = int(row[ix_key])
            iy = int(row[iy_key])

            if ix < 0 or ix >= field.shape[0] or iy < 0 or iy >= field.shape[1]:
                sampled.append(np.nan)
                continue

            sampled.append(field[ix, iy])

        out[field_key] = np.asarray(sampled)

    return out


# =============================================================================
# 6. Eigensolution CSV data-preparation workflow
# =============================================================================

def default_preprocess_eigensolution_dataframe(
    df: pd.DataFrame,
    *,
    period: float = 500,
    eigenfreq_key: str = "特征频率 (THz)",
    freq_key: str = "频率 (Hz)",
    complex_field_keys: Sequence[str] = (
        "up_cx (V/m)",
        "up_cy (V/m)",
        "down_cx (V/m)",
        "down_cy (V/m)",
    ),
) -> pd.DataFrame:
    """
    Default preprocessing matching your current script.
    """

    df = df.copy()

    if eigenfreq_key in df.columns:
        df[eigenfreq_key] = (
            df[eigenfreq_key]
            .apply(convert_complex)
            .apply(norm_freq, period=period * 1e-9 * 1e12)
        )

    if freq_key in df.columns:
        df[freq_key] = df[freq_key].apply(
            norm_freq,
            period=period * 1e-9,
        )

    for key in complex_field_keys:
        if key in df.columns:
            df[key] = df[key].apply(convert_complex)

    return df


def load_and_preprocess_eigensolution_csvs(
    csv_paths: Union[str, Sequence[str]],
    *,
    sep: str = "\t",
    period: float = 500,
    default_preprocess: bool = True,
    preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    read_csv_kwargs: Optional[Mapping[str, Any]] = None,
    concat_csvs: bool = True,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load and preprocess one or multiple CSV files.

    If concat_csvs=True:
        Return one concatenated DataFrame.

    If concat_csvs=False:
        Return a list of DataFrames.
    """

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    read_csv_kwargs = dict(read_csv_kwargs or {})

    dfs: List[pd.DataFrame] = []

    for path in csv_paths:
        df = read_data_cached(path, sep=sep, **read_csv_kwargs)

        if default_preprocess:
            df = default_preprocess_eigensolution_dataframe(
                df,
                period=period,
            )

        if preprocess_fn is not None:
            df = preprocess_fn(df)

        df.attrs["data_path"] = path
        dfs.append(df)

    if concat_csvs:
        out = pd.concat(dfs, ignore_index=True)
        out.attrs["data_path"] = list(csv_paths)
        return out

    return dfs


def _resolve_grid_value(
    grid_values: ArrayLike,
    requested_value: Any,
    *,
    key: str,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    snap: bool = True,
) -> Any:
    arr = np.asarray(grid_values)

    try:
        arr_float = arr.astype(float)
        req_float = float(requested_value)

        close = np.isclose(arr_float, req_float, atol=atol, rtol=rtol)
        if np.any(close):
            return _to_python_scalar(arr[np.where(close)[0][0]])

        nearest_idx = int(np.nanargmin(np.abs(arr_float - req_float)))
        nearest_value = arr[nearest_idx]

        if snap and np.isclose(float(nearest_value), req_float, atol=atol, rtol=rtol):
            return _to_python_scalar(nearest_value)

        raise ValueError(
            f"Requested {key}={requested_value!r} not found in grid. "
            f"Nearest available value is {nearest_value!r}."
        )

    except ValueError:
        raise

    except Exception:
        matches = np.where(arr == requested_value)[0]
        if len(matches) > 0:
            return _to_python_scalar(arr[matches[0]])

        raise ValueError(
            f"Requested {key}={requested_value!r} not found in grid."
        )


def _snap_fixed_params_to_grid(
    grid_coords: Mapping[str, ArrayLike],
    fixed_params: Mapping[str, Any],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    snap: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for key, value in fixed_params.items():
        if key in grid_coords:
            out[key] = _resolve_grid_value(
                grid_coords[key],
                value,
                key=key,
                atol=atol,
                rtol=rtol,
                snap=snap,
            )
        else:
            out[key] = value

    return out


def _extract_cell_channel(cell: Any, channel_idx: int) -> np.ndarray:
    if cell is None:
        return np.asarray([])

    try:
        if len(cell) <= channel_idx:
            return np.asarray([])
        return np.asarray(cell[channel_idx])
    except Exception:
        return np.asarray([])


def _reorder_filtered_grid_to_xy(
    new_coords: Mapping[str, ArrayLike],
    Z_filtered: np.ndarray,
    *,
    x_key: str,
    y_key: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Make sure the filtered 2D grid follows:

        axis 0 -> x_key
        axis 1 -> y_key
    """

    remaining_keys = list(new_coords.keys())

    if set(remaining_keys) != {x_key, y_key}:
        raise ValueError(
            "After filtering, the remaining free coordinates must be exactly "
            f"{x_key!r} and {y_key!r}. Got remaining keys: {remaining_keys}. "
            "Please add other parameters to fixed_params."
        )

    if remaining_keys == [x_key, y_key]:
        coords_xy = {
            x_key: np.asarray(new_coords[x_key]),
            y_key: np.asarray(new_coords[y_key]),
        }
        return coords_xy, Z_filtered

    if remaining_keys == [y_key, x_key]:
        coords_xy = {
            x_key: np.asarray(new_coords[x_key]),
            y_key: np.asarray(new_coords[y_key]),
        }
        return coords_xy, np.transpose(Z_filtered, axes=(1, 0))

    raise RuntimeError("Unexpected coordinate ordering.")


def default_eigensolution_field_builder(
    *,
    new_coords: Mapping[str, ArrayLike],
    Z_grouped: np.ndarray,
    additional_Z_grouped: np.ndarray,
    z_keys: Sequence[str],
    band_index: int,
) -> Dict[str, np.ndarray]:
    """
    Default field extraction logic matching your current script.
    """

    _, Z_target = group_solution(
        new_coords,
        Z_grouped,
        freq_index=band_index,
    )

    (
        eigenfreq,
        qfactor,
        up_tanchi,
        up_phi,
        down_tanchi,
        down_phi,
        fake_factor,
        freq,
        u_factor_raw,
        up_cx,
        up_cy,
        down_cx,
        down_cy,
    ) = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=band_index,
    )

    qfactor = np.asarray(qfactor)
    u_factor_raw = np.asarray(u_factor_raw)

    with np.errstate(divide="ignore", invalid="ignore"):
        qlog = np.log10(qfactor).real

    u_abs = np.abs(np.real(u_factor_raw))

    with np.errstate(divide="ignore", invalid="ignore"):
        u_eff = -(1 - u_abs) / (1 + u_abs)

    return {
        "eigenfreq": np.asarray(Z_target),
        "eigenfreq_real": np.asarray(Z_target).real,
        "eigenfreq_imag": np.asarray(Z_target).imag,
        "qfactor": qfactor,
        "qlog": qlog,
        "up_tanchi": np.asarray(up_tanchi),
        "up_phi": np.asarray(up_phi),
        "down_tanchi": np.asarray(down_tanchi),
        "down_phi": np.asarray(down_phi),
        "fake_factor": np.asarray(fake_factor),
        "freq": np.asarray(freq),
        "u_factor_raw": u_factor_raw,
        "u_factor": u_abs,
        "u_eff": u_eff,
        "up_cx": np.asarray(up_cx),
        "up_cy": np.asarray(up_cy),
        "down_cx": np.asarray(down_cx),
        "down_cy": np.asarray(down_cy),
    }


def build_single_eigensolution_slice_from_grid(
    *,
    grid_coords: Mapping[str, ArrayLike],
    Z: np.ndarray,
    z_keys: Sequence[str],
    x_key: str,
    y_key: str,
    slice_key: str,
    slice_value: Any,
    fixed_params: Mapping[str, Any],
    filter_conditions: Mapping[str, Mapping[str, Any]],
    band_index: int,
    grouping_z_key: str = "特征频率 (THz)",
    max_num: int = 10,
    deltas: Optional[Tuple[float, float]] = None,
    value_weights: Optional[np.ndarray] = None,
    deriv_weights: Optional[np.ndarray] = None,
    auto_split_streams: bool = False,
    field_builder: Optional[FieldBuilderFn] = None,
    snap_fixed_params: bool = True,
    snap_atol: float = 1e-9,
    snap_rtol: float = 1e-9,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SliceDataset:
    """
    Build one prepared SliceDataset for one slice value.
    """

    from eigenmode_analysis.tracking import group_vectors_one_sided_hungarian

    if slice_key not in grid_coords:
        raise KeyError(
            f"slice_key={slice_key!r} not found in grid_coords. "
            "Please ensure slice_key is included in param_keys."
        )

    resolved_slice_value = _resolve_grid_value(
        grid_coords[slice_key],
        slice_value,
        key=slice_key,
        atol=snap_atol,
        rtol=snap_rtol,
        snap=snap_fixed_params,
    )

    fixed = dict(fixed_params)
    fixed[slice_key] = resolved_slice_value

    fixed = _snap_fixed_params_to_grid(
        grid_coords,
        fixed,
        atol=snap_atol,
        rtol=snap_rtol,
        snap=snap_fixed_params,
    )

    new_coords_raw, Z_filtered_raw, min_lens = advanced_filter_eigensolution(
        grid_coords,
        Z,
        z_keys=z_keys,
        fixed_params=fixed,
        filter_conditions=filter_conditions,
    )

    coords_xy, Z_filtered = _reorder_filtered_grid_to_xy(
        new_coords_raw,
        Z_filtered_raw,
        x_key=x_key,
        y_key=y_key,
    )

    if grouping_z_key not in z_keys:
        raise KeyError(
            f"grouping_z_key={grouping_z_key!r} not found in z_keys."
        )

    grouping_idx = list(z_keys).index(grouping_z_key)

    Z_new = np.empty(Z_filtered.shape, dtype=object)

    for idx in np.ndindex(Z_filtered.shape):
        Z_new[idx] = _extract_cell_channel(
            Z_filtered[idx],
            grouping_idx,
        )

    if deltas is None:
        deltas = _infer_deltas_for_xy(coords_xy, x_key=x_key, y_key=y_key)

    if value_weights is None:
        value_weights = _make_default_weights(ndim=2)

    if deriv_weights is None:
        deriv_weights = _make_default_weights(ndim=2)

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new],
        deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=max_num,
        auto_split_streams=auto_split_streams,
    )

    builder = field_builder or default_eigensolution_field_builder

    fields = builder(
        new_coords=coords_xy,
        Z_grouped=Z_grouped,
        additional_Z_grouped=additional_Z_grouped,
        z_keys=z_keys,
        band_index=band_index,
    )

    ds_metadata = {
        "band_index": band_index,
        "max_num": max_num,
        "fixed_params": fixed,
        "filter_conditions": dict(filter_conditions),
        "min_lens": min_lens,
    }

    if metadata:
        ds_metadata.update(dict(metadata))

    return SliceDataset(
        slice_key=slice_key,
        slice_value=resolved_slice_value,
        coords={
            x_key: np.asarray(coords_xy[x_key]),
            y_key: np.asarray(coords_xy[y_key]),
        },
        fields=fields,
        metadata=ds_metadata,
    )


def build_eigensolution_slice_series_from_dataframe(
    df: pd.DataFrame,
    *,
    selected_slice_values: Optional[Sequence[Any]] = None,
    slice_key: str,
    x_key: str,
    y_key: str,
    param_keys: Sequence[str],
    z_keys: Sequence[str],
    fixed_params: Mapping[str, Any],
    filter_conditions: Mapping[str, Mapping[str, Any]],
    band_index: Sequence[int] = (),
    max_num: Sequence[int] = (),
    slice_specs: Optional[Sequence[SliceBuildSpec]] = None,
    deduplication: bool = False,
    grouping_z_key: str = "特征频率 (THz)",
    deltas: Optional[Tuple[float, float]] = None,
    value_weights: Optional[np.ndarray] = None,
    deriv_weights: Optional[np.ndarray] = None,
    auto_split_streams: bool = False,
    field_builder: Optional[FieldBuilderFn] = None,
    snap_fixed_params: bool = True,
    snap_atol: float = 1e-9,
    snap_rtol: float = 1e-9,
    on_missing_slice: str = "warn",
    verbose: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> List[SliceDataset]:
    """
    Build prepared slice datasets from a preprocessed DataFrame.

    Either provide ``slice_specs`` or the legacy ``selected_slice_values`` /
    ``band_index`` /
    ``max_num`` trio.
    """

    if slice_key not in param_keys:
        raise ValueError(f"slice_key={slice_key!r} must be included in param_keys.")

    if x_key not in param_keys:
        raise ValueError(f"x_key={x_key!r} must be included in param_keys.")

    if y_key not in param_keys:
        raise ValueError(f"y_key={y_key!r} must be included in param_keys.")

    grid_coords, Z = create_data_grid(
        df,
        param_keys,
        z_keys,
        deduplication=deduplication,
    )

    if verbose:
        print("Grid parameters:")
        for key, arr in grid_coords.items():
            print(f"  {key}: count = {len(arr)}, values = {arr}")
        print(f"Z shape: {Z.shape}")

    if slice_specs is None:
        if selected_slice_values is None:
            selected_slice_values = list(grid_coords[slice_key])
        slice_specs = _normalize_slice_build_specs(
            selected_slice_values,
            band_index,
            max_num,
        )
    else:
        slice_specs = list(slice_specs)

    out: List[SliceDataset] = []

    for spec in slice_specs:
        slice_value = spec.slice_value
        band_index_i = int(spec.band_index)
        max_num_i = int(spec.max_num)

        try:
            ds = build_single_eigensolution_slice_from_grid(
                grid_coords=grid_coords,
                Z=Z,
                z_keys=z_keys,
                x_key=x_key,
                y_key=y_key,
                slice_key=slice_key,
                slice_value=slice_value,
                fixed_params=fixed_params,
                filter_conditions=filter_conditions,
                band_index=band_index_i,
                grouping_z_key=grouping_z_key,
                max_num=max_num_i,
                deltas=deltas,
                value_weights=value_weights,
                deriv_weights=deriv_weights,
                auto_split_streams=auto_split_streams,
                field_builder=field_builder,
                snap_fixed_params=snap_fixed_params,
                snap_atol=snap_atol,
                snap_rtol=snap_rtol,
                metadata=metadata,
            )
            out.append(ds)

            if verbose:
                first_field_key = next(iter(ds.fields))
                print(
                    f"Built slice {slice_key}={ds.slice_value}, "
                    f"band_index={band_index_i}, "
                    f"max_num={max_num_i}, "
                    f"shape={ds.get_field(first_field_key).shape}"
                )

        except Exception as exc:
            msg = (
                f"Failed to build slice {slice_key}={slice_value!r}, "
                f"band_index={band_index_i}, max_num={max_num_i}. "
                f"Reason: {exc}"
            )

            if on_missing_slice == "raise":
                raise RuntimeError(msg) from exc
            if on_missing_slice == "warn":
                warnings.warn(msg, RuntimeWarning)
            elif on_missing_slice == "ignore":
                pass
            else:
                raise ValueError(
                    "on_missing_slice must be 'raise', 'warn', or 'ignore'."
                )

    if len(out) == 0:
        raise RuntimeError("No SliceDataset was successfully built.")

    return _sort_slices(out)


def _deduplicate_slice_datasets(
    datasets: Sequence[SliceDataset],
    *,
    policy: str = "error",
) -> List[SliceDataset]:
    if policy == "keep":
        return list(datasets)

    seen: Dict[Any, SliceDataset] = {}

    for ds in datasets:
        key = ds.slice_value

        if key in seen:
            if policy == "error":
                raise ValueError(
                    f"Duplicate slice_value={key!r} found. "
                    "Use duplicate_slice_policy='first', 'last', or 'keep'."
                )
            if policy == "first":
                continue
            if policy == "last":
                seen[key] = ds
                continue

            raise ValueError(
                "duplicate_slice_policy must be 'error', 'first', 'last', or 'keep'."
            )

        seen[key] = ds

    return _sort_slices(list(seen.values()))


def build_eigensolution_slice_series_from_csvs(
    csv_paths: Union[str, Sequence[str]],
    *,
    selected_slice_values: Optional[Sequence[Any]] = None,
    slice_key: str,
    x_key: str,
    y_key: str,
    param_keys: Sequence[str],
    z_keys: Sequence[str],
    fixed_params: Mapping[str, Any],
    filter_conditions: Mapping[str, Mapping[str, Any]],
    band_index: Sequence[int] = (),
    max_num: Sequence[int] = (),
    slice_specs: Optional[Sequence[SliceBuildSpec]] = None,
    sep: str = "\t",
    period: float = 500,
    default_preprocess: bool = True,
    preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    read_csv_kwargs: Optional[Mapping[str, Any]] = None,
    concat_csvs: bool = True,
    duplicate_slice_policy: str = "error",
    deduplication: bool = False,
    grouping_z_key: str = "特征频率 (THz)",
    deltas: Optional[Tuple[float, float]] = None,
    value_weights: Optional[np.ndarray] = None,
    deriv_weights: Optional[np.ndarray] = None,
    auto_split_streams: bool = False,
    field_builder: Optional[FieldBuilderFn] = None,
    snap_fixed_params: bool = True,
    snap_atol: float = 1e-9,
    snap_rtol: float = 1e-9,
    on_missing_slice: str = "warn",
    verbose: bool = True,
) -> List[SliceDataset]:
    """
    One-stop data-preparation workflow from one or multiple CSV files.

    Either provide ``slice_specs`` or the legacy ``selected_slice_values`` /
    ``band_index`` /
    ``max_num`` trio.
    """

    loaded = load_and_preprocess_eigensolution_csvs(
        csv_paths,
        sep=sep,
        period=period,
        default_preprocess=default_preprocess,
        preprocess_fn=preprocess_fn,
        read_csv_kwargs=read_csv_kwargs,
        concat_csvs=concat_csvs,
    )

    csv_path_list = list(csv_paths) if not isinstance(csv_paths, str) else [csv_paths]

    if concat_csvs:
        assert isinstance(loaded, pd.DataFrame)

        metadata = {
            "csv_paths": csv_path_list,
            "concat_csvs": True,
        }

        return build_eigensolution_slice_series_from_dataframe(
            loaded,
            selected_slice_values=selected_slice_values,
            slice_key=slice_key,
            x_key=x_key,
            y_key=y_key,
            param_keys=param_keys,
            z_keys=z_keys,
            fixed_params=fixed_params,
            filter_conditions=filter_conditions,
            band_index=band_index,
            max_num=max_num,
            slice_specs=slice_specs,
            deduplication=deduplication,
            grouping_z_key=grouping_z_key,
            deltas=deltas,
            value_weights=value_weights,
            deriv_weights=deriv_weights,
            auto_split_streams=auto_split_streams,
            field_builder=field_builder,
            snap_fixed_params=snap_fixed_params,
            snap_atol=snap_atol,
            snap_rtol=snap_rtol,
            on_missing_slice=on_missing_slice,
            verbose=verbose,
            metadata=metadata,
        )

    assert isinstance(loaded, list)

    all_datasets: List[SliceDataset] = []

    for df in loaded:
        path = df.attrs.get("data_path", None)

        metadata = {
            "csv_paths": [path],
            "concat_csvs": False,
        }

        try:
            datasets_i = build_eigensolution_slice_series_from_dataframe(
                df,
                selected_slice_values=selected_slice_values,
                slice_key=slice_key,
                x_key=x_key,
                y_key=y_key,
                param_keys=param_keys,
                z_keys=z_keys,
                fixed_params=fixed_params,
                filter_conditions=filter_conditions,
                band_index=band_index,
                max_num=max_num,
                slice_specs=slice_specs,
                deduplication=deduplication,
                grouping_z_key=grouping_z_key,
                deltas=deltas,
                value_weights=value_weights,
                deriv_weights=deriv_weights,
                auto_split_streams=auto_split_streams,
                field_builder=field_builder,
                snap_fixed_params=snap_fixed_params,
                snap_atol=snap_atol,
                snap_rtol=snap_rtol,
                on_missing_slice=on_missing_slice,
                verbose=verbose,
                metadata=metadata,
            )
            all_datasets.extend(datasets_i)

        except Exception as exc:
            msg = f"Failed to process csv {path!r}. Reason: {exc}"

            if on_missing_slice == "raise":
                raise RuntimeError(msg) from exc
            if on_missing_slice == "warn":
                warnings.warn(msg, RuntimeWarning)
            elif on_missing_slice == "ignore":
                pass
            else:
                raise ValueError(
                    "on_missing_slice must be 'raise', 'warn', or 'ignore'."
                )

    if len(all_datasets) == 0:
        raise RuntimeError("No SliceDataset was successfully built.")

    return _deduplicate_slice_datasets(
        all_datasets,
        policy=duplicate_slice_policy,
    )
