from __future__ import annotations

import numpy as np
from plot_workflows import run_pipeline_1d


class Frame:
    columns = ("k", "freq", "q")

    def __init__(self, rows):
        self._rows = tuple(rows)

    def __getitem__(self, key):
        index = self.columns.index(key)
        return Column([row[index] for row in self._rows])

    def itertuples(self, index=False):
        del index
        return iter(self._rows)


class Column(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def unique(self):
        return np.unique(self)


def _first_candidate_tracker(group_cells, deltas, *, additional_data, **_options):
    deltas
    shape = group_cells[0].shape
    grouped = np.empty(shape, dtype=object)
    additional = np.empty(shape, dtype=object)
    for index in np.ndindex(*shape):
        values = [component[index][0] for component in group_cells]
        grouped[index] = np.asarray([values], dtype=float)
        additional[index] = [additional_data[index][0]]
    return grouped, additional


def test_dataframe_pipeline_kernel_preserves_object_grid_contract():
    frame = Frame([(k, 1.0 + k, 100.0) for k in (0.0, 1.0, 2.0)])
    coords, targets, additional, filtered = run_pipeline_1d(
        frame,
        ["k"],
        ["freq", "q"],
        {},
        {},
        1,
        group_cols=("freq",),
        tracker=_first_candidate_tracker,
    )
    np.testing.assert_array_equal(coords["k"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(targets[0], [1.0, 2.0, 3.0])
    assert additional.shape == filtered.shape == (3,)
