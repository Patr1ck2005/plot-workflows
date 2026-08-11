"""Reusable quasi-3D multislice data and plotting workflows."""

from .data import (
    SliceBuildSpec,
    SliceDataset,
    SliceShapeReport,
    build_eigensolution_slice_series_from_csvs,
    build_eigensolution_slice_series_from_dataframe,
    build_single_eigensolution_slice_from_grid,
    default_eigensolution_field_builder,
    default_preprocess_eigensolution_dataframe,
    find_extrema_2d,
    find_extrema_trajectory,
    inspect_slice_series,
    load_and_preprocess_eigensolution_csvs,
    load_slice_datasets_pkl,
    load_trajectory_csv,
    save_slice_datasets_pkl,
    save_trajectory_csv,
    stack_field,
    validate_slice_dataset,
    configure_multislice_hooks,
)
from .plotting import (
    SliceAxisMapping,
    infer_color_norm,
    make_slice_axis_mapping,
    plot_extrema_trajectory_3d,
    plot_heatmap_slices_3d,
    plot_logq_and_ueff_vs_alpha,
    plot_projected_extrema_trajectory_2d,
    save_figure,
)

__all__ = [name for name in globals() if not name.startswith("_")]
