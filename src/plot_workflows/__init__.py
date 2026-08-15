"""High-level, schema-neutral workflows shared by A and B.

The package owns orchestration over typed arrays and plot payloads.  Consumer
projects remain responsible for translating their DataFrames, manifests,
physical policies, and output paths into these contracts.
"""

__version__ = "0.1.6"

from .plotting import band_plot_spec_from_payloads, render_band_payloads
from .filtering import advanced_filter_eigensolution
from .grid import (
    compress_data_axis,
    create_data_grid,
    extract_adjacent_fields,
    extract_cell,
    extract_cells,
    group_solution,
    query_data_grid,
)
from .pipeline import (
    run_pipeline_1d,
    run_pipeline_1d_from_grid,
    run_pipeline_2d,
    run_pipeline_2d_from_grid,
)
from .grid_tracking import (
    group_vectors_nd_hungarian_shared,
    group_vectors_one_sided_hungarian_shared,
    shared_tracking_provenance,
    track_vectors_nd_hungarian_shared,
)
from .visualization import (
    BasePlotter,
    HeatmapPlotter,
    LinePlotter,
    PlotConfig,
    PolarPlotter,
    ScatterPlotter,
    configure_visualizer_hooks,
)
from .visualizers import (
    BandPlotterOneDim,
    MomentumSpaceEigenPolarizationPlotter,
    MomentumSpaceEigenVisualizer,
    MomentumSpaceSpectrumPlotter,
    OneDimFieldVisualizer,
    TwoDimFieldVisualizer,
)
from .surfaces import multi_surface_spec_from_payloads, surface_heatmap_spec_from_payload
from .style_profiles import (
    axes_spec_for_profile,
    figure_spec_for_profile,
    resolve_plot_profile,
)
from .tracking import (
    CandidateSet,
    TrackingWorkflowOptions,
    TrackingWorkflowResult,
    run_graph_workflow,
    run_path_workflow,
    run_surface_workflow,
    tracking_maps,
)

__all__ = [
    "__version__",
    "CandidateSet",
    "TrackingWorkflowOptions",
    "TrackingWorkflowResult",
    "band_plot_spec_from_payloads",
    "render_band_payloads",
    "advanced_filter_eigensolution",
    "create_data_grid",
    "query_data_grid",
    "compress_data_axis",
    "extract_adjacent_fields",
    "extract_cell",
    "extract_cells",
    "group_solution",
    "run_pipeline_1d",
    "run_pipeline_1d_from_grid",
    "run_pipeline_2d",
    "run_pipeline_2d_from_grid",
    "shared_tracking_provenance",
    "group_vectors_one_sided_hungarian_shared",
    "track_vectors_nd_hungarian_shared",
    "group_vectors_nd_hungarian_shared",
    "PlotConfig",
    "BasePlotter",
    "ScatterPlotter",
    "LinePlotter",
    "PolarPlotter",
    "HeatmapPlotter",
    "configure_visualizer_hooks",
    "BandPlotterOneDim",
    "MomentumSpaceEigenPolarizationPlotter",
    "MomentumSpaceEigenVisualizer",
    "MomentumSpaceSpectrumPlotter",
    "OneDimFieldVisualizer",
    "TwoDimFieldVisualizer",
    "surface_heatmap_spec_from_payload",
    "multi_surface_spec_from_payloads",
    "axes_spec_for_profile",
    "figure_spec_for_profile",
    "resolve_plot_profile",
    "run_path_workflow",
    "run_graph_workflow",
    "run_surface_workflow",
    "tracking_maps",
]
