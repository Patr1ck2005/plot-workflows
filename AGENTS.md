# Plot Workflows Agent Instructions

## Boundary

This package owns reusable, schema-neutral orchestration for tracking,
visualization, batch composition, and multislice workflows. It depends on
`eigenmode-analysis` for numerical primitives and `plot-foundation` for
specifications, styles, and renderers.

Callers own file formats, column-name mappings, physical interpretation,
manifests, output paths, and provenance storage. Do not add simulator or
application-specific imports to this package.

## Data-space decision invariant

Before selecting a pipeline or Visualizer, classify the intrinsic data
dimension and sampling topology, then enumerate compatible views. Rendering
dimension does not determine data dimension: the same 2D scalar grid may be a
heatmap or a 3D surface. Follow the canonical
[Data Space and Visualization Views](docs/data-space-and-visualization.md)
specification and report its six-field Agent decision record. Select a primary
view and only useful supplementary views; do not generate every candidate by
default.

## Development

1. Characterize a workflow with synthetic inputs and caller-side tests.
2. Keep the reusable implementation here and add only thin caller adapters.
3. Run the package suite with optional extras needed by the changed module.
4. Record the installed shared-library versions in acceptance provenance.
5. Build a stable wheel before updating production callers.
6. Keep the canonical data-space specification and consumer Agent links intact.
