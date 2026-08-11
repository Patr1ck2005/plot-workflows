# Plot Workflows Agent Instructions

## Boundary

This package owns reusable, schema-neutral orchestration for tracking,
visualization, batch composition, and multislice workflows. It depends on
`eigenmode-analysis` for numerical primitives and `plot-foundation` for
specifications, styles, and renderers.

Callers own file formats, column-name mappings, physical interpretation,
manifests, output paths, and provenance storage. Do not add simulator or
application-specific imports to this package.

## Development

1. Characterize a workflow with synthetic inputs and caller-side tests.
2. Keep the reusable implementation here and add only thin caller adapters.
3. Run the package suite with optional extras needed by the changed module.
4. Record the installed shared-library versions in acceptance provenance.
5. Build a stable wheel before updating production callers.
