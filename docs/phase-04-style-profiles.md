# Phase 04: Shared presentation profiles

Plot Workflows now consumes Plot Foundation profiles when composing standard
band, heatmap, multi-surface, and Visualizer specifications.

- `diagnostic` preserves titles, axis labels, legends, grids, automatic layout,
  and consumer-supplied operational geometry.
- `publication_minimal` suppresses diagnostic chrome, disables automatic layout
  mutation, and uses semantic paper presets when the consumer requests one.
- `preview` uses minimal presentation with lower save DPI; `paper` remains a
  compatibility alias.

The shared workflow layer does not choose physical aspect, axis ranges, color
normalization, colormaps, paths, or output names. A and B select a profile and
may override geometry/DPI at their adapter boundaries.

Primary helpers are `resolve_plot_profile`, `axes_spec_for_profile`, and
`figure_spec_for_profile`. All tests use a headless Matplotlib backend.
