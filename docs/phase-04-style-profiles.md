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

## Physical-size delivery

For A4 portrait / PowerPoint delivery, follow the canonical
[Plot Foundation style policy](../../plot-foundation/docs/style-policy.md#a4-paper-delivery)
in the adjacent local checkout.
That document owns the current font size, tick direction, A4 column calibration,
compact panel geometry and complete-paper figure requirements. Its 2026-09-26
update replaces the earlier typography/assembly guidance; do not maintain
another numerical style policy in this workflow document.

Composition must preserve the intended physical size and font size. Required
outputs include reusable discrete panels plus their size manifest. A
complete-paper figure task also delivers agent-designed main/supplementary
composites and editable sources, following the canonical policy. Many useful
panels are compatible with a compact final layout; panel count is not the
reason to enlarge individual canvases or shrink lettering.
For exact-size output, use the existing `SaveSpec` override with
`bbox_inches=None`, and verify the actual exported dimensions. Enlarged
diagnostic geometry is not the default PowerPoint delivery geometry.
