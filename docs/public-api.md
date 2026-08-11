# Public API and boundary

`plot_workflows` combines reusable tracking and plotting workflow objects. It
depends on `eigenmode-analysis` for numerical tracking and `plot-foundation`
for specifications, styles, and renderers.

The package accepts caller-owned arrays, records, and DataFrames through
explicit adapters. It does not know a simulator, manifest format, column-name
language, or output directory. Optional `batch`, `campaign`, and `multislice`
extras add file and scientific-data integrations without changing the core
boundary.
