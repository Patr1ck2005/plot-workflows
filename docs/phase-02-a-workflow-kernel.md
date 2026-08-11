# Phase 2: A Workflow Kernel

This phase moves the reusable body of A's mature pipeline into the shared
package. `plot_workflows.pipeline` now owns the grid -> filter -> group ->
target extraction orchestration, while `plot_workflows.grid` owns object-grid
construction, cell extraction, coordinate queries, axis compression, grouped
target extraction, and adjacent-field extraction. `plot_workflows.filtering`
owns fixed-axis slicing and candidate filtering.

`plot_workflows.grid_tracking` owns the legacy-compatible object-grid adapter
that maps candidate cells to the shared path/graph tracker and preserves A's
historical return shapes.

myPlots keeps thin compatibility modules at the historical paths:

- `myplots.pipeline.workflow` maps Chinese defaults and selects legacy/shared/
  polynomial tracker callbacks;
- `myplots.pipeline.grid` keeps only Chinese-schema field extraction;
- `myplots.pipeline.filtering` re-exports the shared filtering function;
- `myplots.pipeline.tracking_adapter` re-exports the shared object-grid
  adapter.

No DataFrame, Chinese column, TSV, manifest, or output-path policy is owned by
the shared kernel. A remains behaviorally compatible, while another consumer
can provide its own schema adapter and tracker callback.

Validation: `plot-workflows` 6 tests, myPlots 63 tests, and Workbench 98 tests
pass after a stable wheel rebuild and installation in both analysis
environments. Plotting validation uses `MPLBACKEND=Agg`.
