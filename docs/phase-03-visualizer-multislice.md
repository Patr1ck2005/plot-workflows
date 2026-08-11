# Phase 3: Visualizer, Batch, and Multislice Workflows

The high-level layer now contains the mature A engineering workflow that is
useful beyond one project:

- `plot_workflows.visualization` owns figure lifecycle, fonts, tick direction,
  figsize, annotations, save/show behavior, and consumer hooks.
- `plot_workflows.visualizers` owns the advanced line, grid, polarization,
  spectrum, and momentum-space Visualizer classes. Their numerical helpers use
  `eigenmode-analysis` and Plot Foundation; they no longer import myPlots.
- `plot_workflows.batch` and `batch_specs` own fixed-parameter case expansion,
  dataset preparation hooks, SVG/PNG composition, and plot-unit orchestration.
  A injects its Chinese TSV reader, tracker wrapper, serializer, logging, and
  legacy plotter defaults.
- `plot_workflows.multislice` owns the quasi-3D slice data containers,
  extrema trajectories, and 2D/3D plotting functions. It is an optional
  dependency group because it uses pandas, matplotlib, and scipy.

The historical A modules are compatibility aliases/wrappers. In particular,
`core.plot_cls`, `myplots.visualization.visualizers`, `myplots.batch.fixed_params`,
and the multislice modules resolve to shared implementations, while A's hooks
preserve JSON/pickle loading, Chinese field policy, output naming, and logging.

Workbench exposes `WorkbenchLineVisualizer` and `WorkbenchGridVisualizer` in
`research_agent_workbench.viz.shared_visualizers`. They adapt B's Workbench
`coordinates + data_list` payload into the shared Visualizer contract. Existing
band/Q, surface, tracking, and scalar tracking adapters continue to own B's
manifest and DatasetRegistry policy.

All tests use `MPLBACKEND=Agg`; shared and consumer tests close figures. The
wheel must be rebuilt and installed in both analysis environments after changes;
the COMSOL interpreter remains outside this dependency boundary.

Release acceptance for `plot-workflows 0.1.3`: 11 shared tests, 63 A tests,
and 100 B tests passed from stable wheel installations without `PYTHONPATH`.
Wheel SHA256: `bf23fd21a89397990ac0dfa2e284714e93bd3f8d64ff5154edd434d0ec7b80fe`.
