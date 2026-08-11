# Phase 1: Shared Workflow Kernel

This phase extracts the reusable orchestration around A's mature tracking and
band and surface plotting workflows into `plot-workflows`.

The package owns typed candidate path/grid traversal, compatibility label maps,
runtime provenance, and the standard raw/tracked band/Q and surface plot
contracts. Consumer
projects retain schema mapping, physical filtering, figure lifecycle, output
paths, and persistence.

The first consumer rollout targets Workbench's scalar metadata tracker and
band/Q and surface renderers. A's historical `core.*` imports remain unchanged;
its adapter delegates path and explicit-graph orchestration while retaining the
old object-grid return shapes.
