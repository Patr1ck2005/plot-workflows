# Data Space and Visualization Views

This document is the canonical plotting-decision specification for
`plot-workflows`, `plot-foundation`, and their consumers.

> Classify the intrinsic data dimension and sampling topology first. Then choose one or more compatible visualization views. Rendering dimension must never be used to infer data dimension.

This is a workflow contract, not a new public `DataSpace` API. Existing plot
specifications, Visualizers, and consumer schemas remain unchanged.

## 1. Three-layer model

Every analysis-and-plotting decision has three distinct layers:

1. **Raw data space**: the independent coordinates, their intrinsic dimension,
   and how they were sampled.
2. **Derived analysis space** (optional): the coordinates and quantity produced
   by an explicit analysis operator applied to the raw data.
3. **Visualization view**: the projection, glyph, color, surface, or volume
   representation selected to answer a particular question.

The layers are related but not identical. An analysis operator may reduce,
expand, preserve, or otherwise transform the raw data space before rendering
is chosen, but it is not mandatory. A raw object may be rendered directly when
the raw observable already answers the scientific question. A view can render
either the raw object or a transformed object in more than one way, and raw and
derived objects may also be shown together when their roles are explicit.
The same 2D scalar grid can produce a heatmap and a 3D surface. A scalar field
sampled on an `x-y` grid is intrinsically two-dimensional whether it is shown
as a heatmap, contours, a colored texture, or a three-dimensional surface. A
3D surface is a rendering of a 2D function `z = f(x, y)`; it is not dense 3D
data.

## 2. Analysis-space transformations (optional)

When a transformation is used, it is a scientific map from a raw data space to
a derived data space. Do not classify it by its operation name alone: compare the
independent coordinate sets and sampling topology before and after the map.
The transformation may be:

- **Reduction (Peak/ridge extraction or Slice-wise aggregation)**: an
  angle-frequency field `I(theta, f)` becomes a ridge `f_peak(theta)`, or a
  quasi-3D family `F(x, y, s)` becomes `F_max(s)`.
- **Lift/expansion**: a fixed eigenfrequency `f_n` is combined with excitation,
  ports, radiation channels, and frequency sampling to produce `R(f)` or
  `S(theta, f)`. This is commonly `0D -> 1D` or `0D -> 2D`, but an existing
  eigenmode branch may already provide one of those coordinates.
- **Same-dimensional transformation**: normalization, fitting, coordinate
  reparameterization, branch tracking, or a physical observable change while
  the independent-coordinate dimension is preserved.
- **Channel expansion**: one coordinate domain gains multiple observables such
  as `R/T/A`, `S11/S21`, or `S1/S2/S3`. More channels do not by themselves add
  an independent coordinate dimension.
- **Model-derived response**: an eigenmode solution is mapped to a scattering,
  emission, or polarization response using auxiliary model inputs. The result
  is not determined by the eigenfrequency value alone.

For every non-trivial transformation, record at least:

1. raw-space coordinates, intrinsic dimension, and sampling topology;
2. the transformation type and analysis/model operator;
3. auxiliary inputs such as excitation, port, radiation, interpolation, branch,
   threshold, mask, or continuity rules;
4. derived coordinates, quantity, units, channels, and resulting dimension; and
5. uncertainty, validity, coverage, and provenance information when available.

Peak and branch extraction must not silently equate a local numerical maximum
with a physical mode. Conversely, a scattering response must not be described
as a direct eigenfrequency field without documenting the coupling/model step.
The resulting data dimension is determined by the actual independent
coordinates, not by whether the output is called a spectrum, surface, or
trajectory. If no transformation is applied, record that the selected view is
a direct raw-data view and retain the raw-space classification as the view's
data-space contract.

## 3. Required classification

Before selecting a pipeline or Visualizer, record all of the following:

1. **Intrinsic data dimension**: 0D, 1D, 2D, dense 3D, quasi-2D, or quasi-3D.
2. **Sampling topology**: point, ordered path, regular grid, scattered samples,
   sparse curve family, sparse slice family, or dense voxel/grid volume.
3. **Parameter-space type**: momentum, geometry/material/environment,
   frequency/angle/polarization, or a mixed space.
4. **Physical quantity**: for example frequency, Q/loss, directionality,
   Stokes/polarization, a spectrum, or a vector/tensor field.
5. **Compatible view candidates**: the useful views supported by the data and
   the scientific question.
6. **Selected views and rationale**: one primary view and only the
   supplementary views that add information.
7. **Transformation status**: direct raw-data view, transformed derived-data
   view, or an explicit raw-plus-derived comparison.

Do not generate every compatible view by default. Candidate enumeration is a
reasoning guardrail; output selection remains task-driven.

## 4. Data-space classes

### 4.1 0D / point observable

A single observable at fixed independent parameters, such as one
eigenfrequency, one fitted coefficient, or one mode-level diagnostic. It has no
independent sampling axis. It may be lifted to a response curve or response
field when an explicit frequency, angle, port, or polarization sampling step is
introduced.

### 4.2 1D

One independent coordinate sampled as an ordered path or as scattered points.
Examples include a Bloch path, a wavelength scan, or a geometry sweep.

Compatible views include curves, scatter, color-mapped lines, ribbons,
polar curves when the coordinate is angular, and 3D trajectories when two
additional dependent quantities are used as displayed coordinates.

### 4.3 2D

Two independent coordinates sampled on a regular/irregular grid or as
scattered points. Examples include `kx-ky`, frequency-angle, and two-parameter
geometry spaces.

Compatible views include heatmaps, contours, colored textures, 3D surfaces,
vector/director fields, polarization textures, and lower-dimensional
projections. The same 2D scalar object may be the source for both a heatmap and
a surface view.

### 4.4 Dense 3D

Three independent coordinates densely sampled as a voxel or regular/irregular
3D grid. Compatible views include slices, isosurfaces, volume rendering, and
projections.

The current shared stack defines this boundary but does not provide a complete
dense-volume renderer. Do not route dense 3D data through the quasi-3D
multislice workflow merely because both can be displayed with slices.

### 4.5 Quasi-2D

A sparse family of 1D curves indexed by a second parameter. The second
coordinate is not sampled densely enough to be treated as a continuous 2D
field without an explicit interpolation assumption.

Compatible views include separate curves, overlaid/offset curves, stacked
curves, a family summary, trajectory extraction, or a projection. A heatmap is
allowed only when the sampling and interpolation assumptions are stated.

### 4.6 Quasi-3D

A sparse family of 2D grids indexed by a third parameter. Each slice remains a
2D data object; the slice family is not automatically a dense 3D volume.

Compatible views include per-slice 2D plots, stacked/3D slice presentation,
cross-slice trajectories, and 2D projections or reductions. These are the
four core multislice views guarded by the shared examples.

`quasi-1D` is intentionally not part of this specification. A sparse set of
scalar observations should be described directly by its sampling topology.

## 5. Compatibility matrix

| Data space | Typical topology | Compatible primary views | Useful supplementary views | Important caveat |
|---|---|---|---|---|
| 0D | single point | annotated value/table | response curve after an explicit lift | no independent sampling axis |
| 1D | ordered path | line, scatter, color-mapped line, ribbon | polar curve, 3D trajectory | displayed 3D coordinates do not make the data 3D |
| 2D | regular/irregular grid | heatmap, contour, surface | vector/director field, polarization texture, projection | heatmap and surface may share the same x-first scalar grid |
| 2D | scattered samples | scatter/triangulated contour | interpolated heatmap or surface | interpolation must be explicit |
| quasi-2D | sparse curve family | separate or overlaid curves | stacked curves, trajectory, projection | do not imply a dense 2D field without justification |
| quasi-3D | sparse 2D slice family | per-slice heatmaps/contours | stacked slices, cross-slice trajectory, projection | not a dense volume |
| dense 3D | voxel/3D grid | slices, isosurface, volume | projection/reduction | complete volume rendering is outside the current shared API |

All shared grid specifications use the Plot Foundation x-first convention:
`values.shape == (len(x), len(y))`. Consumers with y-first payloads must
transpose exactly once at their adapter boundary.

## 6. Physical-quantity guidance

The physical quantity and task determine which compatible view is useful.

| Physical quantity or task | Recommended primary views | Typical supplementary views |
|---|---|---|
| Eigenfrequency or dispersion | 1D band line/scatter; 2D heatmap or surface | gap map, selected-band surface, trajectory across slices |
| Q factor, loss, linewidth | log-scaled line or heatmap; color-mapped band | contours at thresholds, slice trajectory, clipped diagnostic view |
| Directionality or unidirectionality | signed/separate directional maps or curves | difference, ratio, frequency surface colored by directionality |
| Stokes or polarization | S1/S2/S3 maps, ellipse/director texture | Poincare trajectory/scatter, singularity overlay, projected path |
| Spectrum (R/T/A, ports, emission) | frequency curve or frequency-angle heatmap | phase, polarization-resolved panels, selected cuts |
| Vector/tensor field | quiver/director/texture on the underlying 2D grid | magnitude heatmap, streamlines, selected components |
| Quasi-3D evolution | per-slice 2D diagnostic or stacked slices | extrema trajectory and 2D projection/reduction |

These are recommendations, not automatic plot bundles. Preserve the physical
normalization, aspect, limits, colormap, and sign conventions owned by the
consumer project.

## 7. Agent decision record

Before implementation, an Agent should provide a compact record such as:

```text
Intrinsic dimension: 2D
Sampling topology: regular kx-ky grid
Parameter space: momentum
Physical quantity: log10(Q)
Compatible views: heatmap, contour, surface, threshold projection
Selection: heatmap as the primary diagnostic; contour as a threshold overlay;
surface omitted because it does not add information for this task
Transformation status: direct raw-data view
```

When an analysis transformation is applied, append a transformation record
before selecting a view:

```text
Raw space: 0D eigenfrequency f_n at fixed k and structure
Transformation: model-derived scattering response with port excitation
Auxiliary inputs: radiation channels, frequency sampling, and coupling model
Derived space: 1D R(f), T(f), A(f), with validity and resonance mask
Selected view: line as primary; confidence ribbon as supplementary
Transformation status: transformed derived-data view
```

Only after this record should the Agent select `run_pipeline_*`, a Visualizer,
a Plot Foundation spec, or a multislice workflow.

## 8. Library boundaries

- `plot-foundation` owns domain-neutral typed plot specifications, style
  policy, and caller-owned-axes renderers.
- `plot-workflows` owns schema-neutral orchestration, view-spec composition,
  Visualizers, batch workflows, and quasi-3D multislice workflows.
- Consumers own schema mapping, parameter meaning, physical interpretation,
  output selection, filenames, persistence, and provenance.
- `eigenmode-analysis` and `campaign-data` do not own visualization-selection
  policy.
