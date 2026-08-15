# Changelog

## 0.1.6

- Added the analysis-space transformation layer to the canonical data-space
  specification: raw data space -> explicit transformation -> derived
  coordinates/quantity -> visualization view, with direct raw-data views kept
  as a first-class path.
- Documented reduction, lift/expansion, same-dimensional transforms, channel
  expansion, model-derived responses, and transformation provenance without
  adding a public analysis API.
- Added characterization cases for a 2D angle-frequency field reduced to a 1D
  ridge, a quasi-3D slice family reduced to a 1D extrema trajectory, and a
  point eigenfrequency lifted to a sampled frequency response.

## 0.1.5

- Added the canonical intrinsic-data-dimension and sampling-topology decision
  specification without changing the public plotting API.
- Documented compatible views for 1D, 2D, dense 3D, quasi-2D, and quasi-3D
  data, including the distinction between a 2D field and a 3D surface view.
- Added guardrail examples for shared 2D heatmap/surface inputs and the four
  quasi-3D multislice view families.
- Raised the Plot Foundation floor to `0.1.12`.
- Included the canonical specification, Agent instructions, and tests in the
  sdist and modernized MIT SPDX metadata for clean public builds.

## 0.1.4

- Published schema-neutral tracking, plotting, visualization, batch, and
  multislice workflow primitives.
- Kept file formats, schema adapters, output paths, and application policy in
  callers.

Earlier internal development history is intentionally not part of the public
release branch.
