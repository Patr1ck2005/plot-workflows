# Plot Workflows

`plot-workflows` is a high-level, schema-neutral workflow layer. It sits above
`eigenmode-analysis` and `plot-foundation` and is designed to be embedded by
multiple applications.

The package owns orchestration over candidate paths/grids, standard band/Q and
surface plot specifications, reusable Visualizer lifecycle, batch composition,
and quasi-3D multislice workflows. Callers provide schema adapters, readers,
manifests, and output directories.

## Documentation by task

| Task | Entry |
|---|---|
| Identify raw/derived data spaces and choose views | [Canonical data-space specification](docs/data-space-and-visualization.md) |
| Apply styles and compose article figures | [Style-profile adaptation](docs/phase-04-style-profiles.md); [canonical figure-delivery policy](D:/Dev/Projects/Work/plot-foundation/docs/style-policy.md#a4-paper-delivery) |
| Use Visualizers, batch or multislice interfaces | [Public API](docs/public-api.md) |
| Maintain the package and consumer boundary | [Agent rules](AGENTS.md) |
| Follow the personal research system | [MyPhysics system entry](D:/Obsidian/MyPhysics/System/README.md); project semantics remain with callers |

Plot selection follows the canonical
[Data Space and Visualization Views](docs/data-space-and-visualization.md)
specification: classify intrinsic dimension and sampling topology first, then
choose compatible views. A 3D surface is a view of 2D data, not evidence of
dense 3D sampling.

Band, heatmap, multi-surface, and Visualizer composition accept a Plot
Foundation `style_profile`. Use `diagnostic` for exploratory output and
`publication_minimal` for A-derived compact paper figures. Plot Workflows maps
the selected presentation policy into axes/figure specs; consumers still own
physical aspect, limits, normalization, colormaps, filenames, and saving.

## Development

```powershell
python -m pip install -e .[dev]
python -m pytest -q
```

Install `.[multislice]` when using the optional pandas/matplotlib/scipy
multislice API. Importing the package root does not load that optional module.
Install `.[batch]` for SVG/PNG batch composition dependencies.

Production callers install a stable wheel. They must not add sibling source
paths through `sys.path` or `PYTHONPATH`.

See [docs/public-api.md](docs/public-api.md) for package boundaries and
optional dependencies.
