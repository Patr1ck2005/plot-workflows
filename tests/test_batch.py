import pandas as pd

from plot_workflows.batch import GridSpec, PlotSpec, PlotUnit, load_and_prepare_dataframe


def test_batch_specs_validate_layout():
    spec = PlotSpec(name="line", plot_func=lambda *args, **kwargs: None)
    unit = PlotUnit(layout=(1, 1), specs=[spec])
    assert unit.unit_figsize == spec.figsize
    assert GridSpec(grid_dims=("sweep",)).grid_dims == ("sweep",)


def test_batch_loader_supports_named_frequency_key(tmp_path):
    path = tmp_path / "scan.tsv"
    pd.DataFrame({"f_thz": ["1+0.1i", "2+0.2i"]}).to_csv(path, sep="\t", index=False)
    frame = load_and_prepare_dataframe(
        str(path),
        period_nm=500,
        frequency_key="f_thz",
        normalized_frequency_key="f_norm",
    )
    assert "f_norm" in frame
    assert frame["f_norm"].notna().all()
