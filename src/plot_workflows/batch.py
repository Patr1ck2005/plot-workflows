from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .pipeline import run_pipeline_1d_from_grid as _shared_pipeline_1d
from .pipeline import run_pipeline_2d_from_grid as _shared_pipeline_2d
from .grid import extract_adjacent_fields, create_data_grid
from .visualization import PlotConfig
from .visualizers import OneDimFieldVisualizer, TwoDimFieldVisualizer


_BATCH_HOOKS = {
    "pipeline_1d": _shared_pipeline_1d,
    "pipeline_2d": _shared_pipeline_2d,
    "prepare_plot_data": None,
    "one_dim_plotter": OneDimFieldVisualizer,
    "two_dim_plotter": TwoDimFieldVisualizer,
    "print_debug": print,
    "print_grid_info": print,
    "print_progress": print,
    "print_saved": print,
}


def configure_batch_hooks(**hooks):
    """Inject consumer pipeline, serialization, plotter, and logging policy."""

    unknown = sorted(set(hooks) - set(_BATCH_HOOKS))
    if unknown:
        raise ValueError(f"unknown batch hooks: {unknown}")
    _BATCH_HOOKS.update(hooks)


def _convert_complex(value):
    return complex(value.replace("i", "j")) if isinstance(value, str) else value


def _norm_freq(freq, period):
    return freq / (299792458.0 / period)


def _default_prepare_plot_data(coords, data_class="Eigensolution", dataset_list=None,
                               fixed_params=None, save_dir="./rsl", **kwargs):
    """Minimal pickle writer matching the historical A payload contract."""

    del kwargs
    target_dir = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "plot_data_default.pkl"
    payload = {
        "coords": coords,
        "data_list": list(dataset_list or []),
        "metadata": {
            "fixed_params": dict(fixed_params or {}),
            "data_class": data_class,
        },
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path.resolve())


def _prepare_plot_data(*args, **kwargs):
    callback = _BATCH_HOOKS["prepare_plot_data"] or _default_prepare_plot_data
    return callback(*args, **kwargs)


def _print_debug(*args, **kwargs):
    _BATCH_HOOKS["print_debug"](*args, **kwargs)


def _print_grid_info(grid_coords, Z):
    callback = _BATCH_HOOKS["print_grid_info"]
    if callback is print:
        print("Grid parameters:", grid_coords, "Z shape:", getattr(Z, "shape", None))
    else:
        callback(grid_coords, Z)


def _print_progress(idx, total, name):
    callback = _BATCH_HOOKS["print_progress"]
    if callback is print:
        print(f"[{idx}/{total}] {name}")
    else:
        callback(idx, total, name)


def _print_saved(path, kind="Summary"):
    callback = _BATCH_HOOKS["print_saved"]
    if callback is print:
        print(f"{kind}: {path}")
    else:
        callback(path, kind=kind)


# =========================================================
# Dataclass definitions
# =========================================================

@dataclass
class PlotSpec:
    name: str
    plot_func: Callable  # (plotter, x_key, max_num, **kwargs) -> None
    figsize: Tuple[float, float] = (2.0, 2.0)
    kwargs: dict | None = None
    plotter_cls: type = OneDimFieldVisualizer
    fig_type: str = "2d"  # "2d" or "3d"


@dataclass
class PlotUnit:
    layout: Tuple[int, int]
    specs: List[PlotSpec]
    name: str = "unit"
    gap: Tuple[int, int] = (2, 2)  # (h_gap, v_gap) in px

    def __post_init__(self):
        expected = self.layout[0] * self.layout[1]
        if len(self.specs) != expected:
            raise ValueError(
                f"PlotUnit '{self.name}': layout {self.layout} expects "
                f"{expected} specs, got {len(self.specs)}"
            )

    @property
    def unit_figsize(self) -> Tuple[float, float]:
        rows, cols = self.layout
        max_w = max(s.figsize[0] for s in self.specs)
        max_h = max(s.figsize[1] for s in self.specs)
        return (max_w * cols, max_h * rows)


@dataclass
class GridSpec:
    grid_dims: Tuple[str, ...]
    split_dims: Tuple[str, ...] = ()
    n_cols: int | None = None
    gap: Tuple[int, int] = (5, 5)  # (h_gap, v_gap) in px
    footer_gap: int = 4  # px gap between last row and footer text


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_case_name(
    fixed_params: Dict[str, Any],
    keys_for_name: List[str],
) -> str:
    parts = []
    for k in keys_for_name:
        if k in fixed_params:
            safe_k = (
                k.replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
            )
            v = fixed_params[k]
            if isinstance(v, float):
                v_str = f"{v:g}"
            else:
                v_str = str(v)
            parts.append(f"{safe_k}_{v_str}")
    return "case_" + "__".join(parts)


def summarize_batch_params(
    base_fixed_params: Dict[str, Any],
    fixed_params_list: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    scanned: Dict[str, set] = {}
    for case in fixed_params_list:
        for k, v in case.items():
            if k not in base_fixed_params or base_fixed_params[k] != v:
                scanned.setdefault(k, set()).add(v)
    scanned_sorted: Dict[str, List[Any]] = {}
    for k, vals in scanned.items():
        try:
            scanned_sorted[k] = sorted(vals)
        except TypeError:
            scanned_sorted[k] = sorted(vals, key=lambda x: str(x))
    return scanned_sorted


def print_batch_params_summary(
    data_path: str,
    base_fixed_params: Dict[str, Any],
    fixed_params_list: List[Dict[str, Any]],
):
    scanned = summarize_batch_params(base_fixed_params, fixed_params_list)
    print("\n========== Batch Params ==========")
    print(f"[Source File: {data_path}]")
    print("[Fixed Params]")
    for k in sorted(base_fixed_params.keys()):
        print(f"- {k}: {base_fixed_params[k]}")
    print("[Scan Params]")
    if not scanned:
        print("- (none)")
    else:
        for k in sorted(scanned.keys()):
            values = scanned[k]
            preview = ", ".join(map(str, values[:10]))
            suffix = "" if len(values) <= 10 else f", ... (total {len(values)})"
            print(f"- {k}: [{preview}{suffix}]")
    print(f"[Cases] total={len(fixed_params_list)}")
    print("==================================\n")


def load_and_prepare_dataframe(data_path: str, period_nm: float, *,
                               frequency_key: str = "特征频率 (THz)",
                               normalized_frequency_key: str = "频率 (Hz)",
                               sep: str = "\t") -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=sep).copy()
    df[frequency_key] = (
        df[frequency_key]
        .apply(_convert_complex)
        .apply(_norm_freq, period=period_nm * 1e-9 * 1e12)
    )
    df[normalized_frequency_key] = np.real(df[frequency_key])
    return df


# =========================================================
# 数据管道适配
# =========================================================

def build_datasets_from_pipeline(
    new_coords: Dict[str, np.ndarray],
    Z_targets: List[np.ndarray],
    additional_Z_grouped: np.ndarray,
    z_keys: List[str],
    max_num: int,
) -> List[Dict[str, np.ndarray]]:
    """
    把 run_pipeline_1d 输出转为 Visualizer 可用的 dataset dicts。

    按 z_keys 顺序自动构建 derived fields：
    - qfactor 位置 → qlog
    - U_factor (1) 位置 → u_eff, -u_factor
    - up_S3 (1) 位置 → up_s3
    - up_tanchi (1) 位置 → up_tanchi
    """
    datasets = []
    for i in range(max_num):
        Z_target = Z_targets[i] if i < len(Z_targets) else None
        if Z_target is not None:
            dataset = {
                "eigenfreq_real": Z_target.real,
                "eigenfreq_imag": Z_target.imag,
            }
        else:
            dataset = {
                "eigenfreq_real": np.full_like(new_coords[list(new_coords.keys())[0]], np.nan, dtype=float),
                "eigenfreq_imag": np.full_like(new_coords[list(new_coords.keys())[0]], np.nan, dtype=float),
            }

        fields = extract_adjacent_fields(
            additional_Z_grouped, z_keys=z_keys, band_index=i,
        )

        for idx, key in enumerate(z_keys):
            if "品质因子" in key:
                q_safe = np.where(np.real(fields[idx]) > 0, fields[idx], np.nan)
                dataset["qlog"] = np.log10(q_safe).real
            elif key == "U_factor (1)":
                u_factor = fields[idx]
                dataset["-u_factor"] = np.abs(u_factor.real)
                u_abs = np.abs(u_factor.real)
                dataset["u_eff"] = -(1 - u_abs) / (1 + u_abs)
            elif key == "up_S3 (1)":
                dataset["up_s3"] = fields[idx].real
            elif key == "up_tanchi (1)":
                dataset["up_tanchi"] = fields[idx].real

        datasets.append(dataset)
    return datasets


# =========================================================
# SVG 工具
# =========================================================

def save_current_plotter_fig(plotter, save_path: Path, tight_bbox: bool = True,
                              formats=("svg",), png_dpi: int = 150):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = None
    if hasattr(plotter, "ax") and plotter.ax is not None:
        fig = plotter.ax.figure
    elif hasattr(plotter, "fig") and plotter.fig is not None:
        fig = plotter.fig
    else:
        raise RuntimeError("无法从 plotter 中提取 matplotlib figure，请检查 plotter 的内部属性。")

    if not tight_bbox:
        fig.tight_layout(pad=0.3)

    for fmt in formats:
        out = save_path.with_suffix(f".{fmt}")
        dpi = 72 if fmt == "svg" else png_dpi
        fig.savefig(out, format=fmt,
                    bbox_inches="tight" if tight_bbox else None,
                    transparent=True, dpi=dpi)
    plt.close(fig)


def compose_svgs_grid(
    svg_paths: List[Path],
    layout: Tuple[int, int],
    cell_size: Tuple[int, int],
    output_path: Path,
    gap: Tuple[int, int] = (0, 0),
    footer_lines: list | None = None,
    footer_gap: int = 4,
    png: bool = False,
):
    from svgutils.compose import Figure, SVG, Text

    h_gap, v_gap = gap

    rows, cols = layout
    cell_w, cell_h = cell_size
    total_w = cell_w * cols + h_gap * (cols - 1)
    line_height = 12
    footer_h = (line_height * len(footer_lines) + footer_gap) if footer_lines else 0
    total_h = cell_h * rows + v_gap * (rows - 1) + footer_h

    elements = []
    for idx, svg_path in enumerate(svg_paths):
        r, c = divmod(idx, cols)
        x0 = c * (cell_w + h_gap)
        y0 = r * (cell_h + v_gap)
        text = Path(svg_path).read_text(encoding="utf-8", errors="replace")
        text = text.replace("−", "-")
        Path(svg_path).write_text(text, encoding="utf-8")
        elements.append(SVG(str(svg_path)).move(x0, y0).scale(1.0))

    if footer_lines:
        for li, line in enumerate(footer_lines):
            elements.append(
                Text(line, x=5, y=total_h - footer_gap - (len(footer_lines) - 1 - li) * line_height,
                     size=8, weight="normal", font="sans-serif")
            )

    fig = Figure(f"{total_w}px", f"{total_h}px", *elements)
    fig.save(str(output_path))


# =========================================================
# 绘图渲染
# =========================================================

def render_plot_unit(
    data_path: str,
    plot_unit: PlotUnit,
    case_output_dir: Path,
    x_key: str,
    max_num: int,
    title: str | None = None,
    tight_bbox: bool = True,
    formats=("svg",),
    png_dpi: int = 150,
    y_key: str | None = None,
) -> Path:
    rows, cols = plot_unit.layout
    svg_paths: List[Path] = []

    for idx, spec in enumerate(plot_unit.specs):
        config = PlotConfig(
            plot_params={"scale": 1},
            annotations={
                "xlabel": "", "ylabel": "",
                "show_axis_labels": True, "show_tick_labels": True,
            },
        )
        config.update(figsize=spec.figsize, tick_direction="in")

        plotter_cls = spec.plotter_cls or (
            _BATCH_HOOKS["two_dim_plotter"]
            if spec.fig_type == "3d" or spec.plotter_cls is TwoDimFieldVisualizer
            else _BATCH_HOOKS["one_dim_plotter"]
        )
        plotter = plotter_cls(config=config, data_path=data_path)
        plotter.load_data()

        extra = spec.kwargs or {}
        if issubclass(plotter_cls, TwoDimFieldVisualizer):
            band_index = extra.pop("band_index", 0)
            if spec.fig_type == "3d":
                plotter.new_3d_fig(figsize=spec.figsize)
            else:
                plotter.new_2d_fig(figsize=spec.figsize)
            spec.plot_func(plotter, x_key=x_key, y_key=y_key, band_index=band_index, **extra)
        else:
            plotter.new_2d_fig()
            spec.plot_func(plotter, x_key=x_key, max_num=max_num, **extra)

        if title:
            fig = plotter.ax.figure if hasattr(plotter, "ax") else plotter.fig
            fig.suptitle(title, fontsize=8, y=0.99)

        svg_path = case_output_dir / f"{spec.name}.svg"
        save_current_plotter_fig(plotter, svg_path, tight_bbox=tight_bbox,
                                 formats=formats, png_dpi=png_dpi)
        svg_paths.append(svg_path)

    max_w = max(s.figsize[0] for s in plot_unit.specs)
    max_h = max(s.figsize[1] for s in plot_unit.specs)
    dpi = 72
    cell_w_px = int(max_w * dpi)
    cell_h_px = int(max_h * dpi)

    target = case_output_dir / f"{plot_unit.name}.svg"
    if len(svg_paths) == 1:
        unit_svg = svg_paths[0]
        if unit_svg != target:
            unit_svg.replace(target)
            unit_svg = target
    else:
        compose_svgs_grid(svg_paths, (rows, cols), (cell_w_px, cell_h_px),
                          target, gap=plot_unit.gap)
        unit_svg = target

    # Also compose unit PNG from spec PNGs
    png_cell_w = int(max_w * png_dpi)
    png_cell_h = int(max_h * png_dpi)
    png_paths = [p.with_suffix(".png") for p in svg_paths]
    png_target = case_output_dir / f"{plot_unit.name}.png"
    if len(png_paths) == 1 and png_paths[0].exists():
        if png_paths[0] != png_target:
            png_paths[0].replace(png_target)
    elif all(p.exists() for p in png_paths):
        _compose_png_grid(png_paths, (rows, cols), (png_cell_w, png_cell_h),
                          png_target, gap=plot_unit.gap, png_dpi=png_dpi)
    return unit_svg


# =========================================================
# 批处理主逻辑
# =========================================================

def _cartesian_product(dim_order: List[str], dim_values: List[List]) -> List[Dict[str, Any]]:
    import itertools
    combos = []
    for combo in itertools.product(*dim_values):
        combos.append(dict(zip(dim_order, combo)))
    return combos


def _compose_png_grid(png_paths: List[Path], layout: Tuple[int, int],
                       cell_size: Tuple[int, int], output_path: Path,
                       gap: Tuple[int, int] = (0, 0),
                       footer_lines: list | None = None,
                       footer_gap: int = 4,
                       png_dpi: int = 150):
    """Stitch PNG files into a grid using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    rows, cols = layout
    h_gap, v_gap = gap
    cell_w, cell_h = cell_size

    font_size = int(8 * png_dpi / 72)
    line_height = int(font_size * 1.6)
    footer_h = (line_height * len(footer_lines) + footer_gap) if footer_lines else 0
    total_w = cell_w * cols + h_gap * (cols - 1)
    total_h = cell_h * rows + v_gap * (rows - 1) + footer_h

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))

    for idx, png_path in enumerate(png_paths):
        if png_path is None or not png_path.exists():
            continue
        r, c = divmod(idx, cols)
        x0 = c * (cell_w + h_gap)
        y0 = r * (cell_h + v_gap)
        cell = Image.open(png_path)
        cell = cell.resize((cell_w, cell_h), Image.LANCZOS)
        img.paste(cell, (x0, y0), cell if cell.mode == "RGBA" else None)

    if footer_lines:
        draw = ImageDraw.Draw(img)
        font = _get_font(font_size)
        for li, line in enumerate(footer_lines):
            y = total_h - footer_h + footer_gap + li * line_height
            draw.text((5, y), line, fill=(0, 0, 0, 200), font=font)

    img.save(output_path, format="PNG")


def _get_font(size: int = 10):
    """Try to load a TrueType font, falling back to default."""
    from PIL import ImageFont
    for path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _make_figure_name(split_key: tuple, grid_spec: GridSpec) -> str:
    if not split_key:
        return "grid"
    parts = []
    for dim, val in zip(grid_spec.split_dims, split_key):
        safe_dim = dim.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        parts.append(f"{safe_dim}_{val}")
    return "_".join(parts)


def _compose_grid_figures(
    grid_data: dict,
    grid_spec: GridSpec,
    plot_unit: PlotUnit,
    output_root: Path,
    base_fixed_params: Dict[str, Any] | None = None,
    show_figure_footer: bool = True,
    png: bool = False,
    footer_supp: str = "",
    footer_filter_conditions: Dict[str, Any] | None = None,
    png_dpi: int = 150,
):
    svg_dpi = 72
    unit_w_px = int(plot_unit.unit_figsize[0] * svg_dpi)
    unit_h_px = int(plot_unit.unit_figsize[1] * svg_dpi)

    for split_key, grid_map in grid_data.items():
        n_dims = len(grid_spec.grid_dims)
        if n_dims == 1:
            dim0_vals = sorted(grid_map.keys(), key=lambda k: k[0])
            ordered = [grid_map[k] for k in dim0_vals]
            if grid_spec.n_cols is not None:
                n_cols = grid_spec.n_cols
            else:
                n_cols = 1
            n_rows = (len(ordered) + n_cols - 1) // n_cols
        elif n_dims == 2:
            col_vals = sorted(set(k[0] for k in grid_map))
            row_vals = sorted(set(k[1] for k in grid_map))
            n_cols = len(col_vals) if grid_spec.n_cols is None else grid_spec.n_cols
            n_rows = len(row_vals)
            ordered = [grid_map.get((cv, rv)) for rv in row_vals for cv in col_vals]
        else:
            raise ValueError(f"grid_dims must have 1 or 2 elements, got {n_dims}")

        ordered = [p for p in ordered if p is not None]

        figure_name = _make_figure_name(split_key, grid_spec)
        output_path = output_root / f"summary_{figure_name}.svg"

        footer_lines = []
        if show_figure_footer:
            line1_parts = []
            if footer_supp:
                line1_parts.append(footer_supp)
            if footer_filter_conditions:
                cond_strs = []
                for k, ops in footer_filter_conditions.items():
                    op_strs = [f"{op}{_fmt_val(v)}" for op, v in ops.items()]
                    cond_strs.append(f"{k}{''.join(op_strs)}")
                line1_parts.append(" | ".join(cond_strs))
            if line1_parts:
                footer_lines.append("  |  ".join(line1_parts))

            line2_parts = []
            if split_key:
                for d, v in zip(grid_spec.split_dims, split_key):
                    line2_parts.append(f"{d}={_fmt_val(v)}")
            if base_fixed_params:
                for k in sorted(base_fixed_params.keys()):
                    line2_parts.append(f"{k}={_fmt_val(base_fixed_params[k])}")
            if line2_parts:
                footer_lines.append("  |  ".join(line2_parts))

        compose_svgs_grid(
            ordered, (n_rows, n_cols),
            (unit_w_px, unit_h_px),
            output_path, gap=grid_spec.gap,
            footer_lines=footer_lines or None, footer_gap=grid_spec.footer_gap,
        )

        # PNG grid from unit PNGs
        if png:
            png_cell_w = int(plot_unit.unit_figsize[0] * png_dpi)
            png_cell_h = int(plot_unit.unit_figsize[1] * png_dpi)
            png_ordered = [p.with_suffix(".png") for p in ordered]
            png_output = output_path.with_suffix(".png")
            _compose_png_grid(
                png_ordered, (n_rows, n_cols),
                (png_cell_w, png_cell_h),
                png_output, gap=grid_spec.gap,
                footer_lines=footer_lines or None, footer_gap=grid_spec.footer_gap,
                png_dpi=png_dpi,
            )

        _print_saved(output_path, kind="Summary")


def batch_run_cases(
    *,
    df_sample: pd.DataFrame,
    output_root: str,
    x_key: str,
    param_keys: List[str],
    z_keys: List[str],
    param_ranges: Dict[str, List[Any]],
    base_fixed_params: Dict[str, Any],
    filter_conditions: Dict[str, Dict[str, float]],
    deltas: tuple,
    max_num: int,
    nan_cost_penalty: float,
    auto_split_streams: bool,
    plot_unit: PlotUnit,
    grid_spec: GridSpec | None = None,
    keys_for_case_name: List[str] | None = None,
    group_cols: Tuple[str, ...] = ("特征频率 (THz)",),
    show_unit_titles: bool = True,
    show_figure_footer: bool = True,
    formats: Tuple[str, ...] = ("svg", "png"),
    footer_supp: str = "",
    png_dpi: int = 150,
    pipeline_2d: bool = False,
    y_key: str | None = None,
    prepare_datasets: Callable | None = None,
):
    output_root = ensure_dir(output_root)

    if grid_spec is None:
        all_keys = tuple(param_ranges.keys())
        grid_spec = GridSpec(grid_dims=(all_keys[0],) if len(all_keys) == 1 else all_keys[-1:],
                             split_dims=all_keys[:-1] if len(all_keys) > 1 else ())

    if keys_for_case_name is None:
        keys_for_case_name = list(grid_spec.grid_dims) + list(grid_spec.split_dims)

    referenced = set(grid_spec.grid_dims) | set(grid_spec.split_dims)
    unknown = referenced - set(param_ranges.keys())
    if unknown:
        raise ValueError(f"GridSpec references unknown param keys: {unknown}")

    dim_order = list(grid_spec.split_dims) + list(grid_spec.grid_dims)
    dim_values = [[v for v in param_ranges[d]] for d in dim_order]
    all_combos = _cartesian_product(dim_order, dim_values)

    # 全量建 grid 一次，loop 内复用
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    _print_grid_info(grid_coords, Z)

    grid_data: dict = {}
    total = len(all_combos)

    for idx, combo_dict in enumerate(all_combos, start=1):
        fixed_params = {**base_fixed_params, **combo_dict}
        case_name = make_case_name(fixed_params, keys_for_name=keys_for_case_name)
        case_dir = ensure_dir(output_root / case_name)

        _print_progress(idx, total, case_name)

        if pipeline_2d:
            new_coords, Z_targets, additional_Z_grouped, _ = _BATCH_HOOKS["pipeline_2d"](
                grid_coords, Z,
                z_keys=z_keys,
                fixed_params=fixed_params,
                filter_conditions=filter_conditions,
                max_num=max_num,
                group_cols=group_cols,
                deltas=deltas,
                nan_cost_penalty=nan_cost_penalty,
                auto_split_streams=auto_split_streams,
            )
        else:
            new_coords, Z_targets, additional_Z_grouped, _ = _BATCH_HOOKS["pipeline_1d"](
                grid_coords, Z,
                z_keys=z_keys,
                fixed_params=fixed_params,
                filter_conditions=filter_conditions,
                max_num=max_num,
                group_cols=group_cols,
                deltas=deltas,
                nan_cost_penalty=nan_cost_penalty,
                auto_split_streams=auto_split_streams,
            )

        if prepare_datasets is not None:
            datasets = prepare_datasets(
                new_coords, Z_targets, additional_Z_grouped, z_keys, max_num,
            )
        else:
            datasets = build_datasets_from_pipeline(
                new_coords, Z_targets, additional_Z_grouped, z_keys, max_num,
            )

        data_path = _prepare_plot_data(
            new_coords, data_class="Eigensolution",
            dataset_list=datasets, fixed_params={},
            save_dir=str(ensure_dir(case_dir / "data")),
        )

        grid_key = tuple(combo_dict[d] for d in grid_spec.grid_dims)

        unit_title = None
        if show_unit_titles and grid_spec.grid_dims:
            unit_title = ", ".join(
                f"{d}={_fmt_val(v)}" for d, v in zip(grid_spec.grid_dims, grid_key)
            )

        unit_svg = render_plot_unit(
            data_path=data_path,
            plot_unit=plot_unit,
            case_output_dir=case_dir,
            x_key=x_key,
            max_num=max_num,
            title=unit_title,
            tight_bbox=False,
            formats=formats,
            png_dpi=png_dpi,
            y_key=y_key,
        )

        split_key = tuple(combo_dict[d] for d in grid_spec.split_dims)
        grid_data.setdefault(split_key, {})[grid_key] = unit_svg

    _compose_grid_figures(
        grid_data, grid_spec, plot_unit, output_root,
        base_fixed_params=base_fixed_params,
        show_figure_footer=show_figure_footer,
        png="png" in formats,
        footer_supp=footer_supp,
        footer_filter_conditions=filter_conditions,
        png_dpi=png_dpi,
    )
