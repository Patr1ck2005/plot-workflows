"""A-derived plotting lifecycle and Visualizer base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import pickle
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from plot_foundation import profile

from plot_foundation import (
    add_annotations as render_annotations,
    plot_2d_heatmap,
    plot_2d_multiline,
    plot_line_advanced,
    plot_polar_line,
    plot_scatter_advanced,
)


_HOOKS = {
    "json_loader": None,
    "save_name_factory": None,
    "debug": None,
    "saved": None,
    "success": None,
    "warning": None,
    "annotation_renderer": render_annotations,
    "write_temp_outputs": False,
}


def configure_visualizer_hooks(**hooks):
    """Configure consumer IO/logging hooks without subclass duplication."""

    unknown = sorted(set(hooks) - set(_HOOKS))
    if unknown:
        raise ValueError(f"unknown visualizer hooks: {unknown}")
    _HOOKS.update(hooks)


def _emit(name: str, *args, **kwargs):
    callback = _HOOKS.get(name)
    if callback is not None:
        callback(*args, **kwargs)


def emit_visualizer_event(name: str, *args, **kwargs):
    """Emit a configured consumer logging callback for advanced plotters."""

    _emit(name, *args, **kwargs)


@dataclass
class PlotConfig:
    figsize: tuple[float, float] = (4, 6)
    font: str = "Arial"
    fs: int = 9
    save_dir: str = "./rsl"
    show: bool = True
    plot_params: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    dpi: int = 300
    tick_direction: str = "in"
    style_profile: str = "paper"

    def __post_init__(self):
        profile(self.style_profile)
        self.apply()

    @property
    def resolved_profile(self):
        return profile(self.style_profile)

    @property
    def figure_policy(self):
        return self.resolved_profile.policy

    def apply(self):
        plt.rcParams.update(
            {
                "font.size": self.fs,
                "xtick.direction": self.tick_direction,
                "ytick.direction": self.tick_direction,
                "font.family": self.font,
            }
        )

    def update(self, **values):
        for key, value in values.items():
            setattr(self, key, value)
        if "style_profile" in values:
            profile(self.style_profile)
        if {"fs", "font", "tick_direction", "style_profile"} & set(values):
            self.apply()


class BasePlotter(ABC):
    """Shared load -> prepare -> figure -> plot -> annotate -> save lifecycle."""

    def __init__(self, config: PlotConfig | Mapping[str, Any] | None = None, data_path: str | None = None):
        self.config = PlotConfig(**config) if isinstance(config, Mapping) else config or PlotConfig()
        self.data_path = data_path
        self.fig = None
        self.ax = None
        self.raw_datasets = None
        self.data_list = None
        self.data_num = None
        self.coordinates = None
        self.plot_xlims = []
        self.plot_zlims = []
        self.xlim = self.ylim = self.zlim = None

    def re_initialized(self, config=None, data_path=None):
        self.config = PlotConfig(**config) if isinstance(config, Mapping) else config or self.config
        self.data_path = data_path or self.data_path
        self.raw_datasets = None
        self.coordinates = None
        _emit("debug", "Re-initialized data/config; figure and axes retained")
        return self

    def re_initialized_plot(self, config=None):
        if config is not None:
            self.config = PlotConfig(**config) if isinstance(config, Mapping) else config
        self.plot_xlims = []
        self.plot_zlims = []
        return self

    def load_data(self):
        if not self.data_path:
            raise ValueError("data_path is required")
        if self.data_path.endswith(".json"):
            self._load_json()
        elif self.data_path.endswith(".pkl"):
            self._load_pickle()
        else:
            raise ValueError(f"unsupported data file: {self.data_path}")
        _emit("success", "Data loaded")

    def _load_json(self):
        loader = _HOOKS.get("json_loader")
        if loader is not None:
            self.raw_datasets = loader(self.data_path)
        else:
            with open(self.data_path, encoding="utf-8") as handle:
                self.raw_datasets = json.load(handle)

    def _load_pickle(self):
        with open(self.data_path, "rb") as handle:
            self.raw_datasets = pickle.load(handle)
        self.coordinates = self.raw_datasets.get("coords", {})
        self.data_list = self.raw_datasets["data_list"]
        self.data_num = len(self.data_list)
        _emit("debug", f"Pickle keys: {list(self.raw_datasets)}")

    def get_datasets(self):
        return self.raw_datasets

    def get_dataset(self, index):
        return self.raw_datasets[index]

    def get_coordinates(self):
        return self.coordinates

    @abstractmethod
    def prepare_data(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError

    def twin_plot_ax(self, twinx=False, twiny=False):
        if twiny:
            if not hasattr(self, "twiny_ax"):
                self.twiny_ax = self.ax.twiny()
            return self.twiny_ax
        if twinx:
            if not hasattr(self, "twinx_ax"):
                self.twinx_ax = self.ax.twinx()
            return self.twinx_ax
        return self.ax

    def new_2d_fig(self, projection="rectilinear", **kwargs):
        self.config.update(**kwargs)
        options = {"figsize": self.config.figsize}
        if projection == "polar":
            options["subplot_kw"] = {"projection": "polar"}
        self.fig, self.ax = plt.subplots(**options)

    def new_3d_fig(self, **kwargs):
        self.config.update(**kwargs)
        self.fig = plt.figure(figsize=self.config.figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")

    def add_annotations(self):
        if self.config.annotations is None:
            _emit("warning", "annotations are not configured")
        self.fig, self.ax = _HOOKS["annotation_renderer"](self.ax, self.config.annotations)

    def add_twinx_annotations(self):
        self.fig, self.twinx_ax = _HOOKS["annotation_renderer"](self.twinx_ax, self.config.annotations)

    def add_twiny_annotations(self):
        self.fig, self.twiny_ax = _HOOKS["annotation_renderer"](self.twiny_ax, self.config.annotations)

    def adjust_view_2dim(self):
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

    def adjust_view_3dim(self):
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)

    def save_and_show(self, save=True, save_type="svg", custom_name=None, custom_abs_path=None):
        if save:
            params = self.config.plot_params or {}
            factory = _HOOKS.get("save_name_factory")
            if custom_abs_path:
                image_path = custom_abs_path
            elif custom_name:
                image_path = os.path.join(self.config.save_dir, custom_name)
            elif factory is not None:
                image_path = factory(self.config.save_dir, params)
            else:
                image_path = os.path.join(self.config.save_dir, "plot")
            os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
            self.fig.savefig(
                image_path + f".{save_type}", dpi=self.config.dpi,
                bbox_inches="tight", transparent=True,
            )
            _emit("saved", image_path)
            if _HOOKS.get("write_temp_outputs"):
                self.fig.savefig("temp_output.svg", dpi=self.config.dpi, bbox_inches="tight", transparent=True)
                self.fig.savefig("temp_output.png", dpi=self.config.dpi, bbox_inches="tight", transparent=True)
        if self.config.show:
            plt.show()

    def run_full(self):
        self.load_data()
        self.prepare_data()
        self.new_2d_fig()
        self.plot()
        self.add_annotations()
        self.save_and_show()


class ScatterPlotter(BasePlotter):
    def plot_scatter(self, x, z1, **kwargs):
        params = {**(self.config.plot_params or {}), **kwargs}
        ax = self.twin_plot_ax(kwargs.get("twinx", False), kwargs.get("twiny", False))
        self.ax = plot_scatter_advanced(ax, x, z1=z1, z3=z1, **params)


class LinePlotter(BasePlotter):
    def plot_line(self, x, z1, **kwargs):
        params = {**(self.config.plot_params or {}), **kwargs}
        ax = self.twin_plot_ax(kwargs.get("twinx", False), kwargs.get("twiny", False))
        self.ax = plot_line_advanced(ax, x, z1=z1, **params)


class PolarPlotter(BasePlotter):
    def plot_polar(self, theta, radial, **kwargs):
        params = {**(self.config.plot_params or {}), **kwargs}
        self.ax = plot_polar_line(self.ax, theta, radial, **params)
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_thetalim(np.deg2rad(-60), np.deg2rad(60))


class HeatmapPlotter(BasePlotter):
    def plot_heatmap(self, values, x_vals=None, y_vals=None, **kwargs):
        params = {**(self.config.plot_params or {}), **kwargs}
        x_vals = np.arange(values.shape[0]) if x_vals is None else x_vals
        y_vals = np.arange(values.shape[1]) if y_vals is None else y_vals
        self.fig, self.ax = plot_2d_heatmap(self.ax, x_vals, y_vals, values, params)

    def plot_multiline_2d(self, values, x_vals=None, y_vals=None, **kwargs):
        params = {**(self.config.plot_params or {}), **kwargs}
        x_vals = np.arange(values.shape[0]) if x_vals is None else x_vals
        y_vals = np.arange(values.shape[1]) if y_vals is None else y_vals
        self.fig, self.ax = plot_2d_multiline(self.ax, x_vals, y_vals, values, params)

    def show_colorbar(self, **kwargs):
        if self.ax and self.ax.collections:
            mappable = self.ax.collections[0]
        elif self.ax and self.ax.images:
            mappable = self.ax.images[0]
        else:
            raise ValueError("no mappable artist is available for a colorbar")
        colorbar = self.fig.colorbar(mappable, ax=self.ax, **kwargs)
        colorbar.ax.tick_params(labelsize=self.config.fs)


__all__ = [
    "PlotConfig",
    "BasePlotter",
    "ScatterPlotter",
    "LinePlotter",
    "PolarPlotter",
    "HeatmapPlotter",
    "configure_visualizer_hooks",
]
