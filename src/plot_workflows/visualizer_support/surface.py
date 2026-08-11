"""Compatibility adapters for shared 3D surface rendering.

Matplotlib surface construction lives in Plot Foundation. Callers retain
their historical function signatures and the optional s3dlib smoothing backend used
by finite multi-band plots.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _auto_norm(data: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> Normalize:
    finite = np.asarray(data)[np.isfinite(data)]
    if vmin is None:
        vmin = float(np.min(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.max(finite)) if finite.size else 1.0
    if vmin == vmax:
        delta = 1e-12 if vmin == 0 else abs(vmin) * 1e-9
        vmin -= delta
        vmax += delta
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def _x_first_rgba(rgba: np.ndarray, expected: tuple[int, int], alpha: float) -> np.ndarray:
    """Accept x-first colors and the historical non-square y-first mapper output."""

    colors = np.asarray(rgba, dtype=float)
    if colors.ndim != 3 or colors.shape[-1] not in (3, 4):
        raise ValueError("rgba must have shape (len(x), len(y), 3 or 4)")
    if colors.shape[:2] == expected:
        pass
    elif colors.shape[:2] == expected[::-1] and expected[0] != expected[1]:
        colors = np.transpose(colors, (1, 0, 2))
    else:
        raise ValueError(
            f"rgba grid shape {colors.shape[:2]} must match x-first shape {expected}"
        )
    if colors.shape[-1] == 3:
        colors = np.concatenate(
            [colors, np.full((*expected, 1), alpha, dtype=float)], axis=-1
        )
    else:
        colors = colors.copy()
        colors[..., 3] *= alpha
    return colors


def _apply_collection_options(artists, options: dict[str, Any]) -> None:
    setters = {
        "linewidth": "set_linewidth",
        "linewidths": "set_linewidth",
        "edgecolor": "set_edgecolor",
        "edgecolors": "set_edgecolor",
        "antialiased": "set_antialiased",
        "antialiaseds": "set_antialiased",
    }
    unsupported = sorted(set(options) - set(setters))
    if unsupported:
        raise TypeError(f"unsupported surface options: {', '.join(unsupported)}")
    for artist in artists:
        for option, value in options.items():
            getattr(artist, setters[option])(value)


def plot_advanced_surface(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    z3: Optional[np.ndarray] = None,
    rgba: Optional[np.ndarray] = None,
    *,
    mapping: Dict[str, Any],
    elev: float = 30,
    azim: float = 25,
    x_key: str = "",
    y_key: str = "",
    z_label: str = "",
    rstride: int = 1,
    cstride: int = 1,
    box_aspect: Sequence[float] = (1, 1, 1),
    **kwargs,
) -> Tuple[plt.Axes, ScalarMappable]:
    """Render one physical surface through Plot Foundation.

    ``z1`` is height, ``z2`` is color, and optional ``z3`` is normalized into
 per-point opacity. The public signature remains stable for callers.
    """

    from plot_foundation import (
        MultiSurfacePlotSpec,
        SurfaceLayer,
        SurfaceStyle,
        render_multi_surface_3d,
    )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    expected = (len(x), len(y))
    alpha = float(kwargs.pop("alpha", 1.0))
    shade = bool(kwargs.pop("shade", False))
    colors = _x_first_rgba(rgba, expected, alpha) if rgba is not None else None
    z2_cfg = mapping.get("z2", {})
    z3_cfg = mapping.get("z3", {})
    layer = SurfaceLayer(
        z=z1,
        color_values=z2,
        alpha_values=None if z3 is None else np.asarray(z3, dtype=float),
        rgba=colors,
        alpha=alpha if colors is None else 1.0,
        alpha_vmin=z3_cfg.get("vmin"),
        alpha_vmax=z3_cfg.get("vmax"),
    )
    result = render_multi_surface_3d(
        MultiSurfacePlotSpec(
            x=x,
            y=y,
            layers=(layer,),
            style=SurfaceStyle(
                cmap=mapping.get("cmap", "hot"),
                vmin=z2_cfg.get("vmin"),
                vmax=z2_cfg.get("vmax"),
                rstride=rstride,
                cstride=cstride,
                shade=shade,
            ),
            xlabel=x_key,
            ylabel=y_key,
            zlabel=z_label,
            elev=elev,
            azim=azim,
            box_aspect=tuple(box_aspect),
        ),
        ax=ax,
    )
    _apply_collection_options(result.artists, kwargs)
    return ax, result.artists[0]


def s3d_build_planar_surface_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    rez,
    basetype,
    cmap,
    alpha,
    shade,
    hilite,
    cname: str = "Value",
    geom_scale: float = 1.0,
    z_offset: float = 0.0,
    norm_z2: Optional[Normalize] = None,
):
    """Build the optional smooth s3dlib representation for finite grids."""

    import s3dlib.cmap_utilities as s3dcmap
    import s3dlib.surface as s3d

    expected = (x.size, y.size)
    if z1.shape != expected:
        raise ValueError(f"z1 shape {z1.shape} must be x-first shape {expected}")
    if z2.shape != z1.shape:
        raise ValueError(f"z2 shape {z2.shape} must match z1 shape {z1.shape}")

    z1_yx = np.array(z1, copy=True).T
    z2_yx = np.array(z2, copy=True).T
    surface = s3d.PlanarSurface(rez, basetype=basetype, cmap=cmap)
    surface.cname = cname
    cmap_func = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    dmin = np.nanmin(z2_yx)
    dmax = np.nanmax(z2_yx)
    dspan = dmax - dmin
    if dspan == 0:
        dspan = 1.0
    cmap_i = s3dcmap.op_cmap(
        lambda t: cmap_func(norm_z2(dmin + t * dspan))[:, :3].T,
        rgb=True,
        name=None,
    )
    surface.map_cmap_from_datagrid(z2_yx, cmap=cmap_i)
    surface._facecolor3d[:, 3] = alpha
    surface.map_geom_from_datagrid(z1_yx, scale=geom_scale)

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    surface.transform(scale=[0.5 * (x_max - x_min), 0.5 * (y_max - y_min), 1.0])
    surface.transform(translate=[0.5 * (x_min + x_max), 0.5 * (y_min + y_max), z_offset])
    if shade:
        surface.shade().hilite(hilite)
    return surface


def s3d_plot_multi_surfaces_combined(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z1_list: Sequence[np.ndarray],
    z2_list: Optional[Sequence[np.ndarray]] = None,
    z3_list: Optional[Sequence[np.ndarray]] = None,
    *,
    rez: int = 4,
    basetype: str = "oct1",
    cmap: str = "hot",
    vmin=None,
    vmax=None,
    elev: float = 30,
    azim: float = 25,
    shade: bool = False,
    hilite: float = 0,
    alpha_default: float = 1.0,
) -> Tuple[plt.Axes, Any, ScalarMappable]:
    """Render multiple x-first surfaces with one global color normalization."""

    z1_list = tuple(np.asarray(z, dtype=float) for z in z1_list)
    if not z1_list:
        raise ValueError("z1_list is empty")
    if z2_list is None:
        z2_list = z1_list
    z2_list = tuple(np.asarray(z, dtype=float) for z in z2_list)
    if len(z2_list) != len(z1_list):
        raise ValueError("z2_list length must match z1_list")
    if z3_list is not None:
        z3_list = tuple(np.asarray(z, dtype=float) for z in z3_list)
        if len(z3_list) != len(z1_list):
            raise ValueError("z3_list length must match z1_list")

    use_shared = z3_list is not None or any(
        not np.isfinite(z1).all() or not np.isfinite(z2).all()
        for z1, z2 in zip(z1_list, z2_list)
    )
    if use_shared:
        from plot_foundation import (
            MultiSurfacePlotSpec,
            SurfaceLayer,
            SurfaceStyle,
            render_multi_surface_3d,
        )

        alpha_fields = z3_list if z3_list is not None else (None,) * len(z1_list)
        result = render_multi_surface_3d(
            MultiSurfacePlotSpec(
                x=x,
                y=y,
                layers=tuple(
                    SurfaceLayer(
                        z=z1,
                        color_values=z2,
                        alpha_values=z3,
                        alpha=alpha_default,
                    )
                    for z1, z2, z3 in zip(z1_list, z2_list, alpha_fields)
                ),
                style=SurfaceStyle(cmap=cmap, vmin=vmin, vmax=vmax, shade=shade),
                elev=elev,
                azim=azim,
            ),
            ax=ax,
        )
        return ax, result.artists, result.artists[0]

    all_z2 = np.concatenate([z.ravel() for z in z2_list])
    norm_z2 = _auto_norm(all_z2, vmin=vmin, vmax=vmax)
    combined = None
    for z1, z2 in zip(z1_list, z2_list):
        finite_z = z1[np.isfinite(z1)]
        if not finite_z.size:
            raise ValueError("Some z1 is all non-finite")
        zmin = float(np.min(finite_z))
        zmax = float(np.max(finite_z))
        zrange = zmax - zmin
        surface = s3d_build_planar_surface_from_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            z1,
            z2,
            rez=rez,
            basetype=basetype,
            cmap=cmap,
            alpha=alpha_default,
            shade=shade,
            hilite=hilite,
            geom_scale=zrange,
            z_offset=zmin,
            norm_z2=norm_z2,
        )
        combined = surface if combined is None else combined + surface

    ax.add_collection3d(combined)
    ax.view_init(elev=elev, azim=azim)
    if alpha_default < 1.0:
        combined._edgecolor3d[:, 3] = 0
    mappable = ScalarMappable(norm=norm_z2, cmap=mpl.colormaps.get_cmap(cmap))
    mappable.set_array([])
    return ax, combined, mappable
