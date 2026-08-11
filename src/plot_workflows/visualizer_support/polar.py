import numpy as np
from eigenmode_analysis import wrap_angle_pi
from plot_foundation import imshow_phi, imshow_s3


# =========================
# 绘图函数（均“独立成图”）
# =========================

_wrap_angle_pi = wrap_angle_pi


def plot_skyrmion_quiver(ax, x, y, s1, s2, s3, S0=None,
                         step=(4, 4), normalize=True,
                         color=None, cmap='RdBu', clim=(-1, 1),
                         quiver_scale=None, pivot='mid', width=0.004):
    del S0
    from plot_foundation import (
        VectorFieldPlotSpec,
        VectorFieldStyle,
        render_vector_field,
    )

    render_vector_field(
        VectorFieldPlotSpec(
            x=np.asarray(x)[:, 0],
            y=np.asarray(y)[0, :],
            u=np.asarray(s1),
            v=np.asarray(s2),
            color_values=np.asarray(s3) if color is None else None,
            glyph="arrow",
            step=step,
            style=VectorFieldStyle(
                color=color,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                normalize=normalize,
                scale=quiver_scale,
                pivot=pivot,
                width=width,
            ),
        ),
        ax=ax,
    )
    return ax


def plot_polar_quiver(ax, xgrid, ygrid, s1, s2, s3, S0=None,
                      step=(4, 4), normalize=True,
                      color=None, cmap='RdBu', clim=(-1, 1),
                      quiver_scale=None, pivot='mid', width=0.004):
    del S0
    from plot_foundation import (
        VectorFieldPlotSpec,
        VectorFieldStyle,
        render_vector_field,
    )

    phi = _wrap_angle_pi(0.5 * np.arctan2(s2, s1))
    render_vector_field(
        VectorFieldPlotSpec(
            x=np.asarray(xgrid)[:, 0],
            y=np.asarray(ygrid)[0, :],
            u=np.cos(phi),
            v=np.sin(phi),
            color_values=np.asarray(s3) if color is None else None,
            glyph="director",
            step=step,
            style=VectorFieldStyle(
                color=color,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                normalize=normalize,
                scale=quiver_scale,
                pivot=pivot,
                width=width,
            ),
        ),
        ax=ax,
    )
    return ax

# Existing Visualizer and project imports keep these historical function names;
# the renderer implementation lives only in Plot Foundation.
def plot_polarization_ellipses(ax, xgrid, ygrid, s1, s2, s3, S0=None,
                               step=(6, 6), scale=None,
                               cmap='RdBu', clim=(-1.0, 1.0),
                               lw=1.2, alpha=0.9, zorder=2):
    del S0
    from plot_foundation import render_polarization_ellipses

    x = np.asarray(xgrid)[:, 0]
    y = np.asarray(ygrid)[0, :]
    return render_polarization_ellipses(
        x,
        y,
        s1,
        s2,
        s3,
        ax=ax,
        step=step,
        scale=scale,
        cmap=cmap,
        clim=clim,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    ).axes


def plot_on_poincare_sphere(ax, s1, s2, s3, rgba=None, S0=None,
                            step=(4, 4), c_by='rgba', cmap='RdBu', clim=(-1, 1),
                            s=6, alpha=0.9, sphere_style='wire'):
    del S0
    from plot_foundation import render_poincare_scatter

    color_by = c_by.lower()
    if color_by == 'rgba' and rgba is None:
        color_by = 's3'
    return render_poincare_scatter(
        s1,
        s2,
        s3,
        ax=ax,
        rgba=rgba,
        step=step,
        color_by=color_by,
        cmap=cmap,
        clim=clim,
        size=s,
        alpha=alpha,
        sphere_style=sphere_style,
    ).axes
