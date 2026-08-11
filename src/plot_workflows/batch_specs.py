"""Standard plot functions for batch processing.

1D functions: (plotter, x_key, max_num) -> None.
2D functions: (plotter, x_key, y_key, band_index) -> None.
The framework owns figure creation and saving; these functions only draw.
"""


def plot_colored_band(plotter, x_key, max_num, vmin=2, vmax=8):
    """Gray fill + dynamic color overlay by qlog.

    Set *vmin*/*vmax* to lock the color scale across cases in batch mode.
    """
    for i in range(max_num):
        plotter.plot(index=i, x_key=x_key,
                     z1_key="eigenfreq_real", z2_key="eigenfreq_imag",
                     enable_fill=True, default_color="gray", alpha_fill=0.3, scale=1)
    for i in range(max_num):
        plotter.plot(index=i, x_key=x_key,
                     z1_key="eigenfreq_real", z3_key="qlog", cmap="nipy_spectral",
                     enable_dynamic_color=True, linewidth_base=1,
                     global_color_vmin=vmin, global_color_vmax=vmax)
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()


def plot_plain_band(plotter, x_key, max_num):
    """Plain eigenfrequency lines."""
    for i in range(max_num):
        plotter.plot(index=i, x_key=x_key,
                     z1_key="eigenfreq_real", z2_key="eigenfreq_imag")
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()


def plot_qlog(plotter, x_key, max_num):
    """log10(Q-factor) lines."""
    for i in range(max_num):
        plotter.plot(index=i, x_key=x_key,
                     z1_key="qlog", z2_key="qlog")
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()


def plot_tanchi_qlog(plotter, x_key, max_num):
    """up_tanchi colored by qlog."""
    for i in range(max_num):
        plotter.plot(index=i, x_key=x_key,
                     z1_key="up_tanchi", z3_key="qlog", cmap="nipy_spectral")
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()


def plot_u_factor(plotter, x_key, max_num):
    """-u_factor colored by qlog, log-scale y axis."""
    for i in range(max_num):
        plotter.plot(index=i, x_key=x_key,
                     z1_key="-u_factor", z3_key="qlog", cmap="nipy_spectral")
    plotter.adjust_view_2dim_auto()
    plotter.ax.set_yscale("log")
    plotter.add_annotations()


# =========================================================
# 2D plot functions (TwoDimFieldVisualizer)
# Signature: (plotter, x_key, y_key, band_index, **kwargs) -> None
# =========================================================


def plot_ulog_heatmap(plotter, x_key, y_key, band_index, vmin=-5, vmax=5):
    """ulog field heatmap (RdBu colormap)."""
    plotter.imshow_field_shared(index=band_index, x_key=x_key, y_key=y_key,
                         field_key="ulog", cmap="RdBu", aspect="auto",
                         vmin=vmin, vmax=vmax)


def plot_qlog_heatmap(plotter, x_key, y_key, band_index, vmin=1, vmax=6):
    """qlog field heatmap (hot colormap)."""
    plotter.imshow_field_shared(index=band_index, x_key=x_key, y_key=y_key,
                         field_key="qlog", cmap="hot", aspect="auto",
                         vmin=vmin, vmax=vmax)


def plot_3d_ulog_surface(plotter, x_key, y_key, band_index, vmin=-4, vmax=4):
    """3D surface: eigenfreq_real height colored by ulog."""
    plotter.plot_3d_surface(index=band_index, x_key=x_key, y_key=y_key,
                            z1_key="eigenfreq_real", z2_key="ulog",
                            cmap="RdBu", elev=45, shade=False,
                            vmin=vmin, vmax=vmax)
    for axis_name in ("x", "y"):
        getattr(plotter.ax, f"{axis_name}axis").set_pane_color((1.0, 1.0, 1.0, 0.0))
    plotter.ax.grid(True)
    plotter.ax.set_box_aspect([1, 1, 0.2])
    plotter.add_annotations()
