from abc import ABC

import numpy as np

from .visualization import (
    HeatmapPlotter,
    LinePlotter,
    emit_visualizer_event,
)
from .visualizer_support.polar import *  # noqa: F401,F403
from .visualizer_support.surface import plot_advanced_surface, s3d_plot_multi_surfaces_combined
from .visualizer_support.colors import map_s1s2s3_color, map_complex2rbg
from eigenmode_analysis import (
    sample_fields_along_path,
    skyrmion_density,
)
from plot_foundation import extract_isofreq_paths, plot_iso_contours2D

DEBUG = False


def print_debug(*args, **kwargs):
    emit_visualizer_event("debug", *args, **kwargs)


def print_warning(*args, **kwargs):
    emit_visualizer_event("warning", *args, **kwargs)


def lorenz_func(delta_omega, gamma=1, gamma_nr=0.001):
    return -(gamma) / ((gamma + gamma_nr) - 1j * delta_omega)


def plot_field_splits(
    ax,
    xgrid,
    ygrid,
    s1,
    s2,
    *,
    color_1="limegreen",
    color2="black",
    color_uncertain="orange",
    lw=2.0,
):
    from eigenmode_analysis import extract_field_splits
    from plot_foundation import plot_field_regime_splits

    splits = extract_field_splits(xgrid, ygrid, s1, s2)
    return plot_field_regime_splits(
        ax,
        splits,
        color_phi0=color_1,
        color_phi90=color2,
        color_uncertain=color_uncertain,
        linewidth=lw,
    )


class BandPlotterOneDim(LinePlotter, ABC):
    """能带图骨架"""

    def prepare_data(self, x_key='k') -> None:  # 手动重写：NaN过滤
        self.x_vals_list = []
        self.y_vals_list = []
        for raw_data in self.raw_datasets["data_list"]:
            sub = raw_data['eigenfreq']
            mask = np.isnan(sub)
            self.x_vals = self.coordinates[x_key]
            if np.any(mask):
                print_warning("NaN 数据已移除")
                self.y_vals = sub[~mask]
                temp_x = self.x_vals[~mask]
            else:
                self.y_vals = sub
                temp_x = self.x_vals
            self.x_vals_list.append(temp_x)
            self.y_vals_list.append(self.y_vals)

    def plot(self) -> None:  # 重写：整体+循环填充
        pass

    def plot_thick_bg(self) -> None:  # 重写：整体+循环填充
        params_bg = {
            'enable_fill': True,
            'gradient_fill': False,
            'enable_dynamic_color': False,
            'cmap': None,
            'add_colorbar': False,
            'global_color_vmin': 1, 'global_color_vmax': 8,
            'default_color': 'gray', 'alpha_fill': 0.3,
            'edge_color': 'none',
            'gradient_direction': 'z3',
            'linewidth_base': 0,
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_bg)  # 填充
            widths = np.abs(y.imag)
            y_mins.append(np.min(y.real - widths))
            y_maxs.append(np.max(y.real + widths))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)

    def plot_colored_bg(self, vmin=0, vmax=5e-3, alpha=1, y_margin=0.02) -> None:  # 重写：整体+循环填充
        params = {
            'enable_fill': True,
            'gradient_fill': True,
            # 'cmap': 'magma',
            'cmap': 'magma',
            'add_colorbar': False,
            'global_color_vmin': vmin, 'global_color_vmax': vmax,
            'default_line_color': 'gray', 'alpha_fill': alpha,
            'edge_color': 'none',
            'gradient_direction': 'z3',
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            self.plot_line(x, z1=y.real, z2=y.imag, z3=y.imag, **params)  # 填充
            widths = np.abs(y.imag)
            y_mins.append(np.min(y.real - widths))
            y_maxs.append(np.max(y.real + widths))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * (1-y_margin), np.nanmax(y_maxs) * (1+y_margin))

    def plot_colored_line(self, vmin=2, vmax=7, cmap='magma', y_margin=0.02) -> None:  # 重写：整体+循环填充
        params_line = {
            'enable_fill': False,
            'gradient_fill': False,
            'enable_dynamic_color': True,
            'cmap': cmap,
            'add_colorbar': False,
            'global_color_vmin': vmin, 'global_color_vmax': vmax,
            'default_line_color': 'gray', 'alpha_fill': 1,
            'linewidth_base': 2,
            'edge_color': 'none',
            'alpha_line': 1
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_line)  # 填充
            y_mins.append(np.min(y.real))
            y_maxs.append(np.max(y.real))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * (1-y_margin), np.nanmax(y_maxs) * (1+y_margin))

    def plot_ordered_line(self, y_margin=0.02, cmap=None) -> None:  # 重写：整体+循环填充
        params_line = {
            'enable_fill': False,
            'gradient_fill': False,
            'enable_dynamic_color': False,
            'cmap': cmap,
            'add_colorbar': False,
            'default_line_color': False, 'alpha_fill': 1,
            'linewidth_base': 2,
            'edge_color': 'none',
            'alpha_line': 0.75,
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_line, index=i)  # 填充
            y_mins.append(np.min(y.real))
            y_maxs.append(np.max(y.real))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * (1-y_margin), np.nanmax(y_maxs) * (1+y_margin))
        # self.ax.legend()

    def plot_ordered_qfactor(self, y_margin=0.02) -> None:  # 重写：整体+循环填充
        params_line = {
            'enable_fill': False,
            'gradient_fill': False,
            'enable_dynamic_color': False,
            'cmap': None,
            'add_colorbar': False,
            'default_line_color': False, 'alpha_fill': 1,
            'linewidth_base': 2,
            'edge_color': 'none',
            'alpha_line': 0.75,
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=Qfactor, z2=Qfactor, z3=Qfactor_log, **params_line, index=i)  # 填充
            y_mins.append(np.min(Qfactor))
            y_maxs.append(np.max(Qfactor))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * (1-y_margin), np.nanmax(y_maxs) * (1+y_margin))
        # self.ax.legend()

    def plot_diffraction_cone(self, env_n=1, scale=1, upper_limit=1, color='lightgreen', alpha=0.2) -> None:
        # 绘制衍射锥线
        kx = self.x_vals
        c = 1  # 光速归一化
        diffraction_cone = (1 - c * np.abs(kx)) / env_n
        # 填充衍射极限以上的区域
        self.ax.fill_between(kx, diffraction_cone*scale, upper_limit, color=color, alpha=alpha, edgecolor='none', zorder=0)


class MomentumSpaceEigenPolarizationPlotter(HeatmapPlotter, ABC):
    """极化图骨架"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print_warning("This class has been abandoned. Use 'MomentumSpaceEigenVisualizer' instead.")
    def prepare_data(self) -> None:  # 手动重写：NaN过滤
        self.m1 = self.coordinates['m1']
        self.m2 = self.coordinates['m2']
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')
        self.data_list = self.raw_datasets["data_list"]
        self.eigenfreq_list = [self.data_list[i]['eigenfreq'] for i in range(len(self.data_list))]
        self.s1_list = [self.data_list[i]['s1'] for i in range(self.data_num)]
        self.s2_list = [self.data_list[i]['s2'] for i in range(self.data_num)]
        self.s3_list = [self.data_list[i]['s3'] for i in range(self.data_num)]
        self.qlog_list = [self.data_list[i]['qlog'] for i in range(self.data_num)]

    def interpolate_data(self, factor=2, order=3, mode='constant') -> None:
        # 对所有数据进行插值
        from scipy.ndimage import zoom
        self.m1 = self.coordinates['m1']
        self.m2 = self.coordinates['m2']
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')
        new_m1 = np.linspace(self.m1.min(), self.m1.max(), len(self.m1) * factor)
        new_m2 = np.linspace(self.m2.min(), self.m2.max(), len(self.m2) * factor)
        self.m1 = new_m1
        self.m2 = new_m2
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')

        def interp_field(field):
            return zoom(field, zoom=factor, order=order, mode=mode)

        self.eigenfreq_list = [interp_field(self.data_list[i]['eigenfreq']) for i in range(len(self.data_list))]
        self.s1_list = [interp_field(self.data_list[i]['s1']) for i in range(self.data_num)]
        self.s2_list = [interp_field(self.data_list[i]['s2']) for i in range(self.data_num)]
        self.s3_list = [interp_field(self.data_list[i]['s3']) for i in range(self.data_num)]
        self.qlog_list = [interp_field(self.data_list[i]['qlog']) for i in range(self.data_num)]

    def plot(self) -> None:
        pass

    def prepare_chi_phi_data(self) -> None:
        # 通过s123计算phi, tanchi
        self.phi_list = []
        self.tanchi_list = []
        for i in range(self.data_num):
            s1 = self.s1_list[i]
            s2 = self.s2_list[i]
            s3 = self.s3_list[i]
            # 计算 |cos(2χ)|，在 χ ∈ [-π/4, π/4] 的主值范围内 cos(2χ)≥0，因此取正根即可
            cos2chi_abs = np.sqrt(np.maximum(s1 ** 2 + s2 ** 2, 0.0))
            # 2χ = atan2(sin2χ, cos2χ)，其中 cos2χ≥0（主值保证）
            chi = 0.5 * np.arctan2(s3, cos2chi_abs)
            # phi 的 2φ = atan2(s2, s1)，自动给出正确象限；纯圆时 s1=s2=0，phi 无定义
            phi = (0.5 * np.arctan2(s2, s1)) % np.pi
            self.phi_list.append(phi % np.pi)
            self.tanchi_list.append(np.tan(chi))

    def sample_along_isofreq(self, index=0, level=None, show=False):
        assert level is not None, "请提供等频线频率值 level"
        m1f, m2f = self.m1, self.m2
        phi_f, tanchi_f = self.phi_list[index], self.tanchi_list[index]
        Z_f = self.eigenfreq_list[index]
        qlog_f = np.log10(np.where(Z_f.imag != 0, np.abs(Z_f.real / (2 * Z_f.imag)), 1e10))
        paths = extract_isofreq_paths(m1f, m1f, Z_f, level=level)
        # 沿每条等频线插值采样
        fields = {'phi': phi_f, 'tanchi': tanchi_f, 'qlog': qlog_f, 'freq': Z_f.real}
        samples_list = []
        for p in paths:
            samp = sample_fields_along_path(m1f, m2f, fields, p, npts=400)
            samples_list.append(samp)
        if show:
            # 画曲线
            fig, ax = plt.subplots(figsize=(1.25, 1.25))
            ax.plot(samp['s'], samp['phi'])
            fig, ax = plt.subplots(figsize=(1.25, 1.25))
            ax.plot(samp['s'], samp['tanchi'])
            fig, ax = plt.subplots(figsize=(1.25, 1.25))
            ax.plot(samp['s'], samp['qlog'])
            plt.show()
        return samples_list

    def compute_cross_polarization_conversion(self, index=0, freq=None):
        cross_conversion = np.exp(-2j*self.phi_list[index])*lorenz_func(delta_omega=(freq-self.eigenfreq_list[index].real), gamma=self.eigenfreq_list[index].imag, gamma_nr=0)
        self.cross_conversion = cross_conversion


class MomentumSpaceSpectrumPlotter(HeatmapPlotter, ABC):
    """光谱图骨架"""
    def prepare_data(self) -> None:  # 手动重写：NaN过滤
        self.m1 = self.coordinates['m1']
        self.m2 = self.coordinates['m2']
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')
        self.data_list = self.raw_datasets["data_list"]
        self.s11_list = [self.data_list[i]['s11'] for i in range(self.data_num)]
        self.s21_list = [self.data_list[i]['s21'] for i in range(self.data_num)]

    def prepare_helical_basis(self) -> None:
        pass

    def plot(self) -> None:
        pass

    def plot_skyrmion_analysis(self, index) -> None:
        pass

    def get_advanced_color_mapping(self, index) -> np.ndarray:
        rgb = map_s1s2s3_color(self.s1_list[index], self.s2_list[index], self.s3_list[index])
        return rgb

    def imshow_advanced_color_mapping(self, index) -> None:
        rgb = self.get_advanced_color_mapping(index)
        self.ax.imshow(np.transpose(rgb, (1, 0, 2)),
                       extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal')

    def imshow_s11(self, index=0, **kwargs) -> None:
        # if 'cmap' not in kwargs:
        #     kwargs['cmap'] = 'RdBu'
        rgb = map_complex2rbg(self.s11_list[index])
        self.ax.imshow(rgb, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal', **kwargs)

    def imshow_s21(self, index=0, **kwargs) -> None:
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'RdBu'
        rgb = map_complex2rbg(self.s21_list[index])
        self.ax.imshow(rgb, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal', **kwargs)


class OneDimFieldVisualizer(LinePlotter, ABC):
    """一维曲线图骨架"""
    def prepare_data(self) -> None:
        pass

    def plot(self, index, x_key, z1_key, z2_key=None, z3_key=None, **params_bg) -> None:
        x = np.squeeze(self.coordinates[x_key])
        z1 = np.squeeze(self.raw_datasets["data_list"][index][z1_key])
        z2 = np.squeeze(self.raw_datasets["data_list"][index][z2_key]) if z2_key is not None else None
        z3 = np.squeeze(self.raw_datasets["data_list"][index][z3_key]) if z3_key is not None else None
        self.plot_line(x, z1=z1, z2=z2, z3=z3, **params_bg, index=index)
        self.plot_xlims.append((np.nanmin(x), np.nanmax(x)))
        self.plot_zlims.append((np.nanmin(z1), np.nanmax(z1)))

    def plot_shared(
            self, index, x_key, z1_key, z2_key=None, z3_key=None, **plot_params
    ):
        """Render supported line styles through ``plot-foundation``."""

        from .visualizer_adapter import plot_visualizer_line_shared

        return plot_visualizer_line_shared(
            self,
            index=index,
            x_key=x_key,
            z1_key=z1_key,
            z2_key=z2_key,
            z3_key=z3_key,
            **plot_params,
        )

    def scatter(self, index, x_key, z1_key, z2_key=None, z3_key=None, **params_bg) -> None:
        x = np.squeeze(self.coordinates[x_key])
        z1 = np.squeeze(self.raw_datasets["data_list"][index][z1_key])
        z2 = np.squeeze(self.raw_datasets["data_list"][index][z2_key]) if z2_key is not None else None
        z3 = np.squeeze(self.raw_datasets["data_list"][index][z3_key]) if z3_key is not None else None
        self.ax.scatter(x, z1, **params_bg)
        self.plot_xlims.append((np.nanmin(x), np.nanmax(x)))
        self.plot_zlims.append((np.nanmin(z1), np.nanmax(z1)))

    def adjust_view_2dim_auto(self) -> None:
        if self.plot_xlims and self.plot_zlims:
            x_min = min(x[0] for x in self.plot_xlims)
            x_max = max(x[1] for x in self.plot_xlims)
            z_min = np.nanmin([z[0] for z in self.plot_zlims])
            z_max = np.nanmax([z[1] for z in self.plot_zlims])
            self.xlim = (x_min, x_max)
            self.ylim = (z_min, z_max)
        print_debug("自动调整视图范围：")
        print_debug(f"xlim = {self.xlim}, ylim = {self.ylim}")
        self.adjust_view_2dim()


class TwoDimFieldVisualizer(HeatmapPlotter, ABC):
    """
    通用二维参数空间可视化器
    - 不理解物理意义
    - 只管理坐标、字段和绘制策略
    """
    def prepare_data(self) -> None:  # 手动重写
        pass

    def _get_coord_field(self, index, key, x_key, y_key, x_key_lim=None, y_key_lim=None):
        """
        获取字段；如果不存在且 key 是可派生物理量，则即时计算
        """
        return self.coordinates[x_key], self.coordinates[y_key], self.raw_datasets["data_list"][index][key]

    def plot(self) -> None:
        pass

    def interpolate_data(
            self, index, x_key, y_key, factor=2, field_keys=None, order=3,
            mode='constant',
    ):
        """
        对指定二维字段做统一插值，并同步更新坐标。

        mode : str, scipy.ndimage.zoom 的边界模式，默认 'constant'。
               场在边界外不趋零时需显式设为 'reflect'。
        """
        from scipy.ndimage import zoom

        # ---- 坐标 ----
        x = self.coordinates[x_key]
        y = self.coordinates[y_key]

        new_x = np.linspace(x.min(), x.max(), len(x) * factor)
        new_y = np.linspace(y.min(), y.max(), len(y) * factor)

        self.coordinates[x_key] = new_x
        self.coordinates[y_key] = new_y

        # ---- 自动识别字段 ----
        if field_keys is None:
            sample = self.raw_datasets["data_list"][index]
            field_keys = [
                k for k, v in sample.items()
                if isinstance(v, np.ndarray) and v.ndim == 2
            ]

        # ---- 插值函数 ----
        def interp_field(field):
            return zoom(field, zoom=factor, order=order, mode=mode)

        # ---- 批量插值 ----
        for data in self.raw_datasets["data_list"]:
            for key in field_keys:
                data[key] = interp_field(data[key])

    def imshow_field(
            self, index, x_key, y_key, field_key, **kwargs
    ):
        # x = self.coordinates[x_key]
        # y = self.coordinates[y_key]
        # z1 = self.raw_datasets["data_list"][index][field_key]
        x, y, z1 = self._get_coord_field(index, field_key, x_key, y_key)
        im = self.ax.imshow(
            z1.T,
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin='lower',
            **kwargs
        )
        return im

    def imshow_field_shared(
            self, index, x_key, y_key, field_key, **kwargs
    ):
        """Render a scalar field through Plot Foundation."""

        from .visualizer_adapter import plot_visualizer_heatmap_shared

        return plot_visualizer_heatmap_shared(
            self,
            index=index,
            x_key=x_key,
            y_key=y_key,
            field_key=field_key,
            **kwargs,
        )

    def multiline_field(
            self, index, x_key, y_key, field_key, **kwargs
    ):
        x, y, z1 = self._get_coord_field(index, field_key, x_key, y_key)
        im = self.plot_multiline_2d(z1, x, y, **kwargs)
        return im

    def plot_iso_contours2D(self, index, x_key, y_key, levels, colors, field_key, cmap=None, linewidths=1.0) -> None:
        if cmap is not None:
            # 使用 colormap 生成颜色列表
            from matplotlib import cm
            colormap = cm.get_cmap(cmap, len(levels))
            colors = [colormap(i) for i in range(len(levels))]
        _, _, z = self._get_coord_field(index, field_key, x_key, y_key)
        self.ax = plot_iso_contours2D(
            self.ax, self.coordinates[x_key], self.coordinates[y_key], z, levels=levels,
            colors=colors,
            linewidths=linewidths
        )

    def plot_3d_surface(
            self, index, x_key, y_key, z1_key, z2_key=None, z3_key=None, cmap='hot', vmin=None, vmax=None, **kwargs
    ) -> None:
        x = self.coordinates[x_key]
        y = self.coordinates[y_key]
        z1 = self.raw_datasets["data_list"][index][z1_key]
        z2 = self.raw_datasets["data_list"][index][z2_key] if z2_key is not None else z1
        z3 = self.raw_datasets["data_list"][index][z3_key] if z3_key is not None else None
        mapping = {
            'cmap': cmap,
            'z2': {'vmin': vmin, 'vmax': vmax},  # 可选；未给则自动取数据范围
            # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
        }
        self.ax, mappable = plot_advanced_surface(
            self.ax, x=x, y=y,
            mapping=mapping,
            z1=z1,
            z2=z2,
            z3=z3,
            **kwargs
        )

    def plot_3d_surfaces(
            self, indices, x_key, y_key, z1_key, z2_key=None, z3_key=None, cmap='hot', vmin=None, vmax=None, **kwargs
    ) -> None:
        x = self.coordinates[x_key]
        y = self.coordinates[y_key]
        z1_lst = [self.raw_datasets["data_list"][i][z1_key] for i in indices]
        # for color rending
        z2_lst = [self.raw_datasets["data_list"][i][z2_key] for i in indices] if z2_key is not None else z1_lst
        # for alpha rendering
        z3_lst = [self.raw_datasets["data_list"][i][z3_key] for i in indices] if z3_key is not None else None
        self.ax, combined_surface, mappable = s3d_plot_multi_surfaces_combined(
            self.ax,
            x=x, y=y,
            z1_list=z1_lst,
            z2_list=z2_lst,
            z3_list=z3_lst,
            rez=4,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            **kwargs
        )

    def get_advanced_color_mapping(self, index, s1_key="s1", s2_key="s2", s3_key="s3") -> np.ndarray:
        s1 = self.raw_datasets["data_list"][index][s1_key]
        s2 = self.raw_datasets["data_list"][index][s2_key]
        s3 = self.raw_datasets["data_list"][index][s3_key]
        rgb = map_s1s2s3_color(s1, s2, s3)
        return rgb

    def imshow_advanced_color_mapping(self, index, x_key, y_key) -> None:
        rgb = self.get_advanced_color_mapping(index)
        x = self.coordinates[x_key]
        y = self.coordinates[y_key]
        self.ax.imshow(
            np.transpose(rgb, (1, 0, 2)),
            extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', aspect='equal'
        )

    def imshow_compare_datas(self, index_A, index_B, field_key, x_key="m1", y_key="m2", mode="difference", **kwargs) -> None:
        """
        比较多个数据集的同一字段，输出差值
        """
        x = self.coordinates[x_key]
        y = self.coordinates[y_key]
        z1_A = self.raw_datasets["data_list"][index_A][field_key]
        z1_B = self.raw_datasets["data_list"][index_B][field_key]
        if mode == "difference":
            z = abs(z1_A - z1_B)
        else:
            raise ValueError(f"Unknown mode '{mode}' for data comparison.")
        self.ax.imshow(
            z.T,
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin='lower',
            **kwargs
        )


class MomentumSpaceEigenVisualizer(TwoDimFieldVisualizer):
    def imshow_field(
            self, x_key='m1', y_key='m2', **kwargs
    ) -> None:
        super().imshow_field(x_key=x_key, y_key=y_key, **kwargs)

    def imshow_advanced_color_mapping(
            self, x_key='m1', y_key='m2', **kwargs
    ) -> None:
        super().imshow_advanced_color_mapping(x_key=x_key, y_key=y_key, **kwargs)

    def plot_3d_surfaces(
            self, indices, z1_key, z2_key=None, x_key='m1', y_key='m2', z3_key=None, **kwargs
    ) -> None:
        super().plot_3d_surfaces(
            indices, x_key, y_key, z1_key, z2_key, z3_key, **kwargs
        )

    def plot_3d_surface(
            self, index, z1_key, x_key='m1', y_key='m2', **kwargs
    ) -> None:
        super().plot_3d_surface(
            index, x_key, y_key, z1_key, **kwargs
        )

    def _get_coord_field(self, index, key, x_key='m1', y_key='m2', x_key_lim=None, y_key_lim=None):
        """
        获取字段；如果不存在且 key 是可派生物理量，则即时计算
        """
        # check coordinates
        if x_key not in self.coordinates or y_key not in self.coordinates:
            raise KeyError(f"Coordinates '{x_key}' or '{y_key}' not found.")
        # filter by limits
        if x_key_lim is not None or y_key_lim is not None:
            _x, _y = self.coordinates[x_key], self.coordinates[y_key]
            mask_x = np.ones_like(_x, dtype=bool)
            mask_y = np.ones_like(_y, dtype=bool)
            if x_key_lim is not None:
                mask_x = (_x >= x_key_lim[0]) & (_x <= x_key_lim[1])
            if y_key_lim is not None:
                mask_y = (_y >= y_key_lim[0]) & (_y <= y_key_lim[1])
            # apply mask to coordinates
            x = _x[mask_x]
            y = _y[mask_y]
            # apply mask to data
            data = self.raw_datasets["data_list"][index].copy()
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    data[k] = v[np.ix_(mask_x, mask_y)]
        else:
            x, y = self.coordinates[x_key], self.coordinates[y_key]
            data = self.raw_datasets["data_list"][index]

        if key in data:
            return x, y, data[key]

        # ===== 派生字段 =====
        if key in ("phi", "tanchi"):
            s1, s2, s3 = data["s1"], data["s2"], data["s3"]
            cos2chi = np.sqrt(np.maximum(s1 ** 2 + s2 ** 2, 0.0))
            chi = 0.5 * np.arctan2(s3, cos2chi)
            phi = (0.5 * np.arctan2(s2, s1)) % np.pi
            data["phi"] = phi
            data["tanchi"] = np.tan(chi)
            return x, y, data[key]

        if key == "qlog":
            imag = data["eigenfreq_imag"]
            ratio = np.where(imag != 0, np.abs(data["eigenfreq_real"] / (2 * imag)), 1e10)
            qlog = np.log10(ratio)
            data["qlog"] = qlog
            return x, y, qlog

        if key in ("eigenfreq_real", "eigenfreq_imag") and key not in data:
            data["eigenfreq_real"] = data["eigenfreq"].real
            data["eigenfreq_imag"] = data["eigenfreq"].imag
            return x, y, data[key]

        raise KeyError(f"Field '{key}' not found and not derivable.")

    def _mesh(self, x_key="m1", y_key="m2"):
        x = self.coordinates[x_key]
        y = self.coordinates[y_key]
        return np.meshgrid(x, y, indexing="ij")

    def plot_polarization_ellipses(
            self, index, x_key="m1", y_key="m2", step=(1, 1), scale=1e-2, cmap="coolwarm",
            s1_key='s1', s2_key='s2', s3_key='s3'
    ):
        Mx, My = self._mesh(x_key, y_key)
        _, _, s1 = self._get_coord_field(index, s1_key)
        _, _, s2 = self._get_coord_field(index, s2_key)
        _, _, s3 = self._get_coord_field(index, s3_key)

        self.ax = plot_polarization_ellipses(
            self.ax,
            Mx, My, s1, s2, s3,
            step=step,
            scale=scale,
            cmap=cmap,
            alpha=1,
            lw=1,
        )
        # 重新设置画布的视角
        self.ax.set_xlim(Mx.min(), Mx.max())
        self.ax.set_ylim(My.min(), My.max())

    def plot_iso_contours2D(self, index, levels, colors, field_key, x_key='m1', y_key='m2', cmap=None, linewidths=1.0) -> None:
        super().plot_iso_contours2D(index, x_key, y_key, levels, colors, field_key, cmap, linewidths)

    def sample_along_iso(self, index, level, field_key='eigenfreq_real', sample_keys=None,
                         npts=400, x_key='m1', y_key='m2'):
        """沿等值线采样字段。"""
        from plot_foundation import extract_isofreq_paths
        from eigenmode_analysis import sample_fields_along_path

        x, y, Z_iso = self._get_coord_field(index, field_key, x_key, y_key)

        if sample_keys is None:
            sample_keys = ['phi', 'tanchi', 'qlog']

        fields = {}
        for key in sample_keys:
            _, _, data = self._get_coord_field(index, key, x_key, y_key)
            fields[key] = data

        paths = extract_isofreq_paths(x, y, Z_iso, level=level)
        samples_list = []
        for p in paths:
            samp = sample_fields_along_path(x, y, fields, p, npts=npts)
            samples_list.append(samp)
        return samples_list

    def _round_path(self, center, radius, num=360):
        theta = np.linspace(0, 2 * np.pi, num, endpoint=False)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        s = radius * theta  # 或者弧长
        return x, y, s

    def sample_along_round_path(
            self,
            index,
            center=(0.0, 0.0),
            radius=0.05,
            num=360,
            field_keys=("phi",),
            x_key="m1",
            y_key="m2",
            method="direct",
    ):
        """
        在 (m1, m2) 空间沿圆形路径采样字段

        Returns
        -------
        list[dict]
            每个 dict 是一个 band / index 的采样结果
        """
        from scipy.interpolate import RegularGridInterpolator

        # ---- 坐标 ----
        x_grid = self.coordinates[x_key]
        y_grid = self.coordinates[y_key]

        # ---- 路径 ----

        if method == 'direct':
            # ---- 直接索引 ----
            xp, yp, s = self._round_path(center, radius, num)
            result = {"s": s}
            for key in field_keys:
                x, y, field = self._get_coord_field(index, key)
                # 找到最近的网格点索引
                xi = np.abs(x[:, None] - xp[None, :]).argmin(axis=0)
                yi = np.abs(y[:, None] - yp[None, :]).argmin(axis=0)
                result[key] = field[xi, yi]
        else:
            xp, yp, s = self._round_path(center, radius, num)
            result = {"s": s}
            # ---- 构造插值器 ----
            for key in field_keys:
                _, _, field = self._get_coord_field(index, key)

                interp = RegularGridInterpolator(
                    (x_grid, y_grid),
                    field,
                    method=method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

                pts = np.stack([xp, yp], axis=-1)
                result[key] = interp(pts)

        return [result]

    def plot_field_along_round_path(
            self,
            index,
            center,
            radius,
            field_key="phi",
            num=360,
            normalize_s=True,
            **kwargs
    ):
        """
        沿圆形路径采样并绘制 1D 字段

        Returns
        -------
        dict
            采样结果（包含 's' 和 field_key）
        """

        samples = self.sample_along_round_path(
            index=index,
            center=center,
            radius=radius,
            num=num,
            field_keys=(field_key,),
        )

        sample = samples[0]
        s = sample["s"]
        y = sample[field_key]

        if normalize_s:
            s = s / np.nanmax(s)

        self.ax.plot(s, y, **kwargs)

        # print max/min value
        if DEBUG:
            print(f"Max {field_key}: {np.nanmax(y)}, Min {field_key}: {np.nanmin(y)}")
        return sample

    def plot_on_poincare_sphere_along_around_path(
            self,
            index,
            center,
            radius,
            num=360,
            cmap='rainbow',
            sphere_style='wire',
            arrow_length_ratio=None,
            lw=1,
            **kwargs
    ):
        """
        沿圆形路径采样并绘制 Poincare 球

        Returns
        -------
        dict
            采样结果（包含 's' 和 field_key）
        """

        # 背景球
        u = np.linspace(0, 2 * np.pi, 120)
        v = np.linspace(0, np.pi, 60)
        X = np.outer(np.cos(u), np.sin(v))
        Y = np.outer(np.sin(u), np.sin(v))
        Z = np.outer(np.ones_like(u), np.cos(v))
        if sphere_style == 'surface':
            self.ax.plot_surface(X, Y, Z, rstride=4, cstride=4, color='lightgray', alpha=0.15, linewidth=0)
        else:
            self.ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, color='lightgray', linewidth=0.5, alpha=0.6)

        samples = self.sample_along_round_path(
            index=index,
            center=center,
            radius=radius,
            num=num,
            field_keys=("s1", "s2", "s3"),
        )

        sample = samples[0]
        s1 = sample["s1"]
        s2 = sample["s2"]
        s3 = sample["s3"]

        # self.ax.plot(s1, s2, s3, lw=2)
        # self.ax.scatter(s1, s2, s3, c=np.linspace(0, 1, len(s1)), cmap=cmap, s=15)
        # plot colored line with arrows
        self.ax.set_xlabel(r'$s_1$')
        self.ax.set_ylabel(r'$s_2$')
        self.ax.set_zlabel(r'$s_3$')
        # 绘制3D渐变线
        colors = cm.get_cmap(cmap)(np.linspace(0, 1, len(s1)))
        if arrow_length_ratio is None:
            for i in range(len(s1) - 1):
                self.ax.plot(
                    s1[i:i + 2], s2[i:i + 2], s3[i:i + 2],
                    color=colors[i], lw=lw
                )
        else:
            # 绘制带方向箭头的3D渐变线
            for i in range(len(s1) - 1):
                self.ax.quiver(
                    s1[i], s2[i], s3[i],
                    s1[i + 1] - s1[i], s2[i + 1] - s2[i], s3[i + 1] - s3[i],
                    color=colors[i], arrow_length_ratio=arrow_length_ratio, lw=lw
                )
        self.ax.set_box_aspect([1, 1, 1])

        return sample

    def plot_field_regimes(
            self, index, x_key="m1", y_key="m2", z_key="phi", colors=("lightcoral", "white")
    ):
        _, _, z = self._get_coord_field(index, z_key)
        self.ax.contourf(
            self.coordinates[x_key], self.coordinates[y_key], (np.sin(2 * z.T) > 0),
            levels=[-0.5, 0.5, 1.5], colors=colors, alpha=0.5,
        )

    def plot_field_splits(self, index, x_key="m1", y_key="m2", s1_key="s1", s2_key="s2"):
        _, _, s1 = self._get_coord_field(index, s1_key)
        _, _, s2 = self._get_coord_field(index, s2_key)
        self.ax = plot_field_splits(
            self.ax, self.coordinates[x_key], self.coordinates[y_key], s1, s2, lw=1,
        )

    def imshow_skyrmion_density(self, index, cmap='bwr') -> np.ndarray:
        _, _, s1 = self._get_coord_field(index, "s1")
        _, _, s2 = self._get_coord_field(index, "s2")
        _, _, s3 = self._get_coord_field(index, "s3")

        m1 = self.coordinates["m1"]
        m2 = self.coordinates["m2"]
        nsk = skyrmion_density(s1, s2, s3, x=m1, y=m2)

        self.ax.imshow(
            nsk.T,
            extent=(m1.min(), m1.max(), m2.min(), m2.max()),
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        return nsk

    def plot_on_poincare_sphere(
            self, index, step=(1, 1), sx_key="s1", sy_key="s2", sz_key="s3",
            x_key="m1", y_key="m2", x_key_lim=None, y_key_lim=None,
            cmap='RdBu', clim=(-1, 1), s=8, alpha=0.9, **kwargs
    ):
        x, y, s1 = self._get_coord_field(index, sx_key, x_key, y_key, x_key_lim, y_key_lim)
        _, _, s2 = self._get_coord_field(index, sy_key, x_key, y_key, x_key_lim, y_key_lim)
        _, _, s3 = self._get_coord_field(index, sz_key, x_key, y_key, x_key_lim, y_key_lim)
        # 一个临时的着色方法: 根据xy坐标到固定点的距离着色
        point_x = 0.1156
        # point_x = 0.1165
        point_y = 0.0
        X, Y = np.meshgrid(x, y, indexing='ij')
        distance = np.sqrt((X - point_x) ** 2 + (Y - point_y) ** 2)
        # 通过cmap映射距离到颜色
        norm = plt.Normalize(vmin=0, vmax=0.015)
        rgba = plt.get_cmap(cmap)(norm(distance))

        self.ax = plot_on_poincare_sphere(
            self.ax,
            s1, s2, s3,
            rgba=rgba,
            S0=None,
            step=step,
            # c_by=sz_key,
            c_by='rgba',
            cmap=cmap,
            clim=clim,
            s=s,
            alpha=alpha,
            **kwargs
        )

    def plot_skyrmion_quiver(
            self, index, step=(6, 6), s1_key="s1", s2_key="s2", s3_key="s3", color=None, cmap='RdBu',
            clim=(-1, 1), width=0.006
    ) -> None:
        Mx, My = self._mesh()
        _, _, s1 = self._get_coord_field(index, s1_key)
        _, _, s2 = self._get_coord_field(index, s2_key)
        _, _, s3 = self._get_coord_field(index, s3_key)
        self.ax = plot_skyrmion_quiver(
            self.ax, Mx, My,
            s1, s2, s3,
            step=step,
            normalize=True,
            color=color,
            cmap=cmap,
            clim=clim,
            quiver_scale=None,
            width=width,
        )

    def plot_polar_quiver(
            self, index, step=(6, 6), s1_key="s1", s2_key="s2", s3_key="s3", color=None, cmap='RdBu',
            clim=(-1, 1), width=0.003
    ) -> None:
        Mx, My = self._mesh()
        _, _, s1 = self._get_coord_field(index, s1_key)
        _, _, s2 = self._get_coord_field(index, s2_key)
        _, _, s3 = self._get_coord_field(index, s3_key)
        self.ax = plot_polar_quiver(
            self.ax, Mx, My,
            s1, s2, s3,
            step=step,
            normalize=True,
            color=color,
            cmap=cmap,
            clim=clim,
            quiver_scale=None,
            width=width,
        )
