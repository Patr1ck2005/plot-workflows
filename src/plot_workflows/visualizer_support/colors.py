import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def map_s1s2s3_color(
        s1, s2, s3,
        title=None,
        s3_mode='auto',
        mask_eps=1e-12,
        show=False,
        extent=None,
        figsize=(4, 4),
        dpi=100,
):
    """
    可视化三通道场 S[...,0]=s1, S[...,1]=s2, S[...,2]=s3，使用 HSV 颜色映射：
      - Hue:   来自 angle = atan2(S2, S1)，映射到 [0,1]
      - Sat:   默认 1；在 sqrt(S1^2+S2^2) < mask_eps 处置 0（避免相位不定产生杂色）
      - Value: 由 S3 映射黑到白

    返回
    ----
    rgb : np.ndarray
        形状与 S[..., :3] 的空间维一致的 RGB 图，范围 [0,1]。
    """
    S = np.stack([s1, s2, s3], axis=-1)
    S = np.asarray(S)
    if S.shape[-1] != 3:
        raise ValueError("输入 S 的最后一维必须是 3，对应 (S1, S2, S3)。")

    S1, S2, S3 = S[..., 0], S[..., 1], S[..., 2]

    # 1) Hue：由角度映射到 [0,1]
    angle = np.arctan2(S2, S1)                     # [-pi, pi]
    hue = (angle + np.pi) / (2 * np.pi)            # [0,1]

    # 2) Saturation：默认 1，在相位不确定处降为 0
    mag12 = np.hypot(S1, S2)
    sat = np.ones_like(mag12)
    sat = np.where(mag12 < mask_eps, 0.0, sat)

    # 3) Value：由 S3 映射到 [0,1]，即黑→白
    if s3_mode == '01':
        val = np.clip(S3, 0.0, 1.0)
    elif s3_mode == '-11':
        val = (S3 + 1.0) / 2.0
        val = np.clip(val, 0.0, 1.0)
    elif s3_mode == 'minmax':
        s3_min, s3_max = np.min(S3), np.max(S3)
        if np.isclose(s3_max, s3_min):
            val = np.zeros_like(S3)  # 全常数时置黑
        else:
            val = (S3 - s3_min) / (s3_max - s3_min)
    elif s3_mode == 'auto':
        s3_min, s3_max = np.min(S3), np.max(S3)
        if s3_min >= 0.0 and s3_max <= 1.0:
            val = np.clip(S3, 0.0, 1.0)
        elif s3_min >= -1.0 and s3_max <= 1.0:
            val = (S3 + 1.0) / 2.0
            val = np.clip(val, 0.0, 1.0)
        else:
            if np.isclose(s3_max, s3_min):
                val = np.zeros_like(S3)
            else:
                val = (S3 - s3_min) / (s3_max - s3_min)
    else:
        raise ValueError("未知 s3_mode，需为 {'auto','01','-11','minmax'} 之一。")

    # 组合 HSV -> RGB
    hsv = np.stack([hue, 1-np.abs(1-2*val)**2, val], axis=-1)
    # hsv = np.stack([hue, 1-np.abs(1-2*val), val], axis=-1)
    # hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # 绘图
    if show:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(rgb, origin='lower', extent=extent)
        if title is not None:
            plt.title(title)
        plt.axis('off')
        plt.show()

    # return rgb
    return np.transpose(rgb, (1, 0, 2))


def make_angle_s3_test(width=1000, height=400):
    """
    生成矩形测试图案：
      - 宽度方向：角度 theta 从 -pi 线性到 +pi
      - 高度方向：S3 从 -1 线性到 +1
    """
    # 水平长边是角度
    theta = np.linspace(-np.pi, np.pi, num=width, endpoint=True)         # [-pi, pi]
    # 垂直方向是 S3
    s3 = np.linspace(-1.0, 1.0, num=height, endpoint=True)               # [-1, 1]
    # 网格
    T, S3 = np.meshgrid(theta, s3)                                       # 形状 (H, W)

    # S1, S2 由角度给出；这样 hue 只由角度控制
    S1 = np.cos(T)
    S2 = np.sin(T)

    # 画图（extent 标注物理坐标范围：横轴角度，纵轴 S3）
    rgb = map_s1s2s3_color(
        S1, S2, S3,
        title="Angle (long side) vs S3 (height)",
        s3_mode='-11',              # 明确告知 S3∈[-1,1]
        show=True,
        extent=(-np.pi/2, np.pi/2, -1.0, 1.0),
        figsize=(8, 4),
    )
    return rgb

def map_complex2rbg(field, title=None, show=False):
    amp = np.abs(field)
    phase = np.angle(field)
    amp_norm = amp / amp.max()
    hue = (phase + np.pi) / (2*np.pi)
    hsv = np.stack((hue, np.ones_like(hue), amp_norm), axis=-1)
    rgb = hsv_to_rgb(hsv)
    if show:
        plt.figure(figsize=(4,4))
        plt.imshow(rgb, origin='lower', extent=(-1,1,-1,1))
        plt.title(title)
        plt.axis('off')
        plt.show()
    return np.transpose(rgb, (1, 0, 2))

if __name__ == '__main__':

    # 运行测试
    _ = make_angle_s3_test()
