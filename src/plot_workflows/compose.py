"""On-demand PNG grid composition ("组图").

Public API promoted from the saturation-and-composition protocol's reference
implementation (NearFieldVortex ``compose_png_grid``). Composes already
rendered PNG tiles into one image — used when the user explicitly asks for a
combined figure ("帮我组一下 / combo"), never by default.

Pure Pillow (no matplotlib), stdlib-plus-Pillow only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

__all__ = ["compose_png_grid", "find_default_font"]


def find_default_font() -> Optional[str]:
    """First available TrueType font path for label gutters (None -> PIL default)."""
    from PIL import ImageFont

    for path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]:
        try:
            ImageFont.truetype(path, 12)
            return path
        except Exception:
            continue
    return None


def compose_png_grid(
    tiles: Sequence[Sequence[Optional[object]]],
    *,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    out_path=None,
    font_path: Optional[str] = None,
    label_font_size: Optional[int] = None,
    header_font_size: Optional[int] = None,
    cell_size: Optional[Tuple[int, int]] = None,
    gap: Tuple[int, int] = (0, 0),
    pad: int = 6,
    bg: Tuple[int, int, int, int] = (255, 255, 255, 255),
):
    """Compose a grid of existing PNG images into one image (PIL; no matplotlib).

    tiles: 2D list (rows x cols) of image paths; ``None`` -> blank cell.
    row_labels / col_labels: optional short text drawn in a left / top gutter;
        fonts auto-scale to ~7.5% of the tile side (floored at 28 px) unless
        label_font_size / header_font_size are given explicitly.
    cell_size: optional (w, h); default = max tile size so all cells align.
        Tiles are centered inside their cell, never resized.
    gap: (horizontal, vertical) spacing between cells in pixels.
    out_path: if given, save the composed image there (format by extension).
    Returns the composed PIL.Image.
    """
    from PIL import Image, ImageDraw, ImageFont

    rows = len(tiles)
    cols = len(tiles[0]) if rows else 0
    imgs = [[Image.open(p).convert("RGBA") if p else None for p in row] for row in tiles]

    if cell_size is None:
        sizes = [im.size for row in imgs for im in row if im]
        cell_size = (max(s[0] for s in sizes), max(s[1] for s in sizes)) if sizes else (400, 400)
    cw, ch = cell_size
    h_gap, v_gap = gap

    # Auto-scaled label fonts: ~7.5% of the tile side (floored at 28 px) so
    # labels stay legible on wide montages viewed at document scale.
    if font_path is None and (row_labels or col_labels):
        font_path = find_default_font()
    if label_font_size is None or header_font_size is None:
        auto_size = max(28, int(round(0.075 * min(cw, ch))))
        if label_font_size is None:
            label_font_size = auto_size
        if header_font_size is None:
            header_font_size = auto_size

    gutter_w = header_h = 0
    label_font = header_font = None
    if font_path and (row_labels or col_labels):
        label_font = ImageFont.truetype(font_path, label_font_size)
        header_font = ImageFont.truetype(font_path, header_font_size)
        temp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        if row_labels:
            gutter_w = int(max(temp.textbbox((0, 0), t, font=label_font)[2] for t in row_labels)) + pad * 2
        if col_labels:
            header_h = int(max(temp.textbbox((0, 0), t, font=header_font)[3] for t in col_labels)) + pad * 2

    total_w = gutter_w + cols * cw + h_gap * max(cols - 1, 0)
    total_h = header_h + rows * ch + v_gap * max(rows - 1, 0)
    canvas = Image.new("RGBA", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    for ri, row in enumerate(imgs):
        for ci, im in enumerate(row):
            if im is None:
                continue
            x = gutter_w + ci * (cw + h_gap) + (cw - im.width) // 2
            y = header_h + ri * (ch + v_gap) + (ch - im.height) // 2
            canvas.alpha_composite(im, (int(x), int(y)))

    if col_labels and header_font:
        for ci, t in enumerate(col_labels):
            cx = gutter_w + ci * (cw + h_gap) + cw // 2
            bb = draw.textbbox((0, 0), t, font=header_font)
            draw.text((cx - (bb[2] - bb[0]) // 2, header_h // 2 - (bb[3] - bb[1]) // 2),
                      t, font=header_font, fill="black")
    if row_labels and label_font:
        for ri, t in enumerate(row_labels):
            ry = header_h + ri * (ch + v_gap) + ch // 2
            bb = draw.textbbox((0, 0), t, font=label_font)
            draw.text((gutter_w // 2 - (bb[2] - bb[0]) // 2, ry - (bb[3] - bb[1]) // 2),
                      t, font=label_font, fill="black")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
    return canvas
