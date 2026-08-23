from __future__ import annotations

from PIL import Image


def _tile(tmp_path, name, size, color):
    p = tmp_path / name
    Image.new("RGBA", size, color).save(p)
    return p


def test_compose_png_grid_labels_and_blank_cells(tmp_path):
    from plot_workflows import compose_png_grid

    tiles = [
        [_tile(tmp_path, "a.png", (40, 30), (255, 0, 0, 255)), None],
        [None, _tile(tmp_path, "d.png", (20, 50), (0, 0, 255, 255))],
    ]
    out = tmp_path / "combo.png"
    img = compose_png_grid(
        tiles,
        row_labels=["f325", "f350"],
        col_labels=["S0", "I_LP45"],
        out_path=out,
    )
    assert out.exists()
    # cell size = max tile (40, 50); labels add left gutter + top header
    assert img.size[0] > 2 * 40
    assert img.size[1] > 2 * 50
    # blank cells stay background: sample the top-right cell interior
    px = img.convert("RGBA").getpixel((img.size[0] - 10, 60))
    assert px[:3] == (255, 255, 255)


def test_compose_default_font_fallback(tmp_path):
    from plot_workflows import compose_png_grid, find_default_font

    # No font_path given: must fall back to a system font (or PIL default) without raising.
    tiles = [[_tile(tmp_path, "t.png", (32, 32), (0, 128, 0, 255))]]
    img = compose_png_grid(tiles, row_labels=["r0"], col_labels=["c0"])
    assert img.size[0] > 32
    assert find_default_font() is None or isinstance(find_default_font(), str)
