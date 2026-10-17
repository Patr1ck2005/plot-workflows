"""Small local documentation helper; not part of the installed package API."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata as metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_foundation import SaveSpec, save_figure

COLORS = ("#1257df", "#e31a35", "#009b83", "#8b40bc")
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 9, "axes.labelsize": 8, "axes.titlesize": 9,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
         "lines.linewidth": .75, "axes.linewidth": .5,
         "xtick.direction": "in", "ytick.direction": "in", "svg.hashsalt": "synthetic-showcase-v1"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tidy3d(ax):
    from matplotlib.ticker import MaxNLocator
    ax.tick_params(pad=0, labelsize=7)
    ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 1
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.set_major_locator(MaxNLocator(4))


class Showcase:
    def __init__(self, out, title, source, conventions, decisions):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.title, self.source = title, Path(source)
        self.conventions, self.decisions = conventions, decisions
        self.items = []

    def save(self, fig, stem, title, caption):
        fig.canvas.draw()
        size = [round(float(v) * 25.4, 4) for v in fig.get_size_inches()]
        for ext in ("png", "svg"):
            save_figure(fig, self.out / f"{stem}.{ext}",
                        SaveSpec(dpi=180, bbox_inches=None, transparent=False))
        self.items.append({"stem": stem, "title": title, "caption": caption,
                           "size_mm": size, "dpi": 180,
                           "png_sha256": sha(self.out / f"{stem}.png"),
                           "svg_sha256": sha(self.out / f"{stem}.svg")})
        plt.close(fig)

    def finish(self, checks, seed=1729):
        packages = ("numpy", "scipy", "matplotlib", "pandas", "eigenmode-analysis",
                    "plot-foundation", "plot-workflows", "campaign-data")
        versions = {}
        artifacts = {}
        for name in packages:
            try:
                versions[name] = metadata.version(name)
                direct = metadata.distribution(name).read_text("direct_url.json")
                if direct:
                    details = json.loads(direct)
                    archive_hash = details.get("archive_info", {}).get("hashes", {}).get("sha256")
                    commit = details.get("vcs_info", {}).get("commit_id")
                    if archive_hash or commit:
                        artifacts[name] = {"archive_sha256": archive_hash, "vcs_commit_id": commit}
            except metadata.PackageNotFoundError:
                pass
        manifest = {"evidence_type": "computed synthetic software demonstration",
                    "data_origin": "independent generic mathematical definitions in source; no research data",
                    "seed": seed, "source": f"../../../examples/{self.source.name}",
                    "source_sha256": sha(self.source), "helper_sha256": sha(__file__),
                    "generation_command": f"python examples/{self.source.name} --out docs/assets/showcase",
                    "versions": versions, "installed_artifacts": artifacts, "conventions": self.conventions,
                    "data_space_decisions": self.decisions, "checks": checks,
                    "style": "explicit documentation geometry; 180 DPI preview; fixed canvas; Arial 8/9 pt",
                    "images": self.items}
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        cards = []
        for item in self.items:
            stem = item["stem"]
            cards.append(f'<section><h2>{html.escape(item["title"])}</h2><img src="{stem}.png" alt="{html.escape(item["caption"])}"><p>{html.escape(item["caption"])}</p><a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a></section>')
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        page += f'<title>{html.escape(self.title)}</title><style>body{{max-width:1100px;margin:36px auto;padding:0 22px;font:16px/1.6 system-ui;color:#172238;background:#f6f8fc}}section{{background:white;padding:20px;margin:24px 0;border:1px solid #dbe2ed;border-radius:12px}}img{{width:100%;height:auto}}h1,h2{{line-height:1.2}}a{{color:#1257df}}</style>'
        page += f'<h1>{html.escape(self.title)}</h1><p>Computed synthetic examples · software demonstration · no experimental or full-wave evidence.</p><h2>Reading conventions</h2><p>{html.escape(self.conventions)}</p><p><a href="manifest.json">Provenance and checks</a> · <a href="../../../examples/{self.source.name}">Runnable source</a> · <a href="../../../README.md">README</a></p>'
        page += ''.join(cards) + '</html>'
        (self.out / "gallery.html").write_text(page, encoding="utf-8")
        print(json.dumps({"output": str(self.out), "images": len(self.items), "checks": checks}))
