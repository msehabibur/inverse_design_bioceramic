"""Shared figure style for the bioceramics manuscript revision.

Arial Narrow, very large type, compact and symmetric panel gaps, no all-caps text,
distinguishable line styles and markers, 300 dpi, PDF + PNG output.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FAMILY = "Arial Narrow"
_have = {f.name for f in font_manager.fontManager.ttflist}
if FAMILY not in _have:
    FAMILY = "Nimbus Sans Narrow" if "Nimbus Sans Narrow" in _have else "DejaVu Sans"

BASE = 32
# A wide canvas scaled into a journal column renders type at roughly a quarter of
# its authored size: 36 pt on a 26 inch canvas prints at about 9.4 pt in a 6.9 inch
# text block. Nothing should be authored below BASE - 8, which prints near 6.4 pt.
MIN_OFFSET = 8

RC = {
    "font.family": FAMILY,
    "font.size": BASE,
    "axes.titlesize": BASE + 2,
    "axes.labelsize": BASE + 4,
    "xtick.labelsize": BASE,
    "ytick.labelsize": BASE,
    "legend.fontsize": BASE - 2,
    "figure.titlesize": BASE + 4,
    "axes.linewidth": 1.6,
    "axes.labelpad": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 10.0,
    "ytick.major.size": 10.0,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.minor.size": 4.0,
    "ytick.minor.size": 4.0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "lines.linewidth": 3.6,
    "lines.markersize": 16,
    "legend.frameon": False,
    "legend.handlelength": 2.6,
    "savefig.dpi": 300,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.30,
    "errorbar.capsize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Tab palette, colour-blind safe ordering; every series also gets its own
# line style and marker so the panels stay readable in greyscale print.
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
          "#8c564b", "#17becf", "#7f7f7f"]
DASHES = [(None, None), (6, 2), (2, 2), (8, 2, 2, 2), (4, 1, 1, 1),
          (1, 1), (10, 3), (5, 2, 1, 2)]
# Every series is a circle, by house rule. Series stay separable in greyscale
# through the tab colour, the line style and whether the circle is filled.
MARKERS = ["o"] * 8

CASE_COLOR = {"enamel": "#1f77b4", "dentin": "#d62728", "implant": "#2ca02c"}
CASE_MARKER = {"enamel": "o", "dentin": "o", "implant": "o"}
CASE_DASH = {"enamel": (None, None), "dentin": (6, 2), "implant": (2, 2)}
CASE_LABEL = {"enamel": "enamel", "dentin": "dentin",
              "implant": "implant interface"}


# Axis labels and legends are set in title case: every word takes a capital first
# letter. Units, chemical formulas and mathtext are left untouched.
_KEEP = {"eV", "GPa", "cm", "atom", "and", "of", "per", "in", "the", "as", "a",
         "vs", "to", "by", "for"}


def tc(text):
    """Title case for a figure label, leaving units and formulas alone."""
    out = []
    for tok in text.split(" "):
        core = tok.strip("()[],")
        if not core:
            out.append(tok)
            continue
        if core in _KEEP or any(c.isupper() for c in core) or "$" in tok or any(
                c.isdigit() for c in core):
            out.append(tok)
            continue
        i = tok.index(core[0])
        out.append(tok[:i] + core[0].upper() + tok[i + 1:])
    return " ".join(out)


def use():
    plt.rcParams.update(RC)


def line(ax, x, y, i=0, label=None, **kw):
    """Plot series i with its own colour, dash pattern and marker."""
    kw.setdefault("color", COLORS[i % len(COLORS)])
    kw.setdefault("marker", MARKERS[i % len(MARKERS)])
    kw.setdefault("markevery", max(1, len(x) // 8) if len(x) > 12 else 1)
    d = DASHES[i % len(DASHES)]
    (ln,) = ax.plot(x, y, label=label, **kw)
    if d[0] is not None:
        ln.set_dashes(list(d))
    return ln


def headroom(ax, frac):
    """Raise the upper y limit so the data fill only the lower (1 - frac) of the
    axes, leaving a band at the top in which a legend cannot touch it."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) / (1.0 - frac))


def legend_above(ax, ncol=1, **kw):
    """Legend in a band opened above the axes, clear of every data point."""
    kw.setdefault("fontsize", BASE - 6)
    kw.setdefault("columnspacing", 1.2)
    kw.setdefault("handletextpad", 0.6)
    kw.setdefault("labelspacing", 0.4)
    return ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
                     frameon=False, borderaxespad=0.0, **kw)


def panel_tag(ax, tag, dx=-0.16, dy=1.04):
    ax.text(dx, dy, "(%s)" % tag, transform=ax.transAxes,
            fontsize=RC["axes.labelsize"], va="bottom", ha="left")


def grid(fig, nrow, ncol, **kw):
    kw.setdefault("wspace", 0.28)
    kw.setdefault("hspace", 0.30)
    return fig.add_gridspec(nrow, ncol, **kw)


def _bb(artist, renderer, shrink=1.5):
    b = artist.get_window_extent(renderer)
    return b.from_extents(b.x0 + shrink, b.y0 + shrink, b.x1 - shrink, b.y1 - shrink)


def _hit(bbox, a, renderer):
    """True when data artist `a` draws anywhere inside bbox."""
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.path import Path
    if not a.get_visible():
        return False
    if isinstance(a, Line2D):
        xy = a.get_transform().transform(np.column_stack(a.get_data()).astype(float))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            return False
        if a.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
            if Path(xy).intersects_bbox(bbox, filled=False):
                return True
        if a.get_marker() not in (None, "None", "none", "", " "):
            r = a.get_markersize() * renderer.points_to_pixels(1.0) / 2.0
            inside = ((xy[:, 0] > bbox.x0 - r) & (xy[:, 0] < bbox.x1 + r)
                      & (xy[:, 1] > bbox.y0 - r) & (xy[:, 1] < bbox.y1 + r))
            if inside.any():
                return True
        return False
    if isinstance(a, PathCollection):
        off = a.get_offset_transform().transform(np.asarray(a.get_offsets(), float))
        off = off[np.isfinite(off).all(axis=1)]
        sz = a.get_sizes()
        r = (np.sqrt(sz.max()) if len(sz) else 6.0) * renderer.points_to_pixels(1.0) / 2.0
        return bool(((off[:, 0] > bbox.x0 - r) & (off[:, 0] < bbox.x1 + r)
                     & (off[:, 1] > bbox.y0 - r) & (off[:, 1] < bbox.y1 + r)).any())
    if isinstance(a, LineCollection):
        t = a.get_transform()
        return any(len(seg) > 1 and Path(t.transform(np.asarray(seg, float))).intersects_bbox(bbox, filled=False)
                   for seg in a.get_segments())
    if isinstance(a, PolyCollection):
        t = a.get_transform()
        return any(t.transform_path(p).intersects_bbox(bbox, filled=True) for p in a.get_paths())
    if isinstance(a, Patch):
        return a.get_transform().transform_path(a.get_path()).intersects_bbox(bbox, filled=True)
    return False


def find_overlaps(fig):
    """Every place where a piece of text touches other text or drawn data.

    Text is every axes text, legend, axis label, title, figure text and every tick
    label inside the view. Data is every line, marker, error bar, bar, band and
    scatter point, tested only where it is drawn (inside its axes). A text given
    gid "inside" is meant to sit on a bar: it is not tested against bars and bands,
    but a line or a marker through it still counts.
    """
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    texts = []
    for k, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        tag = "axes %d" % k
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                texts.append((ax, "%s text %r" % (tag, t.get_text()[:40]), t))
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            texts.append((ax, "%s legend" % tag, lg))
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if lab.get_visible() and lab.get_text().strip():
                texts.append((None, "%s label %r" % (tag, lab.get_text()[:40]), lab))
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            span = (hi - lo) or 1.0
            for tick in axis.get_major_ticks():
                if not (lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span):
                    continue
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        texts.append((None, "%s tick %r" % (tag, lab.get_text()[:20]), lab))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            texts.append((None, "figure text %r" % t.get_text()[:40], t))
    for lg in fig.legends:
        texts.append((None, "figure legend", lg))

    boxes = [(ax, name, art, _bb(art, renderer)) for ax, name, art in texts]
    found = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][3], boxes[j][3]
            if a.width > 0 and b.width > 0 and a.overlaps(b):
                found.append("%s  x  %s" % (boxes[i][1], boxes[j][1]))
    from matplotlib.patches import Patch
    for ax, name, art, box in boxes:
        inside = art.get_gid() == "inside"
        for dax in fig.axes:
            if not dax.get_visible():
                continue
            clip = box.from_extents(max(box.x0, dax.bbox.x0), max(box.y0, dax.bbox.y0),
                                    min(box.x1, dax.bbox.x1), min(box.y1, dax.bbox.y1))
            if clip.x1 <= clip.x0 or clip.y1 <= clip.y0:
                continue
            for a in list(dax.lines) + list(dax.collections) + list(dax.patches):
                if inside and isinstance(a, Patch):
                    continue
                if _hit(clip, a, renderer):
                    found.append("%s  x  data %s in axes %d" % (name, type(a).__name__, fig.axes.index(dax)))
                    break
        # shapes drawn straight onto the figure (cards, colour bars, rules): text
        # may sit inside a card but must not cross any outline
        for p in fig.patches:
            if p.get_visible() and p.get_transform().transform_path(p.get_path()).intersects_bbox(box, filled=False):
                found.append("%s  x  figure shape at y=%.3f" % (name, p.get_window_extent(renderer).y0 / fig.bbox.height))
                break
    return found


def check(fig, stem):
    """Refuse to write a figure in which any text touches text or data."""
    import os
    found = find_overlaps(fig)
    for f in found:
        print("OVERLAP [%s] %s" % (stem, f), flush=True)
    if found and os.environ.get("PS_OVERLAP", "fail") != "report":
        raise SystemExit("%s: %d overlapping element(s); fix the layout" % (stem, len(found)))
    return found


def save(fig, stem, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    check(fig, stem)
    if os.environ.get("PS_DRYRUN"):
        plt.close(fig)
        return os.path.join(outdir, stem + ".png")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "%s.%s" % (stem, ext)))
    plt.close(fig)
    return os.path.join(outdir, stem + ".png")
