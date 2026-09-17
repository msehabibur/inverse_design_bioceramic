"""Redraw the three author-supplied schematics with the revised claims.

The originals are raster images with no source file, and three statements inside
them survived the revision: the graphical abstract says one surrogate is trained
once, the design-target figure calls the dentin target unreachable by any
single-phase oxide, and the workflow figure advertises about forty experiments,
an optimal search and a years-to-months saving that this work never measured.
Captions cannot fix text that is inside a picture, so the pictures are redrawn
here, plainly, in the style of the rest of the figures: no icons, no colour
coding beyond the case colours, sentence case throughout.

    ./05_config_envs/bioceramics_venv/bin/python \
        02_scripts/figure_scripts/revision_schematics.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "04_manuscript", "revision_figures")

import pandas as pd

_D = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "01_data")
_best = {c: float(pd.read_csv(os.path.join(_D, f)).pred_modulus_GPa.iloc[0])
         for c, f in (("enamel", "designed_compositions.csv"),
                      ("dentin", "designed_dentin_compositions.csv"),
                      ("implant", "designed_implant_compositions.csv"))}
_en = pd.read_csv(os.path.join(_D, "revision_enumeration_full.csv"))
_floor = float(_en[_en.pool == "dentin"].pred_modulus_GPa.min())

CASE = {"enamel": "#1f77b4", "dentin": "#d62728", "implant": "#2ca02c"}
INK = "#1a1a1a"
RULE = "#b8b8b8"


def card(fig, x, y, w, h, edge="#d0d0d0"):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure, facecolor="white", edgecolor=edge,
        linewidth=2.0, zorder=1))


def bar(fig, x, y, w, colour):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, 0.008, boxstyle="round,pad=0.0,rounding_size=0.004",
        transform=fig.transFigure, facecolor=colour, edgecolor="none", zorder=2))


def line(fig, x, y, w):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, 0.0012, boxstyle="square,pad=0.0", transform=fig.transFigure,
        facecolor=RULE, edgecolor="none", zorder=2))


def txt(fig, x, y, s, size, colour=INK, weight="normal", ha="center"):
    return fig.text(x, y, s, ha=ha, va="top", fontsize=size, color=colour,
                    fontweight=weight, zorder=3, linespacing=1.35)


class Flow:
    """Set text downwards from a starting height, each line placed below the
    measured bottom of the one before it.

    The earlier layouts put every line at a typed figure coordinate, so a name
    that rendered taller than planned printed over its subtitle and a card sized
    by hand left a third of itself empty. Here the height of every rendered line
    advances the cursor, and a card is drawn afterwards around what it holds.
    """

    def __init__(self, fig, y):
        self.fig, self.y = fig, y
        self.r = fig.canvas.get_renderer()

    def text(self, x, s, size, gap=0.0, **kw):
        self.y -= gap
        t = txt(self.fig, x, self.y, s, size, **kw)
        self.y -= t.get_window_extent(self.r).height / self.fig.bbox.height
        return t

    def skip(self, gap):
        self.y -= gap


def _save(fig, stem):
    ps.check(fig, stem)
    for e in (() if os.environ.get("PS_DRYRUN") else ("pdf", "png")):
        fig.savefig(os.path.join(OUT, stem + "." + e), dpi=300,
                    bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(OUT, stem + ".pdf"), flush=True)


def _cards(fig, xs, w, top, bottoms, colours, pad=0.030):
    """One card per column, all ending at the lowest content, with the colour bar
    on top."""
    low = min(bottoms) - pad
    for x, colour in zip(xs, colours):
        card(fig, x, low, w, top - low)
        bar(fig, x, top, w, colour)
    return low


# ------------------------------------------------------------------ graphical abstract
def graphical_abstract():
    fig = plt.figure(figsize=(13.0, 8.2))
    head = Flow(fig, 0.985)
    head.text(0.5, "Inverse design of bioceramics by machine learning", 31, weight="bold")
    head.text(0.5, "Gradient-boosting models and a genetic algorithm propose oxide "
                   "compositions\nat the stiffness and density of enamel, dentin and "
                   "a Ti-6Al-4V reference", 19, gap=0.012)
    top = head.y - 0.045
    col = [("enamel", "Enamel", "outer shell", "85", "Sr–Ca–Mg–Si–P\nphospho-silicate",
            "SrCaSiP$_{2}$O$_{9}$ · 95 to 105 GPa", "bioactive-glass chemistry",
            "Predicted within\nthe model error"),
           ("dentin", "Dentin", "compliant core", "20", "K–Na–Ca–P\nalkali phosphate",
            "K$_{5}$Na$_{2}$Ca(PO$_{4}$)$_{3}$ · 44 to 51 GPa", "bone-mineral chemistry only",
            "Lowest prediction in the\nreachable set is %.1f GPa,\na model-dependent screen"
            % _floor),
           ("implant", "Implant interface", "Ti-6Al-4V set-point", "110",
            "Sr–Ca–La/Ce\nphospho-oxide", "SrCaLaPO$_{6}$ · 102 to 118 GPa",
            "retargeting demonstration", "Needs a bone-matched\nmodulus to be useful")]
    x0, w, gap = 0.045, 0.293, 0.026
    xs, bottoms, rules = [], [], []
    for i, (key, name, sub, tgt, fam, ex, chem, verdict) in enumerate(col):
        x = x0 + i * (w + gap)
        c = x + w / 2
        f = Flow(fig, top - 0.030)
        f.text(c, name, 24, weight="bold")
        f.text(c, sub, 18, colour="#555555", gap=0.006)
        f.text(c, "target Young's modulus", 18, weight="bold", gap=0.020)
        f.text(c, "%s GPa" % tgt, 34, colour=CASE[key], weight="bold", gap=0.006)
        f.skip(0.026); rules.append((x, f.y)); f.skip(0.026)
        f.text(c, "genetic-algorithm family", 18, weight="bold")
        f.text(c, fam, 21, weight="bold", gap=0.008)
        f.text(c, ex, 19, gap=0.014)
        f.text(c, chem, 18, colour="#555555", gap=0.006)
        f.skip(0.026); rules.append((x, f.y)); f.skip(0.026)
        f.text(c, verdict, 19, colour=CASE[key], weight="bold")
        xs.append(x); bottoms.append(f.y)
    low = _cards(fig, xs, w, top, bottoms, [CASE[k[0]] for k in col])
    for x, y in rules:
        line(fig, x + 0.03, y, w - 0.06)
    Flow(fig, low - 0.030).text(
        0.5, "Outputs are hypothetical oxide compositions requiring biological testing",
        18, colour="#555555")
    _save(fig, "graphical_abstract_rev")


# ------------------------------------------------------------------ design targets
def design_targets():
    fig = plt.figure(figsize=(15.0, 7.4))
    head = Flow(fig, 0.985)
    head.text(0.5, "Three inverse-design targets across the dental restoration", 27,
              weight="bold")
    top = head.y - 0.040

    col = [("enamel", "Showcase 1", "Enamel", "Target fingerprint",
            ("85", "4", "3.0"), "Sr–Ca–Mg–Si–P phospho-silicate",
            "e.g. SrCaSiP$_{2}$O$_{9}$ · bioactive-glass chemistry",
            "Best prediction %.1f GPa,\nwithin the model error" % _best["enamel"]),
           ("dentin", "Showcase 2", "Dentin", "Target fingerprint",
            ("20", "0.7", "2.1"), "K–Na–(Ca,Mg)–P alkali phosphate",
            "e.g. K$_{5}$Na$_{2}$Ca(PO$_{4}$)$_{3}$ · bone-mineral chemistry",
            "Lowest prediction in the reachable\nset is %.1f GPa: a model-dependent\n"
            "screen, not a physical limit" % _floor),
           ("implant", "Showcase 3", "Implant interface", "Target fingerprint (Ti-6Al-4V)",
            ("110", "3.5", "4.4"), "Sr–Ca–La/Ce phospho-oxide",
            "e.g. SrCaLaPO$_{6}$ · phospho-oxide chemistry",
            "Best prediction %.1f GPa,\na retargeting demonstration" % _best["implant"])]

    x0, w, gap = 0.035, 0.303, 0.023
    xs, bottoms, rules, boxes = [], [], [], []
    r = fig.canvas.get_renderer()
    for i, (key, tag, name, fp, vals, fam, ex, note) in enumerate(col):
        x = x0 + i * (w + gap)
        c = x + w / 2
        f = Flow(fig, top - 0.026)
        f.text(c, tag, 18, colour=CASE[key], weight="bold")
        f.text(c, name, 25, weight="bold", gap=0.006)
        f.text(c, fp, 18, colour="#555555", gap=0.010)
        # the three property boxes: symbol over value, boxed to their own height
        box_top = f.y - 0.018
        lows = []
        for j, (sym, v, unit) in enumerate(zip(("$E$", "$H$", r"$\rho$"), vals,
                                               ("GPa", "GPa", "g/cm$^{3}$"))):
            bx = x + 0.022 + j * (w - 0.044) / 3.0
            bw = (w - 0.044) / 3.0 - 0.012
            g = Flow(fig, box_top - 0.012)
            g.text(bx + bw / 2, sym, 19, colour="#555555")
            g.text(bx + bw / 2, "%s %s" % (v, unit), 19, colour=CASE[key],
                   weight="bold", gap=0.002)
            lows.append((bx, bw, g.y))
        box_low = min(l[2] for l in lows) - 0.014
        for bx, bw, _ in lows:
            boxes.append((bx, box_low, bw, box_top - box_low))
        f.y = box_low
        f.skip(0.030); rules.append((x, f.y)); f.skip(0.030)
        f.text(c, "Genetic-algorithm family", 18, colour="#555555")
        f.text(c, fam, 20, weight="bold", gap=0.008)
        f.text(c, ex, 18, gap=0.014)
        f.skip(0.030); rules.append((x, f.y)); f.skip(0.030)
        f.text(c, note, 18, colour=CASE[key])
        xs.append(x); bottoms.append(f.y)
    _cards(fig, xs, w, top, bottoms, [CASE[k[0]] for k in col])
    # drawn after the card that holds them, or the card paints over them
    for b in boxes:
        card(fig, *b)
    for x, y in rules:
        line(fig, x + 0.025, y, w - 0.05)
    _save(fig, "Figure1_design_targets_rev")


# ------------------------------------------------------------------ workflow
def workflow():
    fig = plt.figure(figsize=(13.0, 9.2))
    head = Flow(fig, 0.985)
    head.text(0.5, "Conventional and data-driven materials design", 27, weight="bold")
    head.text(0.5,
              "The two development cycles as described in the literature the introduction "
              "cites; no comparison between them was measured in this work", 17,
              colour="#555555", gap=0.010)
    top = head.y - 0.040

    left = [("Chemical intuition", "expert-guided, narrow scope"),
            ("Bulk synthesis", "material cost per variant"),
            ("Full characterisation", "weeks per candidate"),
            ("Failure and iteration", "local optima, reactive"),
            ("Clinical trials", "in vitro to in vivo gap")]
    right = [("Data curation", "database records, high-dimensional"),
             ("Surrogate model training", "nonlinear, uncertainty quantified"),
             ("In silico screening", "no material consumed"),
             ("Genetic-algorithm inverse design", "charge-balanced integer compositions"),
             ("Computational shortlist", "requires experimental testing")]

    xs, bottoms, rules, colours = [], [], [], []
    w = 0.425
    for k, (title, items, colour) in enumerate(
            (("Conventional trial-and-error cycle", left, "#8c6d1f"),
             ("Data-driven design, this work", right, "#1f6f8c"))):
        x = 0.045 + k * 0.485
        f = Flow(fig, top - 0.026)
        f.text(x + w / 2, title, 22, colour=colour, weight="bold")
        f.skip(0.030)
        for n, (name, sub) in enumerate(items):
            if n:
                f.skip(0.024); rules.append((x, f.y)); f.skip(0.024)
            f.text(x + 0.025, name, 20, weight="bold", ha="left")
            f.text(x + 0.025, sub, 17, colour="#555555", ha="left", gap=0.006)
        xs.append(x); bottoms.append(f.y); colours.append(colour)
    low = _cards(fig, xs, w, top, bottoms, colours)
    for x, y in rules:
        line(fig, x + 0.025, y, w - 0.05)
    Flow(fig, low - 0.030).text(
        0.5, "The right-hand column is the pipeline this work implements. The saving it is "
             "intended to produce is in experiment count and\ncalendar time, and it is not "
             "measured here: no specimen was synthesised and no candidate was tested.", 17,
        colour="#333333")
    _save(fig, "FigureS1_workflow_rev")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": ps.FAMILY, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    graphical_abstract()
    design_targets()
    workflow()
    print("REVISION_SCHEMATICS_DONE", flush=True)
