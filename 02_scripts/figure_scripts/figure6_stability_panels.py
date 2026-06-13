"""
Figure 6 of the paper: MP convex-hull stability check across all three GA
showcases (enamel, dentin, implant interface).

The figure has 2 rows x 3 columns matching the visual rhythm of figures 3-5:

  Row 1: per-showcase bar plots of the convex-hull energy at each GA
         candidate's composition.  Within a chemsys, lower (more negative)
         means the GA composition needs a deeper formation energy to lie on
         the hull -- i.e. a 'harder' stability target.
    (a)  Showcase 1  --  enamel candidates
    (b)  Showcase 2  --  dentin candidates
    (c)  Showcase 3  --  implant-interface candidates

  Row 2: cross-showcase context.
    (d)  Chemsys coverage  --  number of MP entries available in each unique
         chemical system that the GA explored
    (e)  Hull energy vs predicted modulus  --  scatter of all 45 candidates,
         colored by showcase, to see whether 'stiffer target' correlates
         with 'deeper hull' chemistry
    (f)  Decomposition diversity  --  how many distinct phases each candidate
         is predicted to decompose into (proxy for chemical novelty)

Input: data/stability_check.csv  (produced by scripts/stability_check.py)
Output: Figure7_stability_panels.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":       ["Arial Narrow", "Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":         22,
    "font.weight":       "normal",
    "axes.titlesize":    22,
    "axes.titleweight":  "normal",
    "axes.labelsize":    22,
    "axes.labelweight":  "normal",
    "xtick.labelsize":   22,
    "ytick.labelsize":   16,
    "legend.fontsize":   22,
    "figure.titlesize":  22,
    "figure.titleweight":"normal",
})

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA, MANUSCRIPT

BLUE   = "#1A6090"
ORANGE = "#D97F33"
GREEN  = "#2E7D5A"
GREY   = "#777777"

SHOWCASE_COLOR = {
    "enamel":  BLUE,
    "dentin":  ORANGE,
    "implant": GREEN,
}
SHOWCASE_LABEL = {
    "enamel":  "Showcase 1 — Enamel",
    "dentin":  "Showcase 2 — Dentin",
    "implant": "Showcase 3 — Implant",
}

df = pd.read_csv(os.path.join(DATA, "stability_check.csv"))
print(f"Loaded {len(df)} stability records")

fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))


# (a) (b) (c) per-showcase hull-energy bars ---------------------------------
for col, sc in enumerate(["enamel", "dentin", "implant"]):
    ax = axes[0, col]
    sub = df[df["showcase"] == sc].sort_values("rank")
    color = SHOWCASE_COLOR[sc]
    labels = [str(r) for r in sub["rank"]]
    vals   = sub["hull_energy_eV"].values
    ax.bar(labels, vals, color=color, alpha=0.85)
    ax.set_xlabel("Candidate (ranked)")
    ax.set_ylabel("Hull energy at composition\n(eV/atom)")
    panel = "abc"[col]
    ax.set_title(f"({panel})  {SHOWCASE_LABEL[sc]}", loc="left")
    ax.tick_params(axis="x", rotation=0, labelsize=15)


# (d) chemsys coverage --------------------------------------------------------
ax = axes[1, 0]
chemsys_df = df.groupby("chemsys").agg(
    n_mp_entries=("n_mp_entries", "first"),
    showcase=("showcase", lambda s: ", ".join(sorted(set(s)))),
).reset_index().sort_values("n_mp_entries", ascending=True)
bar_colors = [SHOWCASE_COLOR[s.split(",")[0].strip()] for s in chemsys_df["showcase"]]
ax.barh(chemsys_df["chemsys"], chemsys_df["n_mp_entries"], color=bar_colors,
        alpha=0.85, height=0.78)
ax.set_xlabel("MP entries in chemsys")
ax.set_ylabel("")
ax.set_title("(d)  Chemistry coverage", loc="left")
ax.tick_params(axis="x", labelsize=15)
ax.margins(y=0.01)
ax.set_xlim(0, chemsys_df["n_mp_entries"].max() * 1.22)
# show only a few y-tick labels (every Nth) at a larger, readable font
_n = len(chemsys_df)
_step = max(1, _n // 6)
_pos = list(range(0, _n, _step))
ax.set_yticks(_pos)
ax.set_yticklabels([chemsys_df["chemsys"].iloc[i] for i in _pos], fontsize=15)


# (e) hull energy vs predicted modulus -- all candidates --------------------
ax = axes[1, 1]
for sc in ["enamel", "dentin", "implant"]:
    sub = df[df["showcase"] == sc]
    ax.scatter(sub["pred_modulus_GPa"], sub["hull_energy_eV"],
               s=80, alpha=0.7, color=SHOWCASE_COLOR[sc],
               edgecolor="white", lw=0.8,
               label=SHOWCASE_LABEL[sc])
ax.set_xlabel("Predicted modulus (GPa)")
ax.set_ylabel("Hull energy (eV/atom)")
ax.set_title("(e)  Hull energy vs predicted modulus", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=18)
ax.grid(alpha=0.25)


# (f) decomposition diversity (per-showcase summary) ------------------------
ax = axes[1, 2]
def n_phases(s):
    if isinstance(s, str) and "+" in s:
        return len([p for p in s.split("+") if p.strip()])
    if isinstance(s, str) and "(" in s:
        return 1
    return 0
df["n_decomp_phases"] = df["decomposition"].apply(n_phases)

labels  = ["Enamel", "Dentin", "Implant"]
means   = []
stds    = []
colors  = []
for sc in ["enamel", "dentin", "implant"]:
    sub = df[df["showcase"] == sc]["n_decomp_phases"]
    means.append(sub.mean())
    stds.append(sub.std())
    colors.append(SHOWCASE_COLOR[sc])
xpos = np.arange(len(labels))
ax.bar(xpos, means, yerr=stds, color=colors, alpha=0.85,
       capsize=8, ecolor="#444444")
# annotate each bar with its mean value
for x, m in zip(xpos, means):
    ax.text(x, m + 0.05, f"{m:.1f}", ha="center", va="bottom")
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("Distinct phases in decomp.\n(mean across 15 candidates)")
ax.set_title("(f)  Decomposition diversity", loc="left")

fig.tight_layout(pad=1.4, h_pad=2.4, w_pad=1.6)
out = os.path.join(MANUSCRIPT, "Figure7_stability_panels.png")
fig.savefig(out, dpi=220)
print(f"\nsaved -> {out}")
