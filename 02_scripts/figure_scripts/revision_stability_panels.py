"""Figure 5, the five-model MLFF stability screen, in the revision figure style.

Same four panels and the same numbers as 02_scripts/mlff_stability/
figure7_mlff_stability.py, which drew the submitted version: the case colours now
match every other figure (dentin was orange there and red everywhere else), the
scatter is opaque, panel titles give way to the (a) to (d) tags the caption
already describes, and the threshold is named in the legend instead of printed
across the bars.

Writes Figure7_mlff_stability_rev.{pdf,png} into 04_manuscript/revision_figures/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
from paths import DATA, MANUSCRIPT, ROOT

ps.use()
OUT = os.path.join(MANUSCRIPT, "revision_figures")
S = os.path.join(ROOT, "06_mlff_stability")
ORDER = ["enamel", "dentin", "implant"]
MODELS = ["chgnet", "m3gnet", "mace", "mattersim", "omni"]
MLAB = {"chgnet": "CHGNet", "m3gnet": "M3GNet", "mace": "MACE",
        "mattersim": "MatterSim", "omni": "SevenNet"}
LOW, HIGH = -0.05, 2.5          # the window the submitted figure applied
THRESH = 0.05

df = pd.read_csv(os.path.join(S, "stability_mlff.csv"))
mods = []
for sc, fn in (("enamel", "designed_compositions.csv"),
               ("dentin", "designed_dentin_compositions.csv"),
               ("implant", "designed_implant_compositions.csv")):
    t = pd.read_csv(os.path.join(DATA, fn))[["composition", "pred_modulus_GPa"]]
    t["showcase"] = sc
    mods.append(t)
df = df.merge(pd.concat(mods, ignore_index=True), on=["composition", "showcase"], how="left")

fig = plt.figure(figsize=(21.0, 15.0), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# (a) consensus energy above hull per candidate, model-to-model range as whiskers
ax = fig.add_subplot(gs[0, 0])
d = df.sort_values("ehull_median").reset_index(drop=True)
x = np.arange(len(d))
mcols = [m for m in MODELS if m in d.columns]
rlo, rhi = [], []
for _, row in d.iterrows():
    vals = [row[m] for m in mcols if pd.notna(row[m]) and LOW <= row[m] <= HIGH]
    vals = vals or [row["ehull_median"]]
    rlo.append(min(vals)); rhi.append(max(vals))
rlo, rhi = np.array(rlo), np.array(rhi)
med = d["ehull_median"].values
ax.bar(x, med, width=0.78, color=[ps.CASE_COLOR[s] for s in d.showcase],
       edgecolor="white", linewidth=0.8)
ax.errorbar(x, med, yerr=[(med - rlo).clip(min=0), (rhi - med).clip(min=0)], fmt="none",
            ecolor="0.20", elinewidth=1.8, capsize=4, capthick=1.8)
ax.axhline(THRESH, color="0.20", linewidth=2.6, linestyle=":")
ax.set_xlim(-1.0, len(d))
ax.set_ylim(0, max(rhi.max(), med.max()) * 1.04)
ps.headroom(ax, 0.30)
ax.set_xlabel(ps.tc("candidate, sorted"))
ax.set_ylabel(ps.tc("energy above hull (eV/atom)"))
hands = [Patch(color=ps.CASE_COLOR[s], label=ps.tc(ps.CASE_LABEL[s])) for s in ORDER]
hands.append(Line2D([], [], color="0.20", linewidth=2.6, linestyle=":",
                    label="Reference Line, 0.05 eV/atom"))
ax.legend(handles=hands, loc="upper left", ncol=2, fontsize=ps.BASE - 6,
          columnspacing=1.2, labelspacing=0.35)
ps.panel_tag(ax, "a")

# (b) spread by force field, median as a red bar
ax = fig.add_subplot(gs[0, 1])
present = [m for m in MODELS if m in df.columns]
vdata = []
for m in present:
    v = df[m].dropna()
    vdata.append(v[(v >= LOW) & (v <= HIGH)].values)
parts = ax.violinplot(vdata, showmeans=False, showextrema=False, showmedians=False)
for b in parts["bodies"]:
    b.set_facecolor(ps.COLORS[0]); b.set_edgecolor(ps.COLORS[0])
    b.set_alpha(0.35); b.set_linewidth(1.6)
for j, v in enumerate(vdata):
    ax.hlines(np.median(v), j + 1 - 0.16, j + 1 + 0.16, color=ps.COLORS[1], linewidth=5.0)
ax.axhline(THRESH, color="0.20", linewidth=2.6, linestyle=":")
ax.set_xticks(range(1, len(present) + 1))
ax.set_xticklabels([MLAB[m] for m in present], fontsize=ps.BASE - 6)
ax.set_xlim(0.4, len(present) + 0.6)
ax.set_ylim(0, max(v.max() for v in vdata) * 1.06)
ax.set_ylabel(ps.tc("energy above hull (eV/atom)"))
ps.panel_tag(ax, "b")

# (c) consensus energy above hull against the predicted modulus
ax = fig.add_subplot(gs[1, 0])
for s in ORDER:
    sub = df[df.showcase == s]
    ax.plot(sub.pred_modulus_GPa, sub.ehull_median, "o", color=ps.CASE_COLOR[s],
            markersize=16, markeredgecolor="white", markeredgewidth=1.0,
            linestyle="none", label=ps.tc(ps.CASE_LABEL[s]))
ax.set_xlabel(ps.tc("predicted Young's modulus (GPa)"))
ax.set_ylabel(ps.tc("energy above hull (eV/atom)"))
ax.margins(x=0.06, y=0.08)
ps.legend_above(ax, ncol=3)
ps.panel_tag(ax, "c")

# (d) consensus energy above hull by case study, every candidate shown
ax = fig.add_subplot(gs[1, 1])
data = [df[df.showcase == s].ehull_median.values for s in ORDER]
parts = ax.violinplot(data, showmeans=False, showextrema=False)
for k, b in enumerate(parts["bodies"]):
    b.set_facecolor(ps.CASE_COLOR[ORDER[k]]); b.set_edgecolor(ps.CASE_COLOR[ORDER[k]])
    b.set_alpha(0.30)
for k, s in enumerate(ORDER):
    yv = df[df.showcase == s].ehull_median.values
    jit = np.random.RandomState(k).uniform(-0.08, 0.08, len(yv))
    ax.plot(np.full(len(yv), k + 1) + jit, yv, "o", color=ps.CASE_COLOR[s], markersize=13,
            markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=3)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([ps.tc(ps.CASE_LABEL[s]) for s in ORDER], fontsize=ps.BASE - 6)
ax.set_ylabel(ps.tc("energy above hull (eV/atom)"))
ps.panel_tag(ax, "d")

print("wrote", ps.save(fig, "Figure7_mlff_stability_rev", OUT), flush=True)
print("above threshold: %d of %d, mean median %.3f eV/atom"
      % ((df.ehull_median > THRESH).sum(), len(df), df.ehull_median.mean()), flush=True)
print("REVISION_STABILITY_PANELS_DONE", flush=True)
