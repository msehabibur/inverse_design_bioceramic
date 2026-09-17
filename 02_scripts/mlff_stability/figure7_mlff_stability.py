#!/usr/bin/env python3
"""Figure 7: six-model MLFF convex-hull stability of all 45 GA candidates."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": ["Arial Narrow", "Arial", "DejaVu Sans"],
    "font.size": 30, "font.weight": "normal",
    "axes.titlesize": 30, "axes.labelsize": 30,
    "xtick.labelsize": 26, "ytick.labelsize": 26, "legend.fontsize": 24,
})
# the scratch home was purged; the project tree is the only copy
BASE = "/anvil/projects/x-mat260059/Inverse_Design_of_Bioceramics_by_Machine_Learning/"
S = BASE + "06_mlff_stability/"
DATA = BASE + "01_data/"
MAN = BASE + "04_manuscript/"
COL = {"enamel": "#1A6090", "dentin": "#D97F33", "implant": "#2E7D5A"}
LAB = {"enamel": "Enamel", "dentin": "Dentin", "implant": "Implant"}
MODELS = ["chgnet", "m3gnet", "mace", "mattersim", "omni"]
MLAB = {"chgnet": "CHGNet", "m3gnet": "M3GNet", "mace": "MACE",
        "mattersim": "MatterSim", "grace": "GRACE", "omni": "SevenNet"}

df = pd.read_csv(S + "stability_mlff.csv")
lng = pd.read_csv(S + "stability_mlff_long.csv")
# merge predicted modulus by composition
mods = []
for sc, fn in [("enamel", "designed_compositions.csv"), ("dentin", "designed_dentin_compositions.csv"),
               ("implant", "designed_implant_compositions.csv")]:
    t = pd.read_csv(DATA + fn)[["composition", "pred_modulus_GPa"]]; t["showcase"] = sc; mods.append(t)
mods = pd.concat(mods, ignore_index=True)
df = df.merge(mods, on=["composition", "showcase"], how="left")

fig, ax = plt.subplots(2, 2, figsize=(21, 15))

# (a) per-candidate consensus ehull with min-max whiskers, sorted
d = df.sort_values("ehull_median").reset_index(drop=True)
x = np.arange(len(d))
colors = [COL[s] for s in d["showcase"]]
# robust per-candidate spread: drop single-model below-hull / extreme outliers
LOW, HIGH = -0.05, 2.5
mcols = [m for m in MODELS if m in d.columns]
rlo, rhi = [], []
for _, row in d.iterrows():
    vals = [row[m] for m in mcols if pd.notna(row[m]) and LOW <= row[m] <= HIGH]
    if not vals:
        vals = [row["ehull_median"]]
    rlo.append(min(vals)); rhi.append(max(vals))
rlo, rhi = np.array(rlo), np.array(rhi)
lo = (d["ehull_median"].values - rlo).clip(min=0)
hi = (rhi - d["ehull_median"].values).clip(min=0)
ax[0, 0].bar(x, d["ehull_median"], color=colors, alpha=0.85)
ax[0, 0].errorbar(x, d["ehull_median"], yerr=[lo, hi], fmt="none", ecolor="#333333", lw=1.3, capsize=3)
ax[0, 0].axhline(0.05, ls=":", color="#C0392B", lw=2.5)
ax[0, 0].text(1, 0.10, "Near-stable threshold (0.05)", color="#C0392B", fontsize=22, va="bottom")
ax[0, 0].set_ylim(0, max(rhi.max(), d["ehull_median"].max()) * 1.08)
ax[0, 0].set_xlabel("Candidate (sorted)")
ax[0, 0].set_ylabel("Energy above hull (eV/atom)")
ax[0, 0].set_title("(a)  Five-model consensus stability", loc="left")
hands = [plt.Rectangle((0, 0), 1, 1, color=COL[s]) for s in COL]
ax[0, 0].legend(hands, [LAB[s] for s in COL], frameon=False, loc="upper left")

# (b) per-model spread -- violin (outliers removed)
present = [m for m in MODELS if m in df.columns]
vdata = []
for m in present:
    v = df[m].dropna()
    v = v[(v >= LOW) & (v <= HIGH)]
    vdata.append(v.values)
parts = ax[0, 1].violinplot(vdata, showmeans=False, showextrema=False, showmedians=False)
for b in parts["bodies"]:
    b.set_facecolor("#1A6090"); b.set_alpha(0.45); b.set_edgecolor("#1A6090"); b.set_linewidth(1.2)
for j, v in enumerate(vdata):
    ax[0, 1].scatter([j + 1], [np.median(v)], s=300, color="#C0392B", marker="_", lw=4, zorder=4)
ax[0, 1].axhline(0.05, ls=":", color="#C0392B", lw=2)
ax[0, 1].set_xticks(range(1, len(present) + 1))
ax[0, 1].set_xticklabels([MLAB[m] for m in present], rotation=30, ha="right")
ax[0, 1].set_ylim(0, max(v.max() for v in vdata) * 1.08)
ax[0, 1].set_ylabel("Energy above hull (eV/atom)")
ax[0, 1].set_title("(b)  Agreement across force fields", loc="left")

# (c) ehull vs predicted modulus
for s in COL:
    sub = df[df.showcase == s]
    ax[1, 0].scatter(sub["pred_modulus_GPa"], sub["ehull_median"], s=140, alpha=0.8,
                     color=COL[s], edgecolor="white", lw=1.0, label=LAB[s])
ax[1, 0].set_xlabel("Predicted Young's modulus (GPa)")
ax[1, 0].set_ylabel("Energy above hull (eV/atom)")
ax[1, 0].set_title("(c)  Stability versus target match", loc="left")
ax[1, 0].legend(frameon=False, loc="best")
ax[1, 0].grid(alpha=0.25)

# (d) distribution by case study
order = ["enamel", "dentin", "implant"]
data = [df[df.showcase == s]["ehull_median"].values for s in order]
parts = ax[1, 1].violinplot(data, showmeans=False, showextrema=False)
for k, b in enumerate(parts["bodies"]):
    b.set_facecolor(COL[order[k]]); b.set_alpha(0.5)
for k, s in enumerate(order):
    yv = df[df.showcase == s]["ehull_median"].values
    ax[1, 1].scatter(np.full(len(yv), k + 1) + np.random.RandomState(k).uniform(-0.08, 0.08, len(yv)),
                     yv, s=80, color=COL[s], edgecolor="white", lw=0.7, zorder=3)
ax[1, 1].set_xticks([1, 2, 3]); ax[1, 1].set_xticklabels([LAB[s] for s in order])
ax[1, 1].set_ylabel("Energy above hull (eV/atom)")
ax[1, 1].set_title("(d)  Distribution by case study", loc="left")

fig.tight_layout(pad=1.6, h_pad=2.6, w_pad=2.2)
out = MAN + "Figure7_mlff_stability.png"
fig.savefig(out, dpi=200)
print("saved", out)
print(f"all above hull: {(df.ehull_median>0.05).sum()}/{len(df)}  mean median={df.ehull_median.mean():.3f}")
