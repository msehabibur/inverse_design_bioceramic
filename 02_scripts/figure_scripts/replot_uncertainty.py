"""Restyle Figure8_uncertainty.png from the saved CSV (house style, no recompute)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA, MANUSCRIPT

plt.rcParams.update({
    "font.family": ["Arial Narrow", "Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 30, "font.weight": "normal",
    "axes.titlesize": 30, "axes.titleweight": "normal",
    "axes.labelsize": 30, "axes.labelweight": "normal",
    "xtick.labelsize": 28, "ytick.labelsize": 28, "legend.fontsize": 28,
})
COL = {"enamel": "#1A6090", "dentin": "#D97F33", "implant": "#2E7D5A"}
TARGET = {"enamel": 85.0, "dentin": 20.0, "implant": 110.0}

df = pd.read_csv(os.path.join(DATA, "uncertainty_intervals.csv"))
fig, ax = plt.subplots(1, 3, figsize=(19, 6.8))
for j, sc in enumerate(["enamel", "dentin", "implant"]):
    s = df[df.showcase == sc].reset_index(drop=True)
    x = np.arange(len(s)); yv = s["pred_E_GPa"].values
    ax[j].errorbar(x, yv, yerr=[yv - s["E_lo_GPa"].values, s["E_hi_GPa"].values - yv],
                   fmt="o", color=COL[sc], capsize=4, lw=1.4, ms=6)
    ax[j].axhline(TARGET[sc], ls=":", color="#D97F33", lw=2,
                  label=f"Target {TARGET[sc]:.0f} GPa")
    ax[j].set_title(f"{sc.capitalize()}", loc="left")
    ax[j].set_xlabel("Candidate (Ranked)")
    ax[j].set_ylabel("Predicted Young's Modulus (GPa)")
    ax[j].legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(MANUSCRIPT, "Figure8_uncertainty.png"), dpi=200, bbox_inches="tight")
print("saved restyled Figure8_uncertainty.png")
