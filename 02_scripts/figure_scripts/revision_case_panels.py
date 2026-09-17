"""Rebuild the three case-study figures in the revision style.

Panels, in the layout the manuscript already describes:
  (a) to (c)  surrogate parity for modulus, hardness and density, out-of-fold
  (d)         convergence, across-seed mean running best with a standard-deviation band
              and the five individual seed traces
  (e)         the pooled final population in the plane of predicted modulus and predicted
              density, with the target marked
  (f)         the fifteen best-scoring candidates with 90 percent conformal intervals

Writes Figure4_enamel_panels_rev, Figure5_dentin_panels_rev and
Figure6_implant_panels_rev into 04_manuscript/revision_figures/.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymatgen.core import Composition
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
from features import make_featurizer
from paths import DATA, MANUSCRIPT

ps.use()
OUT = os.path.join(MANUSCRIPT, "revision_figures")
AMU = 1.66054
SEED = 42

CASES = {
    "enamel":  dict(stem="Figure4_enamel_panels_rev", pool="mp_data.csv",
                    target=(85.0, 4.0, 3.0), label="Enamel"),
    "dentin":  dict(stem="Figure5_dentin_panels_rev", pool="dentin_data.csv",
                    target=(20.0, 0.7, 2.1), label="Dentin"),
    "implant": dict(stem="Figure6_implant_panels_rev", pool="mp_data.csv",
                    target=(110.0, 3.5, 4.4), label="Implant Interface"),
}

hist = pd.read_csv(os.path.join(DATA, "revision_ga_history.csv"))
unc = pd.read_csv(os.path.join(DATA, "revision_uncertainty.csv"))
enum = pd.read_csv(os.path.join(DATA, "revision_enumeration_full.csv"))
# the floor is quoted from the summary table the manuscript quotes, so the panel and
# the text round the same number: the per-formula table stores 57.15, which %.1f
# prints as 57.1 while the text, from the unrounded value, says 57.2
FLOOR = pd.read_csv(os.path.join(DATA, "revision_enumeration.csv")).set_index("pool").modulus_floor_GPa
POOL_OF = {"enamel": "enamel", "implant": "enamel", "dentin": "dentin"}

_cache = {}


def parity(poolfile):
    """Out-of-fold predictions for the three targets on one training pool."""
    if poolfile in _cache:
        return _cache[poolfile]
    df = pd.read_csv(os.path.join(DATA, poolfile))
    df["composition_obj"] = df["formula"].apply(Composition)
    f = make_featurizer()
    df = f.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
    comp = f.feature_labels()
    full = comp + ["density", "vol_per_atom"]
    df = df.dropna(subset=full).reset_index(drop=True)
    k = df["G_GPa"] / df["K_GPa"]
    df["hardness_GPa"] = 2 * (k ** 2 * df["G_GPa"]) ** 0.585 - 3
    kf = KFold(5, shuffle=True, random_state=SEED)
    out = {}
    y = np.log10(df["youngs_modulus_GPa"].values)
    out["E"] = (df["youngs_modulus_GPa"].values,
                10 ** cross_val_predict(HistGradientBoostingRegressor(random_state=SEED),
                                        df[full], y, cv=kf))
    hv = df[df["hardness_GPa"] > 0].reset_index(drop=True)
    out["H"] = (hv["hardness_GPa"].values,
                cross_val_predict(RandomForestRegressor(n_estimators=300, n_jobs=4,
                                                        random_state=SEED),
                                  hv[full], hv["hardness_GPa"].values, cv=kf))
    vpa = cross_val_predict(HistGradientBoostingRegressor(random_state=SEED),
                            df[comp], df["vol_per_atom"].values, cv=kf)
    mm = np.array([c.weight / c.num_atoms for c in df["composition_obj"]])
    out["rho"] = (df["density"].values, AMU * mm / np.clip(vpa, 1e-3, None))
    _cache[poolfile] = out
    return out


def sub(formula):
    o, i = "", 0
    while i < len(formula):
        if formula[i].isdigit():
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            o += "$_{%s}$" % formula[i:j]
            i = j
        else:
            o += formula[i]
            i += 1
    return o


for case, cfg in CASES.items():
    p = parity(cfg["pool"])
    tE, tH, tR = cfg["target"]
    col = ps.CASE_COLOR[case]

    fig = plt.figure(figsize=(26.0, 15.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # (a) to (c) parity
    for i, (key, tgt, xl, yl) in enumerate([
            ("E", tE, "Database Young's Modulus (GPa)", "Predicted Modulus (GPa)"),
            ("H", tH, "Chen-model Hardness (GPa)", "Predicted Hardness (GPa)"),
            ("rho", tR, "Database Density (g/cm$^3$)", "Predicted Density (g/cm$^3$)")]):
        ax = fig.add_subplot(gs[0, i])
        x, yv = p[key]
        ax.plot(x, yv, ps.CASE_MARKER[case], color=col, markersize=9, markeredgecolor='white', markeredgewidth=0.8,
                linestyle="none")
        top = max(x.max(), yv.max()) * 1.05
        # pad the low side so markers sitting near zero are not cut by the spines
        lim = [-0.045 * top, top]
        ax.plot([0, top], [0, top], color="0.30", linewidth=2.2, linestyle=":")
        ax.axhline(tgt, color=ps.COLORS[3], linewidth=2.8, linestyle="--")
        ax.text(top * 0.97, tgt, " Target %g" % tgt, color=ps.COLORS[3],
                fontsize=ps.BASE - 6, ha="right", va="bottom")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ps.panel_tag(ax, "abc"[i])

    # (d) convergence across seeds
    ax = fig.add_subplot(gs[1, 0])
    h = hist[hist.case == case]
    for j, (sd, g) in enumerate(h.groupby("seed")):
        g = g.sort_values("generation")
        ax.plot(g.generation, g.running_best, linewidth=1.6, color="0.62",
                label="Individual Seeds" if j == 0 else None)
    piv = h.pivot_table(index="generation", columns="seed", values="running_best")
    m, sd_ = piv.mean(axis=1), piv.std(axis=1)
    ax.fill_between(piv.index, m - sd_, m + sd_, color=col, alpha=0.22,
                    label="Standard Deviation")
    ax.plot(piv.index, m, color=col, linewidth=3.6, marker=ps.CASE_MARKER[case],
            markevery=4, label="Mean of Five Seeds")
    pooled = piv.min(axis=1).values
    ax.plot(piv.index, pooled, color="0.25", linewidth=2.8, linestyle="--",
            label="Pooled Best Across Seeds")
    last = int(np.max(np.nonzero(np.diff(pooled, prepend=pooled[0] + 1))[0]) + 1)
    lo, hi = ax.get_ylim()
    # the marker of the last improvement stops at the highest curve, so it cannot
    # run up through the legend band; its meaning is carried by the legend
    top = float(max(piv.max().max(), (m + sd_).max()))
    ax.vlines(last, lo, top, color="0.30", linewidth=2.2, linestyle=":",
              label="Last Improvement, Generation %d" % last)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Objective Value")
    ps.headroom(ax, 0.52)
    ax.legend(loc="upper right", fontsize=ps.BASE - 6, labelspacing=0.35)
    ps.panel_tag(ax, "d")

    # (e) the reachable set as a distribution, which is what the population
    # scatter was trying to show: 300 individuals collapse to a handful of
    # distinct compositions, while the reachable set is tens of thousands
    ax = fig.add_subplot(gs[1, 1])
    en = enum[enum.pool == POOL_OF[case]]
    v = en.pred_modulus_GPa.values
    counts, _, _ = ax.hist(v, bins=60, color="0.72", edgecolor="0.45", linewidth=1.0,
                           label=ps.tc("reachable space, %s formulas" % format(len(v), ",")))
    hmax = float(counts.max())
    u15 = unc[unc.showcase == case]
    # band, target and floor all stop at the tallest bar, leaving the top of the
    # panel to the legend, which now carries the floor in place of an arrow note
    ax.fill_between([float(u15.pred_E_GPa.min()), float(u15.pred_E_GPa.max())], 0, hmax,
                    color=col, alpha=0.30, linewidth=0, label=ps.tc("fifteen best candidates"))
    ax.vlines(tE, 0, hmax, color=ps.COLORS[3], linewidth=3.0, linestyle="--",
              label=ps.tc("target, %g GPa" % tE))
    ax.vlines(float(v.min()), 0, 0.45 * hmax, color="0.15", linewidth=3.4,
              label="Lowest Prediction, %.1f GPa" % float(FLOOR[POOL_OF[case]]))
    ax.set_xlabel("Predicted Young's Modulus (GPa)")
    ax.set_ylabel("Reachable Formulas")
    ax.set_ylim(0, hmax * 1.04)
    ps.headroom(ax, 0.40)
    ax.legend(loc="upper right", fontsize=ps.BASE - 6, labelspacing=0.35)
    ps.panel_tag(ax, "e")

    # (f) fifteen best with conformal intervals
    ax = fig.add_subplot(gs[1, 2])
    u = unc[unc.showcase == case].sort_values("rank")
    x = np.arange(len(u)) + 1
    ax.bar(x, u.pred_E_GPa, width=0.62, color=col, edgecolor="white", linewidth=1.2)
    ax.errorbar(x, u.pred_E_GPa, yerr=[u.pred_E_GPa - u.E_lo_GPa,
                                       u.E_hi_GPa - u.pred_E_GPa],
                fmt="none", ecolor="0.20", elinewidth=2.6, capsize=6, capthick=2.6,
                label="90 Percent Interval")
    ax.axhline(tE, color=ps.COLORS[3], linewidth=3.0, linestyle="--",
               label="Target, %g GPa" % tE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=ps.BASE - 6)
    ax.set_xlabel("Candidate, Ranked by Objective")
    ax.set_ylabel("Predicted Modulus (GPa)")
    ax.set_ylim(0, float(u.E_hi_GPa.max()) * 1.04)
    ps.headroom(ax, 0.30)
    ax.legend(loc="upper right", fontsize=ps.BASE - 6, labelspacing=0.35)
    ps.panel_tag(ax, "f", dx=-0.20)

    path = ps.save(fig, cfg["stem"], OUT)
    print("wrote", path, flush=True)

print("REVISION_CASE_PANELS_DONE", flush=True)
