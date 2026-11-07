"""Rebuild the three case-study figures in the revision style.

Panels, in the layout the manuscript already describes:
  (a) to (c)  parity for modulus, hardness and density, out-of-fold
  (d)         convergence, across-seed mean running best with a standard-deviation band
              and the five individual seed traces
  (e)         the pooled final population in the plane of predicted modulus and predicted
              density, with the target marked
  (f)         the fifteen candidates selected from the genetic-algorithm searches,
              objective value, with their nominal 90 percent prediction intervals

Arrows name example compounds in every panel: three familiar oxides and salts of the
training pool in (a) to (c), the best composition of the five runs in (d), the
lowest- and highest-modulus reachable formulas in (e) and the three top-ranked
candidates in (f).

Writes Figure4_enamel_panels_rev, Figure5_dentin_panels_rev and
Figure6_implant_panels_rev into 04_manuscript/revision_figures/.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
from paths import DATA, MANUSCRIPT

ps.use()
OUT = os.path.join(MANUSCRIPT, "revision_figures")
os.makedirs(OUT, exist_ok=True)

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
FLOORC = pd.read_csv(os.path.join(DATA, "revision_enumeration.csv")).set_index("pool").floor_composition
pop = pd.read_csv(os.path.join(DATA, "revision_ga_population.csv"))
# the polymorph named in (a) to (c) is the lowest-energy one of that formula in the pool
HULL = pd.concat([pd.read_csv(os.path.join(DATA, f), usecols=["material_id", "energy_above_hull"])
                  for f in ("mp_data.csv", "dentin_data.csv")]).drop_duplicates("material_id")
HULL = HULL.set_index("material_id").energy_above_hull
LANDMARKS = {"mp_data.csv": ("CaCO3", "CaSiO3", "Al2O3"),
             "dentin_data.csv": ("K3PO4", "CaSiO3", "Al2O3")}
COLS = {"E": ("E_GPa", "E_pred_GPa"), "H": ("H_GPa", "H_pred_GPa"),
        "rho": ("density_gcc", "density_pred_gcc")}

_cache = {}


def parity(poolfile):
    """Out-of-fold predictions for the three targets on one training pool.

    Written by revision_validation.py with the released model chain (chain.py),
    refitted inside each shuffled five-fold split, database structural inputs for
    modulus and hardness and the volume-per-atom helper for density.
    """
    if poolfile in _cache:
        return _cache[poolfile]
    d = pd.read_csv(os.path.join(DATA, "revision_parity_oof.csv"))
    d = d[d.pool == ("dentin" if poolfile == "dentin_data.csv" else "enamel_implant")]
    hv = d.dropna(subset=["H_GPa"])
    out = {"E": (d["E_GPa"].values, d["E_pred_GPa"].values),
           "H": (hv["H_GPa"].values, hv["H_pred_GPa"].values),
           "rho": (d["density_gcc"].values, d["density_pred_gcc"].values)}
    _cache[poolfile] = out
    return out


def landmarks(poolfile):
    """Ground-state polymorphs of the named compounds, among those with a hardness value,
    so the same material is marked in all three parity panels."""
    d = pd.read_csv(os.path.join(DATA, "revision_parity_oof.csv"))
    d = d[d.pool == ("dentin" if poolfile == "dentin_data.csv" else "enamel_implant")]
    d = d.dropna(subset=["H_GPa"]).assign(ehull=lambda t: t.material_id.map(HULL))
    rows = []
    for f in LANDMARKS[poolfile]:
        g = d[d.formula == f].dropna(subset=["ehull"]).sort_values("ehull")
        if g.empty:
            raise SystemExit("landmark %s is not in the %s pool" % (f, poolfile))
        rows.append(g.iloc[0])
    return rows


def note(ax, formula, xy, at, ha, side, coords="data"):
    """Name one compound with an arrow pointing at its marker.

    The label is a plain text, so the overlap check tests it against every text and
    data mark like any other label; the arrow leaves from `side` of the label, given as
    a fraction of its box, and is meant to reach into the data."""
    t = ax.text(at[0], at[1], sub(formula), ha=ha, va="center", fontsize=ps.BASE - 6,
                color="0.10", transform=ax.transAxes if coords == "axes fraction" else ax.transData)
    ax.annotate("", xy=xy, xycoords="data", xytext=side, textcoords=t,
                arrowprops=dict(arrowstyle="-|>", color="0.10", linewidth=2.0,
                                mutation_scale=24, shrinkA=3, shrinkB=5))


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
        ax.text(top * 0.97, tgt, (" Reference %g GPa" if key == "H" else " Target %g") % tgt, color=ps.COLORS[3],
                fontsize=ps.BASE - 6, ha="right", va="bottom")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        # labels in a row across the empty top of the panel, in the order of the
        # database values, so the arrows run down without crossing a label
        marks = sorted(landmarks(cfg["pool"]), key=lambda r: float(r[COLS[key][0]]))
        for j, r in enumerate(marks):
            note(ax, r.formula, (float(r[COLS[key][0]]), float(r[COLS[key][1]])),
                 (0.10 + 0.22 * j, 0.88), "center", (0.5, 0.0), coords="axes fraction")
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
    best = pop[pop.case == case].sort_values("fitness").iloc[0]
    if abs(float(best.fitness) - float(pooled[-1])) > 1e-6:
        raise SystemExit("%s: best population member %.4f differs from the pooled best %.4f"
                         % (case, best.fitness, pooled[-1]))
    span = hi - lo
    ax.set_ylim(lo - 0.45 * span, hi)
    # the label sits below the pooled-best line, which is the lowest curve, on the
    # side of the last-improvement line that has room
    note(ax, best.composition, (last, float(pooled[last - 1])),
         (last + 2.5 if last < 13 else last - 2.5, float(pooled[last - 1]) - 0.30 * span),
         "left" if last < 13 else "right", (0.5, 1.0))
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
                    color=col, alpha=0.30, linewidth=0, label=ps.tc("fifteen genetic-algorithm candidates"))
    ax.vlines(tE, 0, hmax, color=ps.COLORS[3], linewidth=3.0, linestyle="--",
              label=ps.tc("target, %g GPa" % tE))
    ax.vlines(float(v.min()), 0, 0.45 * hmax, color="0.15", linewidth=3.4,
              label="Lowest Prediction, %.1f GPa" % float(FLOOR[POOL_OF[case]]))
    ax.set_xlabel("Predicted Young's Modulus (GPa)")
    ax.set_ylabel("Reachable Formulas")
    ax.set_ylim(0, hmax * 1.04)
    top = en.loc[en.pred_modulus_GPa.idxmax()]
    if en.loc[en.pred_modulus_GPa.idxmin()].composition != FLOORC[POOL_OF[case]]:
        raise SystemExit("%s: lowest reachable formula differs from the floor table" % case)
    # the floor label sits above the tallest bar and below the legend band
    note(ax, FLOORC[POOL_OF[case]], (float(v.min()), 0.45 * hmax),
         (float(v.min()), 1.16 * hmax), "left", (0.0, 0.0))
    note(ax, top.composition, (float(top.pred_modulus_GPa), 0.02 * hmax),
         (float(top.pred_modulus_GPa), 0.60 * hmax), "right", (1.0, 0.0))
    ps.headroom(ax, 0.48)
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
    ps.headroom(ax, 0.58)
    ytop = ax.get_ylim()[1]
    # a staircase of labels whose arrows leave from their left ends, so no arrow
    # crosses a label
    for i in range(3):
        r = u.iloc[i]
        note(ax, r.composition, (x[i], float(r.E_hi_GPa)),
             (x[i] + 0.3, ytop * (0.90 - 0.165 * i)), "left", (0.0, 0.0))
    ax.legend(loc="upper right", fontsize=ps.BASE - 6, labelspacing=0.35)
    ps.panel_tag(ax, "f", dx=-0.20)

    path = ps.save(fig, cfg["stem"], OUT)
    print("wrote", path, flush=True)

print("REVISION_CASE_PANELS_DONE", flush=True)
