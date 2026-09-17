"""Revision analysis R4 - how converged the structure sampling is, and how the
surrogate performs on compounds it never saw.

Part A answers reviewer 1 point 3. From the stored per-trial relaxation energies we
report, per composition and per force field: how many of the eight trial structures
returned an energy, how many reached the force convergence criterion, the spread of
relaxed energy across trials, and the gap between the best and second-best trial. We
then compare the energy improvement a hypothetical better structure would need in
order to reach the hull against the spread the sampling actually shows, which is the
quantity that decides whether the verdict is safe against undersampling.

Part B answers reviewer 3 points 5 and the principal comment. We assemble an external
held-out set of Materials Project oxides that carry an elastic tensor and are absent
from the training pool by both material identifier and reduced formula, predict their
Young's modulus with the shipped models through the complete design-time chain, and
report accuracy and the measured coverage of the stated 90 percent interval. This is
a like-for-like test of the surrogate against the physics it predicts, on compounds it
never saw.

Writes 01_data/revision_sampling.csv, 01_data/revision_sampling_summary.csv and
01_data/revision_holdout.csv.
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import requests
import joblib
from pymatgen.core import Composition

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import make_featurizer
from paths import DATA, ROOT

API = "https://api.materialsproject.org/materials/summary/"
KEY = open(os.path.join(DATA, "mp_key.txt")).read().strip()
HDR = {"X-API-KEY": KEY}
AMU = 1.66054
MODELS = ["chgnet", "m3gnet", "mace", "mattersim", "omni"]
MODEL_LABEL = {"omni": "sevennet"}

# ------------------------------------------------------------------ part A
res = []
for f in sorted(glob.glob(os.path.join(ROOT, "06_mlff_stability", "results", "*.csv"))):
    try:
        res.append(pd.read_csv(f))
    except Exception as exc:
        print(f"  could not read {os.path.basename(f)}: {exc}", flush=True)
r = pd.concat(res, ignore_index=True)
man = pd.read_csv(os.path.join(ROOT, "06_mlff_stability", "manifest.csv"))
cand = r[r.kind == "candidate"].merge(
    man[["sid", "showcase", "rank"]], on="sid", how="left")
cand = cand[cand.model.isin(MODELS)]
print(f"per-trial candidate rows: {len(cand)}  models: {sorted(cand.model.unique())}",
      flush=True)

rows = []
for (sc, rk, comp, model), g in cand.groupby(
        ["showcase", "rank", "composition", "model"]):
    e = g["energy_per_atom"].dropna().values
    srt = np.sort(e)
    rows.append(dict(
        showcase=sc, rank=int(rk), composition=comp,
        model=MODEL_LABEL.get(model, model),
        n_trials=len(g), n_energies=len(e),
        n_converged=int((g["converged"] == 1).sum()),
        best_eV_atom=round(float(srt[0]), 4) if len(e) else np.nan,
        spread_eV_atom=round(float(srt[-1] - srt[0]), 4) if len(e) > 1 else np.nan,
        best_to_second_eV_atom=round(float(srt[1] - srt[0]), 4) if len(e) > 1 else np.nan))
samp = pd.DataFrame(rows)
samp.to_csv(os.path.join(DATA, "revision_sampling.csv"), index=False)
print("wrote 01_data/revision_sampling.csv", flush=True)

print("\n---- per force field, across all candidate trials ----", flush=True)
for model, g in samp.groupby("model"):
    print(f"  {model:10s} trials {int(g.n_trials.sum()):4d} | energies returned "
          f"{int(g.n_energies.sum()):4d} | reached force criterion "
          f"{int(g.n_converged.sum()):4d} "
          f"({100.0 * g.n_converged.sum() / max(1, g.n_trials.sum()):.0f}%) | "
          f"median spread across trials {g.spread_eV_atom.median():.3f} eV/atom",
          flush=True)

# does the sampling spread threaten the verdict?
st = pd.read_csv(os.path.join(ROOT, "06_mlff_stability", "stability_mlff.csv"))
sm = samp.groupby(["showcase", "rank", "composition"]).agg(
    median_spread=("spread_eV_atom", "median"),
    max_spread=("spread_eV_atom", "max"),
    best_to_second=("best_to_second_eV_atom", "median"),
    converged_frac=("n_converged", "sum"),
    trials=("n_trials", "sum")).reset_index()
sm["converged_frac"] = (sm.converged_frac / sm.trials).round(3)
sm = sm.merge(st[["showcase", "rank", "ehull_median", "ehull_min", "ehull_max"]],
              on=["showcase", "rank"], how="left")
sm["required_improvement"] = sm.ehull_median          # to reach the hull
sm["margin_over_spread"] = (sm.ehull_median / sm.median_spread).round(2)
sm.to_csv(os.path.join(DATA, "revision_sampling_summary.csv"), index=False)
print("\nwrote 01_data/revision_sampling_summary.csv", flush=True)
print("---- would a better structure change the verdict? ----", flush=True)
for sc in ("enamel", "dentin", "implant"):
    s = sm[sm.showcase == sc]
    print(f"  {sc:8s} energy above hull {s.ehull_median.min():.2f}-"
          f"{s.ehull_median.max():.2f} eV/atom | median trial-to-trial spread "
          f"{s.median_spread.median():.3f} | required improvement is "
          f"{s.margin_over_spread.min():.1f}-{s.margin_over_spread.max():.1f}x "
          f"the observed spread", flush=True)
n_risk = int((sm.margin_over_spread < 1.0).sum())
print(f"  candidates where the required improvement is smaller than the observed "
      f"sampling spread: {n_risk} of {len(sm)}", flush=True)
print(f"  closest such candidates: "
      f"{', '.join(sm.nsmallest(3, 'margin_over_spread').composition)}", flush=True)

# ------------------------------------------------------------------ part B
mp = pd.read_csv(os.path.join(DATA, "mp_data.csv"))
train_ids = set(mp["material_id"])
train_red = {Composition(f).reduced_formula for f in mp["formula"]}
print(f"\ntraining pool: {len(train_ids)} identifiers, {len(train_red)} reduced formulas",
      flush=True)

fields = ("material_id,formula_pretty,energy_above_hull,bulk_modulus,shear_modulus,"
          "density,volume,nsites,nelements,elements")
PER_PAGE = 100
found, page, fails = [], 1, 0
while page <= 80:
    try:
        resp = requests.get(API, headers=HDR, timeout=120,
                            params={"elements": "O", "has_props": "elasticity",
                                    "_fields": fields, "_limit": PER_PAGE,
                                    "_skip": (page - 1) * PER_PAGE})
    except Exception as exc:
        fails += 1
        print(f"  page {page} failed: {exc}", flush=True)
        if fails > 6:
            break
        time.sleep(5)
        continue
    if resp.status_code != 200:
        print(f"  page {page} status {resp.status_code}: {resp.text[:160]}", flush=True)
        break
    data = resp.json().get("data", [])
    if not data:
        break
    found.extend(data)
    if page % 10 == 1 or len(data) < PER_PAGE:
        print(f"  page {page}: {len(data)} entries (total {len(found)})", flush=True)
    if len(data) < PER_PAGE:
        break
    page += 1
    time.sleep(0.25)
print(f"  retrieved {len(found)} oxide entries with an elastic tensor", flush=True)

hold = []
for d in found:
    els = set(d.get("elements") or [])
    if "O" not in els or len(els) < 2 or len(els) > 5:
        continue
    bm, sh = d.get("bulk_modulus"), d.get("shear_modulus")
    if not bm or not sh or bm.get("vrh") is None or sh.get("vrh") is None:
        continue
    K, G = bm["vrh"], sh["vrh"]
    if K <= 0 or G <= 0:
        continue
    try:
        c = Composition(d["formula_pretty"])
    except Exception:
        continue
    if d["material_id"] in train_ids or c.reduced_formula in train_red:
        continue
    hold.append(dict(material_id=d["material_id"], formula=d["formula_pretty"],
                     reduced_formula=c.reduced_formula,
                     K_GPa=K, G_GPa=G,
                     youngs_modulus_GPa=9.0 * K * G / (3.0 * K + G),
                     density=d["density"],
                     vol_per_atom=d["volume"] / d["nsites"],
                     energy_above_hull=d.get("energy_above_hull")))
hd = pd.DataFrame(hold).drop_duplicates("reduced_formula").reset_index(drop=True)
print(f"\nheld-out oxides with an elastic tensor and absent from training: {len(hd)}",
      flush=True)
if not len(hd):
    print("no held-out set could be built", flush=True)
    raise SystemExit

mod = joblib.load(os.path.join(DATA, "model_modulus.joblib"))
vol = joblib.load(os.path.join(DATA, "model_volperatom.joblib"))
f = make_featurizer()
hd["composition_obj"] = hd["formula"].apply(Composition)
hd = f.featurize_dataframe(hd, col_id="composition_obj", ignore_errors=True)
cc = vol["features"]
hd[cc] = hd[cc].fillna(hd[cc].mean())

# design-time chain, exactly as the search uses it
vpa = np.clip(vol["model"].predict(hd[cc]), 1e-3, None)
mm = np.array([c.weight / c.num_atoms for c in hd["composition_obj"]])
rho = AMU * mm / vpa
Xd = hd[cc].copy()
Xd["density"] = rho
Xd["vol_per_atom"] = vpa
pred_design = 10 ** mod["model"].predict(Xd[mod["features"]])

# and with the database structure, for comparison
Xt = hd[cc].copy()
Xt["density"] = hd["density"].values
Xt["vol_per_atom"] = hd["vol_per_atom"].values
pred_true = 10 ** mod["model"].predict(Xt[mod["features"]])

y = hd["youngs_modulus_GPa"].values
val = pd.read_csv(os.path.join(DATA, "revision_validation.csv"))
q = float(val[(val.pool == "enamel_implant") & (val.target == "youngs_modulus_GPa")
              & (val.split == "chemsys") & (val.features == "design")].conformal_q.iloc[0])
lo, hi = pred_design * 10 ** -q, pred_design * 10 ** q
cov = float(np.mean((y >= lo) & (y <= hi)))

out = pd.DataFrame(dict(
    material_id=hd["material_id"], formula=hd["formula"],
    dft_E_GPa=np.round(y, 1),
    pred_design_E_GPa=np.round(pred_design, 1),
    pred_database_structure_E_GPa=np.round(pred_true, 1),
    interval_lo_GPa=np.round(lo, 1), interval_hi_GPa=np.round(hi, 1),
    inside_interval=(y >= lo) & (y <= hi),
    energy_above_hull=hd["energy_above_hull"]))
out.to_csv(os.path.join(DATA, "revision_holdout.csv"), index=False)
print("wrote 01_data/revision_holdout.csv", flush=True)

from sklearn.metrics import r2_score, mean_absolute_error
print("\n---- external held-out set, never seen in training ----", flush=True)
print(f"  n = {len(hd)} oxides", flush=True)
print(f"  design-time chain      : R2 {r2_score(np.log10(y), np.log10(pred_design)):.3f}"
      f"  MAE {mean_absolute_error(y, pred_design):.1f} GPa", flush=True)
print(f"  database structure     : R2 {r2_score(np.log10(y), np.log10(pred_true)):.3f}"
      f"  MAE {mean_absolute_error(y, pred_true):.1f} GPa", flush=True)
print(f"  stated 90 percent interval covers {100 * cov:.1f} percent of them "
      f"(nominal 90)", flush=True)
print(f"  modulus range of the held-out set: {y.min():.0f}-{y.max():.0f} GPa", flush=True)
print("\nREVISION_SAMPLING_DONE", flush=True)
