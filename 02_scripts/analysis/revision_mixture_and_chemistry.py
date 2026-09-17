"""Revision analysis R2 - what the decomposition mixtures are actually like.

Answers reviewer 1 points 5 and 6 and reviewer 3 point 6:

  * mixture modulus. Every candidate is predicted to decompose into 2-5 known
    phases. We take the consensus decomposition of Table S9 and bracket the
    assembly with the Voigt (upper), Reuss (lower) and Hill (mean) averages, so
    the multi-phase assembly carries a modulus of its own. The coefficients in
    that decomposition are pymatgen amounts, which are barycentric coordinates
    in atom-fraction space: coefficient i is the fraction of the assembly's
    atoms that belong to phase i, not the fraction of its formula units. Volume
    fractions therefore come from the volume per atom of each phase, and
    formula-unit mole fractions from dividing by the atoms per formula unit.
    The check that establishes this reading is in revision_audit_v2.py, which
    reconstructs every candidate under both readings. Product moduli come
    from the Materials Project elastic tensor where one exists, and otherwise
    from the same surrogate evaluated on the product's own Materials Project
    density and volume per atom, that is, in its true-descriptor configuration.
  * aqueous risk. For every candidate we report the atom, formula-unit and
    volume fraction of the decomposition that is a water-soluble alkali phase.
    Two sets are reported: the phases whose aqueous behaviour is documented in
    the CRC handbook, and the wider set that also includes the alkali silicates
    and the potassium calcium pyrophosphate, whose solubilities are assigned by
    analogy. This is a screen on phase identity and not a dissolution
    measurement.
  * mixed-anion audit of the virtual-screening shortlist, which was retrieved
    with an elements query and therefore admits oxynitrides and oxyfluorides
    that lie outside the oxide-only training domain, plus a reduced-formula
    overlap test against the training pool.

Writes 01_data/revision_mixture_modulus.csv, 01_data/revision_phase_moduli.csv
and 01_data/revision_shortlist_audit.csv.
"""
import os
import re
import sys
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
FIELDS = ("material_id,formula_pretty,energy_above_hull,bulk_modulus,"
          "shear_modulus,density,volume,nsites")

# Freely water-soluble alkali phases. The classification is by phase identity
# from the literature, not by any model prediction; sources are quoted in the
# response letter.
# Documented in the CRC handbook: the alkali orthophosphates dissolve freely and
# the alkali oxides hydrolyse to the hydroxide.
SOLUBLE_DOCUMENTED = {"K3PO4", "Na3PO4", "K2O", "Na2O"}
# Assigned by analogy with the alkali silicate glasses and the alkali
# pyrophosphates; no primary solubility datum is quoted for these.
SOLUBLE_BY_ANALOGY = {"K2Si2O5", "K2Si4O9", "Na2Si2O5", "K4P2O7", "Na4P2O7",
                      "KPO3", "NaPO3", "K2CaP2O7"}
SOLUBLE = SOLUBLE_DOCUMENTED | SOLUBLE_BY_ANALOGY
MIXED_ANION_ELEMENTS = {"N", "F", "Cl", "Br", "I", "S", "Se", "Te", "C", "H"}


def mp_query(formula):
    for attempt in range(4):
        try:
            r = requests.get(API, headers=HDR, timeout=90,
                             params={"formula": formula, "_fields": FIELDS,
                                     "_limit": 80})
            if r.status_code == 200:
                return r.json().get("data", [])
        except Exception as exc:
            print(f"    retry {formula}: {exc}", flush=True)
        time.sleep(2 + 2 * attempt)
    return []


def parse_decomp(s):
    out = []
    for tok in str(s).split(" + "):
        m = re.match(r"^(.*?)\s*\(([0-9]*\.?[0-9]+)\)$", tok.strip())
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out


def E_from_KG(K, G):
    return 9.0 * K * G / (3.0 * K + G)


TARGET_E = {"enamel": 85.0, "dentin": 20.0, "implant": 110.0}

st = pd.read_csv(os.path.join(ROOT, "06_mlff_stability", "stability_mlff.csv"))
unc = pd.read_csv(os.path.join(DATA, "uncertainty_intervals.csv"))
pred = {(r["showcase"], r["composition"]): float(r["pred_E_GPa"])
        for _, r in unc.iterrows()}

phases = sorted({f for s in st["decomposition_consensus"].dropna()
                 for f, _ in parse_decomp(s)})
print(f"{len(phases)} distinct decomposition phases", flush=True)

info = {}
for i, p in enumerate(phases, 1):
    data = mp_query(p)
    if not data:
        print(f"  [{i:2d}] {p:22s} NOT FOUND", flush=True)
        continue
    base = min(data, key=lambda d: d.get("energy_above_hull") or 9e9)
    el = [d for d in data
          if d.get("bulk_modulus") and d.get("shear_modulus")
          and d["bulk_modulus"].get("vrh") is not None
          and d["shear_modulus"].get("vrh") is not None]
    rec = dict(formula=p, material_id=base["material_id"],
               density=base["density"],
               vol_per_atom=base["volume"] / base["nsites"],
               e_above_hull=base.get("energy_above_hull"))
    if el:
        b = min(el, key=lambda d: d.get("energy_above_hull") or 9e9)
        rec.update(K=b["bulk_modulus"]["vrh"], G=b["shear_modulus"]["vrh"],
                   E=E_from_KG(b["bulk_modulus"]["vrh"], b["shear_modulus"]["vrh"]),
                   E_source="materials project elastic tensor",
                   elastic_material_id=b["material_id"])
    else:
        rec.update(K=np.nan, G=np.nan, E=np.nan, E_source="surrogate",
                   elastic_material_id="")
    info[p] = rec
    print(f"  [{i:2d}] {p:22s} {rec['E_source']:32s} "
          f"E {'-' if np.isnan(rec['E']) else format(rec['E'], '.1f')}", flush=True)

# surrogate fallback, evaluated on each phase's own MP density and vol/atom
need = [p for p, r in info.items() if r["E_source"] == "surrogate"]
if need:
    print(f"\nsurrogate fallback for {len(need)} phases with no MP elastic tensor",
          flush=True)
    mod = joblib.load(os.path.join(DATA, "model_modulus.joblib"))
    f = make_featurizer()
    d = pd.DataFrame({"composition_obj": [Composition(p) for p in need]})
    d = f.featurize_dataframe(d, col_id="composition_obj", ignore_errors=True)
    d["density"] = [info[p]["density"] for p in need]
    d["vol_per_atom"] = [info[p]["vol_per_atom"] for p in need]
    X = d[mod["features"]].copy()
    X = X.fillna(X.mean())
    yp = mod["model"].predict(X)
    yp = 10 ** yp if mod["log10_target"] else yp
    for p, e in zip(need, yp):
        info[p]["E"] = float(e)

pd.DataFrame([info[p] for p in phases if p in info]).to_csv(
    os.path.join(DATA, "revision_phase_moduli.csv"), index=False)
print("wrote 01_data/revision_phase_moduli.csv", flush=True)

rows = []
for _, r in st.iterrows():
    dec = parse_decomp(r["decomposition_consensus"])
    if not dec or any(f not in info for f, _ in dec):
        print(f"  skipping {r['composition']}: missing phase data", flush=True)
        continue
    # coefficient i is the fraction of the assembly's atoms in phase i
    a = np.array([v for _, v in dec], float)
    a = a / a.sum()
    nat = np.array([Composition(f).num_atoms for f, _ in dec], float)
    nfu = a / nat
    nfu = nfu / nfu.sum()                    # formula-unit mole fractions
    vpa = np.array([info[f]["vol_per_atom"] for f, _ in dec], float)
    vf = a * vpa
    vf = vf / vf.sum()                       # volume fractions
    E = np.array([info[f]["E"] for f, _ in dec], float)
    sol = np.array([f in SOLUBLE for f, _ in dec], bool)
    sold = np.array([f in SOLUBLE_DOCUMENTED for f, _ in dec], bool)
    n_mp = sum(1 for f, _ in dec if info[f]["E_source"] != "surrogate")

    E_v = float((vf * E).sum())
    E_r = float(1.0 / (vf / E).sum())
    rows.append(dict(
        showcase=r["showcase"], rank=int(r["rank"]), composition=r["composition"],
        pred_E_single_phase_GPa=pred.get((r["showcase"], r["composition"]), np.nan),
        n_phases=len(dec), n_phases_mp_elastic=n_mp,
        E_voigt_GPa=round(E_v, 1), E_reuss_GPa=round(E_r, 1),
        E_hill_GPa=round(0.5 * (E_v + E_r), 1),
        soluble_atom_frac=round(float(a[sol].sum()), 3),
        soluble_mole_frac=round(float(nfu[sol].sum()), 3),
        soluble_vol_frac=round(float(vf[sol].sum()), 3),
        soluble_vol_frac_documented=round(float(vf[sold].sum()), 3),
        target_E=TARGET_E[r["showcase"]],
        target_in_mixture_bounds=bool(E_r <= TARGET_E[r["showcase"]] <= E_v),
        soluble_phases="+".join(f for (f, _), s in zip(dec, sol) if s),
        decomposition=r["decomposition_consensus"]))

mix = pd.DataFrame(rows)
mix.to_csv(os.path.join(DATA, "revision_mixture_modulus.csv"), index=False)
print(f"\nwrote 01_data/revision_mixture_modulus.csv  ({len(mix)}/45 candidates)",
      flush=True)

TARGET = {"enamel": 85.0, "dentin": 20.0, "implant": 110.0}
print("\n---- mixture modulus by case study ----", flush=True)
for sc in ("enamel", "dentin", "implant"):
    s = mix[mix.showcase == sc]
    if not len(s):
        continue
    print(f"  {sc:8s} target {TARGET[sc]:5.0f} GPa | single phase "
          f"{s.pred_E_single_phase_GPa.min():5.1f}-{s.pred_E_single_phase_GPa.max():5.1f}"
          f" | mixture Hill {s.E_hill_GPa.min():5.1f}-{s.E_hill_GPa.max():5.1f}"
          f" (median {s.E_hill_GPa.median():5.1f})"
          f" | Reuss floor {s.E_reuss_GPa.min():5.1f}", flush=True)
    print(f"           target inside [Reuss, Voigt]: "
          f"{int(s.target_in_mixture_bounds.sum())}/{len(s)}", flush=True)
    print(f"           soluble alkali phase: atoms "
          f"{s.soluble_atom_frac.median():.2f}, formula units "
          f"{s.soluble_mole_frac.median():.2f}, volume "
          f"{s.soluble_vol_frac.median():.2f} (documented phases only "
          f"{s.soluble_vol_frac_documented.median():.2f}); "
          f"{int((s.soluble_vol_frac > 0).sum())}/{len(s)} candidates release one",
          flush=True)

# --------------------------------------------------------- shortlist audit
sl = pd.read_csv(os.path.join(DATA, "shortlist.csv"))
mp = pd.read_csv(os.path.join(DATA, "mp_data.csv"))
train_red = {Composition(f).reduced_formula for f in mp["formula"]}
train_ids = set(mp["material_id"])
aud = []
for _, r in sl.iterrows():
    c = Composition(r["formula"])
    els = {str(e) for e in c.elements}
    extra = sorted(els & MIXED_ANION_ELEMENTS)
    aud.append(dict(material_id=r["material_id"], formula=r["formula"],
                    pred_modulus_GPa=round(float(r["pred_modulus_GPa"]), 1),
                    mixed_anion=bool(extra), foreign_anions=",".join(extra),
                    id_in_training=r["material_id"] in train_ids,
                    reduced_formula_in_training=c.reduced_formula in train_red))
au = pd.DataFrame(aud)
au.to_csv(os.path.join(DATA, "revision_shortlist_audit.csv"), index=False)
print("\nwrote 01_data/revision_shortlist_audit.csv", flush=True)
print(f"  shortlist entries                     : {len(au)}")
print(f"  mixed-anion (non-oxide) entries       : {int(au.mixed_anion.sum())}"
      f"  -> {', '.join(au[au.mixed_anion].formula.tolist())}")
print(f"  material_id already in training pool  : {int(au.id_in_training.sum())}")
print(f"  reduced formula also in training pool : "
      f"{int(au.reduced_formula_in_training.sum())}"
      f"  -> {', '.join(au[au.reduced_formula_in_training].formula.tolist())}",
      flush=True)
print("\nREVISION_MIXTURE_DONE", flush=True)
