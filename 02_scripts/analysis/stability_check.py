"""
Showcase 4 - Stability check via Materials Project convex hull.

For every GA-designed composition from showcases 1 (enamel), 2 (dentin), and
3 (implant interface), we ask: how close is this hypothetical composition to
the Materials Project convex hull?  The composition itself has no DFT energy
(it's a designed candidate, not a computed entry), so we cannot report
'energy above hull' for it directly.  What we CAN report is:

  - the chemical system (chemsys) of the candidate
  - how many MP entries already exist in that chemsys (chemistry coverage)
  - the convex-hull energy at the candidate composition -- i.e. the energy of
    the lowest-energy equilibrium mixture of existing stable phases at this
    composition (lower => the existing mixture is more competitive, so a
    hypothetical single phase would need a deeper formation energy to displace
    it).  NOTE: pymatgen's get_hull_energy() returns a TOTAL energy that is NOT
    normalised per atom; because each candidate is parsed as mole fractions
    summing to ~1 atom, the number is numerically ~eV/atom, but it is not a
    formation energy and is not an energy-above-hull for the candidate.
  - the predicted decomposition: which existing stable phases would form a
    mixture at the candidate's composition, and in what proportions.  This is a
    mixture of 3-5 known phases for every candidate, i.e. none of them is
    predicted to be a single phase.

This addresses (but does not fully resolve) the explicit "GA does not enforce
thermodynamic stability" caveat in the showcase 1, 2 and 3 sections.

Output: data/stability_check.csv (rows = all GA candidates with stability columns)
"""
import os
import time
import pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from mp_api.client import MPRester
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA

KEY = os.environ.get("MP_API_KEY")
if not KEY:
    with open(os.path.join(DATA, "mp_key.txt")) as f:
        KEY = f.read().strip()


def parse_composition(s):
    """'O0.66 Ce0.10 Ti0.10 P0.09 Sr0.06' -> pymatgen Composition."""
    return Composition(s)  # composition column is now a reduced integer formula


def chemsys_str(comp):
    return "-".join(sorted({str(e) for e in comp.elements}))


SHOWCASES = [
    ("enamel",  "designed_compositions.csv"),
    ("dentin",  "designed_dentin_compositions.csv"),
    ("implant", "designed_implant_compositions.csv"),
]


# Collect every candidate first
candidates = []   # list of dicts
unique_chemsys = set()
for showcase_name, csv_file in SHOWCASES:
    df = pd.read_csv(os.path.join(DATA, csv_file))
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        comp = parse_composition(row.composition)
        cs   = chemsys_str(comp)
        candidates.append({
            "showcase":   showcase_name,
            "rank":       rank,
            "composition": row.composition,
            "comp_obj":   comp,
            "chemsys":    cs,
            "pred_E":     row.pred_modulus_GPa,
            "pred_H":     row.pred_hardness_GPa,
            "pred_rho":   row.pred_density_gcc,
            "fitness":    row.fitness,
        })
        unique_chemsys.add(cs)

print(f"Loaded {len(candidates)} GA candidates across "
      f"{len(unique_chemsys)} distinct chemical systems")


# Query MP once per chemsys and build phase diagrams
phase_diagrams = {}
n_entries_by_chemsys = {}
with MPRester(KEY) as mpr:
    for cs in sorted(unique_chemsys):
        elements = cs.split("-")
        print(f"  fetching MP entries for chemsys {cs} ...", end=" ", flush=True)
        try:
            entries = mpr.get_entries_in_chemsys(elements)
        except Exception as e:
            print(f"FAILED: {e}")
            entries = []
        n_entries_by_chemsys[cs] = len(entries)
        if entries:
            try:
                phase_diagrams[cs] = PhaseDiagram(entries)
                print(f"{len(entries)} entries, PD built.")
            except Exception as e:
                print(f"{len(entries)} entries, PD failed: {e}")
                phase_diagrams[cs] = None
        else:
            phase_diagrams[cs] = None
            print("no entries.")
        time.sleep(0.1)


# For each candidate, compute hull energy + decomposition
rows = []
for cand in candidates:
    cs   = cand["chemsys"]
    comp = cand["comp_obj"]
    pd_obj = phase_diagrams.get(cs)
    n_mp  = n_entries_by_chemsys.get(cs, 0)

    if pd_obj is None:
        e_hull = None
        decomp_str = "no MP coverage of this chemsys"
    else:
        try:
            e_hull = pd_obj.get_hull_energy(comp) / comp.num_atoms  # per atom, comparable across formula sizes
            # get_decomposition returns {Entry: fraction}
            decomp = pd_obj.get_decomposition(comp)
            parts = []
            for entry, frac in sorted(decomp.items(), key=lambda kv: -kv[1]):
                parts.append(f"{entry.composition.reduced_formula}({frac:.2f})")
            decomp_str = " + ".join(parts)
        except Exception as e:
            e_hull = None
            decomp_str = f"PD lookup failed: {e}"

    rows.append({
        "showcase":           cand["showcase"],
        "rank":               cand["rank"],
        "composition":        cand["composition"],
        "chemsys":            cs,
        "n_mp_entries":       n_mp,
        "hull_energy_eV":     round(e_hull, 3) if e_hull is not None else None,
        "decomposition":      decomp_str,
        "pred_modulus_GPa":   cand["pred_E"],
        "pred_hardness_GPa":  cand["pred_H"],
        "pred_density_gcc":   cand["pred_rho"],
        "fitness":            cand["fitness"],
    })

out_df = pd.DataFrame(rows)
out_path = os.path.join(DATA, "stability_check.csv")
out_df.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")

# Print summary per showcase
for sc in ("enamel", "dentin", "implant"):
    sub = out_df[out_df["showcase"] == sc]
    print(f"\n[{sc}]  N={len(sub)}  hull-energy at composition "
          f"(total eV for a ~1-atom composition, i.e. ~eV/atom; NOT a "
          f"formation energy; lower => existing mixture is more competitive):")
    print(sub[["rank", "composition", "chemsys",
               "n_mp_entries", "hull_energy_eV"]].to_string(index=False))
