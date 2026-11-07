"""
Synthesizability confirmation for the GA-designed compositions.

The genetic algorithm now emits SMALL-INTEGER, charge-balanced oxide formulae by
construction (see decode() in inverse_ga*.py). This script independently confirms
that, for every shortlisted candidate, the reported formula is:

  (1) a valid stoichiometric oxide (parses to a pymatgen Composition),
  (2) charge-balanced under the same fixed oxidation states the GA used, and
  (3) a SMALL formula unit -- the number of atoms per reduced formula unit, which
      is the practical synthesizability signal (a simple crystalline phase has a
      small cell; a candidate that needs a very large cell is better read as a
      glass / glass-ceramic).

It reports the formula-unit size per candidate so the manuscript can state the
range explicitly rather than asserting synthesizability.

Output: data/synthesizable_candidates.csv
"""
import os
import pandas as pd
from pymatgen.core import Composition
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA

# combined fixed oxidation states across the three GAs (enamel/implant/dentin)
CHARGE = {"Zr": 4, "Ti": 4, "Si": 4, "Ce": 4, "P": 5,
          "Al": 3, "Y": 3, "La": 3, "Ca": 2, "Mg": 2, "Sr": 2,
          "Na": 1, "K": 1}

SMALL_FU_ATOMS = 20    # <= this many atoms / formula unit reads as a simple phase

SHOWCASES = [
    ("enamel",  "designed_compositions.csv"),
    ("dentin",  "designed_dentin_compositions.csv"),
    ("implant", "designed_implant_compositions.csv"),
]


def charge_balanced(comp):
    """True if sum(cation amount * fixed charge) == 2 * O amount."""
    amt = comp.get_el_amt_dict()
    if "O" not in amt:
        return False
    cats = {e: a for e, a in amt.items() if e != "O"}
    if any(e not in CHARGE for e in cats):
        return False
    pos = sum(a * CHARGE[e] for e, a in cats.items())
    return abs(pos - 2 * amt["O"]) < 1e-6


rows = []
for showcase, csv_file in SHOWCASES:
    src = pd.read_csv(os.path.join(DATA, csv_file))
    for rank, r in enumerate(src.itertuples(index=False), start=1):
        comp = Composition(r.composition)
        fu = int(round(comp.num_atoms))            # atoms per reduced formula unit
        bal = charge_balanced(comp)
        rows.append({
            "showcase":            showcase,
            "rank":                rank,
            "composition":         comp.reduced_formula,
            "n_atoms_formula_unit": fu,
            "n_elements":          len(comp.elements),
            "charge_balanced":     bal,
            "simple_phase":        bal and fu <= SMALL_FU_ATOMS,
            "pred_modulus_GPa":    r.pred_modulus_GPa,
            "pred_hardness_GPa":   r.pred_hardness_GPa,
            "pred_density_gcc":    r.pred_density_gcc,
            "fitness":             r.fitness,
        })

out = pd.DataFrame(rows)
out_path = os.path.join(DATA, "synthesizable_candidates.csv")
out.to_csv(out_path, index=False)

n_bal = int(out["charge_balanced"].sum())
n_simple = int(out["simple_phase"].sum())
print(f"Synthesizability confirmation: {n_bal}/{len(out)} candidates are "
      f"charge-balanced integer oxides; {n_simple}/{len(out)} have a small "
      f"formula unit (<= {SMALL_FU_ATOMS} atoms).")
print(f"Formula-unit atom count: min {out['n_atoms_formula_unit'].min()}, "
      f"median {int(out['n_atoms_formula_unit'].median())}, "
      f"max {out['n_atoms_formula_unit'].max()}.\n")
for sc in ("enamel", "dentin", "implant"):
    sub = out[out["showcase"] == sc]
    print(f"[{sc}]  fu atoms {sub['n_atoms_formula_unit'].min()}-"
          f"{sub['n_atoms_formula_unit'].max()}; "
          f"{int(sub['simple_phase'].sum())}/{len(sub)} simple phase")
    print(sub[["rank", "composition", "n_atoms_formula_unit",
               "charge_balanced", "simple_phase"]].to_string(index=False))
    print()
print(f"Saved -> {out_path}")
