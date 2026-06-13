"""
Showcase 2 - Step 1.  Build the dentin-relevant slice of Materials Project.

Dentin's mechanical fingerprint (E ~ 18-20 GPa, H ~ 0.5-1 GPa,
rho ~ 2.0-2.1 g/cm^3) is dominated by hydroxyapatite + a collagen matrix,
i.e. the calcium-phosphate / bioactive-glass family rather than the stiff
zirconia-family ceramics used for the enamel showcase.

We do NOT re-hit the MP API: we filter the already-pulled mp_data.csv to a
broad biocompatible-oxide chemical space relevant to dental ceramics, glass-
ceramics, and bioactive glasses.  The element set covers:

  - alkali / alkaline earths:                   Li, Na, K, Mg, Ca, Sr, Ba
  - main-group glass / apatite formers:         B, Al, Si, P, F
  - common dental structural cations:           Ti, Zr, Y, La, Ce, Hf
  - bioactive transition-metal dopants:         Zn, Cu, Fe, Mn
  - plus O and H (hydroxyapatite-type phases)

A material is kept iff every constituent element is in this set; we no longer
require the presence of Ca or P, because bioactive borosilicate and alkali-
silicate glasses are also dentin-relevant and useful training material.

Output: dentin_data.csv
"""
import os
import pandas as pd
from pymatgen.core import Composition
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA

ALLOWED = {
    "O", "H",
    "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba",
    "B", "Al", "Si", "P", "F",
    "Ti", "Zr", "Y", "La", "Ce", "Hf",
    "Zn", "Cu", "Fe", "Mn",
}
MUST_HAVE_ONE_OF = set()                 # no longer required

print("Loading mp_data.csv ...")
df = pd.read_csv(os.path.join(DATA, "mp_data.csv"))
print(f"  full oxide pool: {len(df)} materials")


def in_bone_family(formula):
    try:
        els = {str(e) for e in Composition(formula).elements}
    except Exception:
        return False
    if not els.issubset(ALLOWED):
        return False
    if MUST_HAVE_ONE_OF and not (els & MUST_HAVE_ONE_OF):
        return False
    return True


keep = df["formula"].apply(in_bone_family)
dentin = df[keep].reset_index(drop=True)

print(f"\nFiltered to bone-family compositions:")
print(f"  N = {len(dentin)} materials")
print(f"  modulus range:   {dentin['youngs_modulus_GPa'].min():.0f} - "
      f"{dentin['youngs_modulus_GPa'].max():.0f} GPa")
print(f"  density range:   {dentin['density'].min():.2f} - "
      f"{dentin['density'].max():.2f} g/cm^3")
print(f"  stable fraction: {(dentin['energy_above_hull'] < 0.05).mean():.1%}")

out = os.path.join(DATA, "dentin_data.csv")
dentin.to_csv(out, index=False)
print(f"\nSaved -> {out}")
print(dentin[["formula", "youngs_modulus_GPa", "density",
              "energy_above_hull"]].head(10))
