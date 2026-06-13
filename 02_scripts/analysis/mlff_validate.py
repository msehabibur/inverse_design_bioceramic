"""
MLFF structure-aware modulus cross-check (NO DFT run by us; CHGNet is a
pretrained ML potential). For a modulus-spanning sample of real MP oxides we
relax with CHGNet and compute the CHGNet elastic moduli, then compare to the
DFT moduli already in mp_data -- a parity that validates the MLFF as an
independent, structure-aware modulus estimator. Begins with an MgO sanity check.

Reads:  data/mlff_sample_structures.json   (prefetched on the login node)
Writes: data/mlff_parity.csv, manuscript/figure_mlff.png
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA, MANUSCRIPT


def build_calc():
    from matcalc.utils import get_universal_calculator
    return get_universal_calculator("CHGNet")


EV_A3_TO_GPA = 160.21766208   # matcalc returns moduli in eV/A^3, not GPa

def relax_and_elastic(struct, calc):
    """Return (E, K, G) in GPa for a relaxed structure, via matcalc."""
    from matcalc.relaxation import RelaxCalc
    from matcalc.elasticity import ElasticityCalc
    relaxed = RelaxCalc(calc, fmax=0.05, relax_cell=True).calc(struct)["final_structure"]
    el = ElasticityCalc(calc, relax_structure=False).calc(relaxed)
    K = float(el["bulk_modulus_vrh"]) * EV_A3_TO_GPA
    G = float(el["shear_modulus_vrh"]) * EV_A3_TO_GPA
    E = 9 * K * G / (3 * K + G) if (3 * K + G) > 0 else 0.0
    return E, K, G


calc = build_calc()
print("CHGNet calculator ready")

# ---- MgO sanity ----------------------------------------------------------
mgo = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21),
                                ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
E, K, G = relax_and_elastic(mgo, calc)
print(f"SANITY MgO: CHGNet E={E:.0f} K={K:.0f} G={G:.0f} GPa  (expt E~300)")

# ---- sample of real MP oxides -------------------------------------------
data = json.load(open(os.path.join(DATA, "mlff_sample_structures.json")))
rows = []
for i, rec in enumerate(data, 1):
    s = Structure.from_dict(rec["structure"])
    try:
        E, K, G = relax_and_elastic(s, calc)
    except Exception as e:
        print(f"  [{i}] {rec['formula']} FAILED: {e}"); continue
    rows.append({"material_id": rec["material_id"], "formula": rec["formula"],
                 "dft_E_GPa": round(rec["dft_E_GPa"], 1),
                 "chgnet_E_GPa": round(float(E), 1),
                 "chgnet_K_GPa": round(float(K), 1), "chgnet_G_GPa": round(float(G), 1)})
    print(f"  [{i}] {rec['formula']:14s} DFT={rec['dft_E_GPa']:6.0f}  CHGNet={E:6.0f} GPa")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA, "mlff_parity.csv"), index=False)

# ---- parity figure -------------------------------------------------------
if len(out):
    x = out["dft_E_GPa"].values; y = out["chgnet_E_GPa"].values
    r = np.corrcoef(x, y)[0, 1] ** 2
    mae = np.mean(np.abs(x - y))
    fig, ax = plt.subplots(figsize=(6, 6))
    hi = max(x.max(), y.max()) * 1.08
    ax.plot([0, hi], [0, hi], "--", color="#555", lw=1.2)
    ax.scatter(x, y, s=60, alpha=0.8, color="#1A6090", edgecolor="white")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal", "box")
    ax.set_xlabel("DFT Young's modulus (GPa, Materials Project)")
    ax.set_ylabel("CHGNet structure-aware modulus (GPa)")
    ax.set_title(f"MLFF vs DFT modulus  (R$^2$={r:.2f}, MAE={mae:.0f} GPa, N={len(out)})", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(MANUSCRIPT, "figure_mlff.png"), dpi=200, bbox_inches="tight")
    print(f"\nMLFF vs DFT: R2={r:.2f}, MAE={mae:.0f} GPa over N={len(out)}")
    print("saved -> figure_mlff.png + data/mlff_parity.csv")
else:
    print("no successful elastic evaluations")
