# Inverse Design of Bioceramics by Machine Learning — Code & Data

Reproducibility package (**code + output data**) for the machine-learning inverse
design of dental/biomedical ceramics. A gradient-boosting surrogate trained on
Materials Project oxides maps composition → mechanical properties (Young's modulus
*E*, Vickers hardness *H*, density *ρ*); a genetic algorithm then *inverts* that
surrogate to propose charge-balanced oxide compositions matching a prescribed
property fingerprint. A Materials Project convex-hull check and several no-DFT
robustness analyses close the loop.

> **Authors:** Jwerai Hoque Nowrin · Saifuddin Zafar · Md Habibur Rahman
> **Contact:** rahma103@purdue.edu
> These are **computational hypotheses**, not validated dental materials. The
> manuscript is published separately; this repository provides the code and data.

---

## The four showcases

| # | Target | Fingerprint | Outcome |
|---|--------|-------------|---------|
| **1 — Enamel** | tooth enamel | *E* ≈ 85 GPa, *H* ≈ 4 GPa, *ρ* ≈ 3.0 g/cm³ | ✅ Sr–Ca–Mg–Si–P phospho-silicate family, e.g. SrCaSiP₂O₉ |
| **2 — Dentin** | tooth dentin | *E* ≈ 20 GPa, *H* ≈ 0.7 GPa, *ρ* ≈ 2.1 g/cm³ | ❌ not reachable by single-phase oxides (~2× too stiff) — a hedged negative result |
| **3 — Implant interface** | Ti-6Al-4V stiffness match | *E* ≈ 110 GPa, *ρ* ≈ 4.4 g/cm³ | ✅ La/Ce–Sr–P phospho-oxide family, e.g. La₂CeMg₂(PO₆)₂ |
| **4 — Stability check** | all 45 GA candidates | MP convex hull | triage: all in well-studied chemsys; each decomposes into a 2–5-phase mixture |

The same pipeline is reused for all three design cases — **only the training-data
slice, the genetic-algorithm cation pool, and the property target change.**

---

## Repository structure

```
.
├── 01_data/                 input data + trained models + GA outputs
│   ├── mp_data.csv              MP oxide training table (N=1,721)
│   ├── dentin_data.csv          biocompatible-oxide subset (N=531)
│   ├── shortlist.csv            Zr-O virtual-screening shortlist
│   ├── designed_compositions.csv          GA candidates — enamel
│   ├── designed_dentin_compositions.csv   GA candidates — dentin
│   ├── designed_implant_compositions.csv  GA candidates — implant
│   ├── stability_check.csv      MP convex-hull results (all 45 candidates)
│   ├── synthesizable_candidates.csv       charge-balance / formula audit
│   ├── uncertainty_intervals.csv          conformal prediction intervals
│   ├── pareto_*.csv             NSGA-II Pareto fronts
│   ├── enumeration_modulus.csv  exhaustive charge-balanced enumeration
│   └── model_*.joblib           6 trained surrogates (enamel + dentin × 3)
├── 02_scripts/
│   ├── paths.py                 shared path constants
│   ├── features.py              composition featurizer (Magpie + structural)
│   ├── pipeline/                pull data → train → screen → GA inverse design
│   ├── figure_scripts/          figure-generation code
│   └── analysis/                stability check, synthesizability, uncertainty, enumeration
└── 05_config_envs/
    └── requirements.txt         pinned Python environment
```

> A Materials Project API key is **required only** for the data-retrieval and
> stability-check steps and is **not** included here. Provide your own
> ([Materials Project](https://next-gen.materialsproject.org/api)) via the
> `MP_API_KEY` environment variable or `01_data/mp_key.txt`. All downstream
> training, inverse design, and analysis run from the provided CSVs with no
> network access.

---

## Installation

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r 05_config_envs/requirements.txt
```

Key dependencies: `pymatgen`, `mp-api`, `matminer`, `scikit-learn`, `pymoo`,
`pandas`, `matplotlib`.

---

## Reproducing the results

Run from the repository root.

```bash
# Showcase 1 — Enamel
python 02_scripts/pipeline/pull_data.py            # MP oxides     → 01_data/mp_data.csv
python 02_scripts/pipeline/train.py                # 3 surrogates  → 01_data/model_*.joblib
python 02_scripts/pipeline/screen.py               # Zr-O screen   → 01_data/shortlist.csv
python 02_scripts/pipeline/inverse_ga.py           # GA design     → 01_data/designed_compositions.csv

# Showcase 2 — Dentin
python 02_scripts/pipeline/pull_data_dentin.py
python 02_scripts/pipeline/train_dentin.py
python 02_scripts/pipeline/inverse_ga_dentin.py

# Showcase 3 — Implant interface (reuses Showcase 1 surrogates)
python 02_scripts/pipeline/inverse_ga_implant.py

# Showcase 4 — Convex-hull stability check (all 45 candidates)
python 02_scripts/analysis/stability_check.py

# No-DFT robustness analyses
python 02_scripts/analysis/uncertainty_domain.py   # conformal prediction intervals
python 02_scripts/analysis/enumerate_space.py      # exhaustive enumeration
python 02_scripts/pipeline/inverse_pareto.py       # NSGA-II Pareto fronts
```

The full per-candidate results for **all 45 GA compositions** (predictions, hull
energies, decompositions, synthesizability flags, conformal intervals) are in the
CSVs under `01_data/` — primarily `stability_check.csv`,
`synthesizable_candidates.csv`, and `uncertainty_intervals.csv`.

---

## Headline results

- **Surrogate models:** Young's modulus CV *R²* = 0.74 (MAE ≈ 24 GPa);
  volume-per-atom *R²* = 0.78; Vickers hardness *R²* ≈ 0.39 (weak — down-weighted in the GA).
- **Enamel:** Sr–Ca–Mg–Si–P phospho-silicate family, predicted *E* 96–105 GPa (best fitness 0.145).
- **Dentin:** no single-phase oxide reaches 20 GPa (best ~34–41 GPa) — corroborated by
  exhaustive enumeration and weight-free Pareto fronts.
- **Implant:** La/Ce–Sr–P phospho-oxide family, predicted *E* 102–117 GPa.
- **Stability triage:** all 45 candidates in well-studied chemsys; each decomposes into a
  2–5-phase mixture (mean ~3) — a prioritised short list for downstream DFT/MLFF validation.

---

## Honest scope

- Models predict MP-computed **elastic** properties — **not** flexural strength, fracture
  toughness, fatigue, or wear (microstructure-governed; absent from MP).
- The GA does not enforce thermodynamic stability; Showcase 4 is a composition-only
  convex-hull *triage*, not DFT/MLFF formation-energy validation.
- Hardness and modulus are both algebraic functions of the same DFT *K*, *G* — partially
  collinear, not independent constraints.
- Model error (MAE ≈ 24 GPa) exceeds the spread among top candidates, so fitness values
  indicate convergence, not a meaningful ranking among shortlisted compositions.

---

## License

Released under the MIT License (see `LICENSE`). If you reuse the data, please also cite
the accompanying manuscript. A `CITATION.cff` is provided for GitHub's "Cite this
repository" feature.
