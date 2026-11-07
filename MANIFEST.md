# Manifest

The script and data file behind every table and figure of the paper (tag `v2.2-revision`).
Paths are relative to the repository root. Use the environment in
`05_config_envs/requirements.txt` (scikit-learn 1.6.1). The validation reads its grouped
folds from `01_data/fold_assignment.csv`, so it gives the same folds under any NumPy version.

## Models

| Pool | Young's modulus | Hardness | Volume per atom | Script |
|---|---|---|---|---|
| enamel and implant (1,721 compounds, `01_data/mp_data.csv`) | histogram gradient boosting | random forest, 300 trees | histogram gradient boosting | `02_scripts/pipeline/train.py` |
| dentin (531 compounds, `01_data/dentin_data.csv`) | histogram gradient boosting | random forest | random forest | `02_scripts/pipeline/train_dentin.py` |

The models are `01_data/model_{modulus,hardness,volperatom}.joblib` and the `_dentin`
versions. `02_scripts/chain.py` refits this chain inside each fold for validation.

## Tables

| Table | Data file(s) | Script |
|---|---|---|
| I and S1, enamel candidates | `01_data/designed_compositions.csv` | `pipeline/inverse_ga.py` |
| II and S2, dentin candidates | `01_data/designed_dentin_compositions.csv` | `pipeline/inverse_ga_dentin.py` |
| III and S3, implant candidates | `01_data/designed_implant_compositions.csv` | `pipeline/inverse_ga_implant.py` |
| S4, charge balance and formula size | `01_data/synthesizable_candidates.csv` | `analysis/synthesizability_filter.py` |
| S5, prediction intervals and domain distances | `01_data/revision_uncertainty.csv`, `01_data/revision_domain.csv` | `analysis/revision_validation.py` |
| S6, the 142 composition descriptors | (descriptor list) | `02_scripts/features.py` |
| S7, enamel and implant training compounds below 20 GPa (full pool in the data file) | `01_data/mp_data.csv` | `pipeline/pull_data.py` |
| S8, dentin training compounds below 40 GPa (full pool in the data file) | `01_data/dentin_data.csv` | `pipeline/pull_data_dentin.py` |
| S9, force-field energies above the hull | `06_mlff_stability/stability_mlff.csv`, `06_mlff_stability/stability_mlff_long.csv` | `mlff_stability/build_hull.py`, then `analysis/fix_consensus_median.py` |
| S10, all validation protocols | `01_data/revision_validation.csv`, `01_data/revision_metric_uncertainty.csv` | `analysis/revision_validation.py` |
| S11, modulus scores under different folds and inputs | `01_data/revision_r2_protocols.csv`, `01_data/revision_validation.csv` | `analysis/revision_referee2.py`, `analysis/revision_validation.py` |
| S12, property targets and sources | (literature values, no data file) | |
| S13, mixture moduli | `01_data/revision_mixture_modulus.csv`, `01_data/revision_phase_moduli.csv` | `analysis/revision_mixture_and_chemistry.py` |
| S14, aqueous classification of the mixture phases | `01_data/revision_mixture_modulus.csv` | `analysis/revision_mixture_and_chemistry.py` |
| S15, structure sampling and convergence | `01_data/revision_sampling.csv`, `01_data/revision_sampling_summary.csv` | `analysis/revision_sampling_and_holdout.py` |
| S16, objective-weight sweep | `01_data/audit_weight_sweep_full.csv` | `analysis/revision_sweep_and_front.py` |
| S17, enumerated composition sets | `01_data/revision_enumeration.csv` | `analysis/revision_search_baseline.py` |
| S18, random-search control | `01_data/revision_search_baseline.csv` | `analysis/revision_search_baseline.py` |
| S19, coverage at low modulus | `01_data/audit_low_modulus_recall.csv`, `01_data/audit_nested_coverage.csv` | `analysis/revision_audit_v2.py` |
| S20, earlier virtual screen | `01_data/revision_shortlist_audit.csv` | `analysis/revision_mixture_and_chemistry.py` |

## Figures

| Figure | Data file(s) | Script |
|---|---|---|
| 1, the three design targets | `01_data/designed_*compositions.csv` | `figure_scripts/revision_schematics.py` |
| 2 to 4, enamel, dentin and implant panels | `01_data/revision_parity_oof.csv`, `01_data/revision_ga_history.csv`, `01_data/revision_ga_population.csv`, `01_data/revision_enumeration_full.csv`, `01_data/revision_enumeration.csv`, `01_data/revision_uncertainty.csv` | `figure_scripts/revision_case_panels.py` |
| S1, effect of the validation protocol | `01_data/revision_validation.csv`, `01_data/revision_uncertainty.csv`, `01_data/uncertainty_intervals.csv` | `figure_scripts/revision_figures.py` |
| S2, random search and enumeration | `01_data/revision_search_baseline.csv`, `01_data/revision_enumeration_full.csv` | `figure_scripts/revision_figures.py` |
| S3, mixture moduli | `01_data/revision_mixture_modulus.csv`, `01_data/revision_phase_moduli.csv` | `figure_scripts/revision_figures.py` |
| S4, force-field screen | `06_mlff_stability/stability_mlff.csv` | `figure_scripts/revision_stability_panels.py` |
| S5, Pareto fronts and weight sweep | `01_data/revision_pareto_exact.csv`, `01_data/audit_weight_sweep_full.csv` | `analysis/revision_sweep_and_front.py` |
| S6, prediction performance across the modulus range | `01_data/audit_oof_enamel_implant.csv`, `01_data/audit_oof_dentin.csv`, `01_data/revision_enumeration_full.csv`, `01_data/revision_validation.csv` | `figure_scripts/revision_figures.py` |

Figure scripts write to `04_manuscript/revision_figures/`.

## Rerunning the validation

```bash
python 02_scripts/analysis/revision_validation.py   # revision_validation, _metric_uncertainty, _parity_oof, _uncertainty, _novelty, _domain
python 02_scripts/analysis/revision_referee2.py     # revision_r2_protocols, _weight_sensitivity, _indistinguishable
python 02_scripts/analysis/revision_audit_v2.py     # audit_* files
```

## Force-field screen

`06_mlff_stability/` holds the trial structures (`structures/`), the relaxation results of
each force field (`results/`), the manifests and the energies above the hull.
`mlff_stability/build_hull.py` rebuilds the hulls, and `analysis/fix_consensus_median.py`
then drops the failed negative energies from the consensus median (Table S9). Rerunning the
relaxations (`mlff_stability/relax_worker.py`) needs CHGNet v0.3.0, MACE-MP medium MPA-0,
MatterSim v1.0.0-5M, M3GNet-PES-MatPES-PBE-2025.2 at Hugging Face revision 8414b23
(matgl 3.0.5) and SevenNet 7net-omni-i12 (mpa head); set `MLFF_ZOO` to a directory that
holds them. `mlff_stability/checkpoint_check.py` compares recorded energies with a given
checkpoint (outputs `06_mlff_stability/checkpoint_check_*.csv`).

## Materials Project API key

`pipeline/pull_data.py`, `pipeline/screen.py`, `pipeline/prefetch_structures.py`,
`analysis/stability_check.py`, `analysis/revision_mixture_and_chemistry.py`,
`analysis/revision_sampling_and_holdout.py` and `mlff_stability/gen_competitors.py` read the
key from `01_data/mp_key.txt`; the first four also read `MP_API_KEY`.
