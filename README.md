# Inverse design of bioceramics by machine learning

Jwearia Hoque Nowrin, Saifuddin Zafar, Md Habibur Rahman (imd.habiburrahman@gmail.com)

Code and data for the paper. Gradient-boosting and random-forest models trained on 1,721
oxygen-bearing Materials Project compounds predict Young's modulus, hardness and volume per
atom, and a genetic algorithm uses them to search oxide formulas for enamel, dentin and
Ti-6Al-4V implant targets. The workflow also runs in the browser at
https://material-hub.github.io/bioceramics.

## Main numbers

- Modulus error under cross-validation grouped by chemical system, with the structural inputs
  predicted as in the search: 38.5 GPa for the enamel and implant model and 49.7 GPa for the
  dentin model.
- Lowest objective values: 0.0899 for enamel, 0.7962 for dentin and 0.0064 for the implant
  target. A random search with the same budget did better in 2 of 10 campaigns for enamel, in
  all 10 for the implant target before rounding, and in none for dentin.
- The phase mixtures listed for each candidate are the competing phases on each force field's
  hull, and their moduli are mixing-model estimates.

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r 05_config_envs/requirements.txt

python 02_scripts/pipeline/train.py                # enamel and implant models
python 02_scripts/pipeline/inverse_ga.py           # enamel
python 02_scripts/pipeline/inverse_ga_implant.py   # implant
python 02_scripts/pipeline/train_dentin.py         # dentin models
python 02_scripts/pipeline/inverse_ga_dentin.py    # dentin
python 02_scripts/analysis/revision_validation.py  # grouped validation and intervals
python 02_scripts/analysis/revision_referee2.py
python 02_scripts/analysis/revision_audit_v2.py    # nested and low-modulus coverage
```

Scripts find the repository root through `02_scripts/paths.py`. Scripts that query the
Materials Project read an API key from `01_data/mp_key.txt`, which you add yourself.
`MANIFEST.md` names the script and data file behind every table and figure.

## Force-field checkpoints

CHGNet v0.3.0, MACE-MP medium MPA-0, MatterSim v1.0.0-5M, SevenNet
`SevenNetCalculator("7net-omni-i12", modal="mpa")`, and M3GNet
`materialyze/M3GNet-PES-MatPES-PBE-2025.2` at Hugging Face revision
`8414b23e68b173d6e36271e4366ee3717364051c` (matgl 3.0.5,
`PESCalculator(..., stress_unit="eV/A3")`). Pin this M3GNet revision, since the model now
published under that name is a later retraining.

## License

MIT. If you use the code or data, please cite the paper (see `CITATION.cff`).
