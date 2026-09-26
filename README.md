# Inverse design of bioceramics by machine learning

Jwearia Hoque Nowrin, Saifuddin Zafar, Md Habibur Rahman (rahma103@purdue.edu)

Code and data for the paper. Gradient-boosting models trained on 1,721 oxygen-bearing
Materials Project compounds predict Young's modulus, hardness and volume per atom. A genetic
algorithm uses them to search charge-balanced oxide formulas for three targets. The workflow
also runs in the browser at https://material-hub.github.io/bioceramics.

## Results

| Target | Goal | Result |
|---|---|---|
| Enamel | E 85 GPa, density 3.0 g/cm³ | Sr-Ca-Mg-Si-P phosphosilicates, predicted E 96 to 105 GPa |
| Dentin | E 20 GPa, density 2.1 g/cm³ | Predicted E 44 to 51 GPa. Enumerating 17,798 formulas lowers the minimum to 39.6 GPa, and its interval includes 20 GPa |
| Implant (Ti-6Al-4V) | E 110 GPa, density 4.4 g/cm³ | La/Ce-Sr-P phospho-oxides, predicted E 102 to 117 GPa |

The modulus error under cross-validation grouped by chemical system is 38.1 GPa for the
enamel and implant model and 50.4 GPa for the dentin model.

## Limitations

- No training compound contains all the elements of any candidate, so every prediction is an extrapolation.
- The model is least reliable below about 40 GPa, where the dentin target lies.
- Most trial relaxations in the five-force-field screen did not converge, so stability is not established.
- The labels are computed elastic properties of ideal crystals. Porosity, processing, strength and biological response are not included.

## Layout

```
01_data/                    training tables, trained models, candidates, analysis outputs
02_scripts/pipeline/        data retrieval, training, screening, genetic algorithm
02_scripts/analysis/        stability, uncertainty, enumeration, revision analyses
02_scripts/figure_scripts/  figures
05_config_envs/             requirements.txt
```

## Run

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r 05_config_envs/requirements.txt

python 02_scripts/pipeline/train.py               # enamel and implant models
python 02_scripts/pipeline/inverse_ga.py          # enamel
python 02_scripts/pipeline/inverse_ga_implant.py  # implant
python 02_scripts/pipeline/train_dentin.py        # dentin models
python 02_scripts/pipeline/inverse_ga_dentin.py   # dentin
```

Everything runs from the CSV files in `01_data/`. Only `pull_data*.py` and
`stability_check.py` need a Materials Project API key, set as `MP_API_KEY` or saved in
`01_data/mp_key.txt` (not included). The scripts in `02_scripts/analysis/` regenerate the
analysis tables in `01_data/`.

## License

MIT. If you use the code or data, please cite the paper (see `CITATION.cff`).
