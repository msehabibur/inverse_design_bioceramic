"""
Project path helper.  Every script under 02_scripts/ imports DATA, FIGURES,
MANUSCRIPT, ROOT from here so that directory moves only need to be reflected
in this one file.  Scripts live in subfolders (pipeline/, figure_scripts/,
analysis/); they add 02_scripts/ to sys.path via a short bootstrap so this
module and features.py remain importable.
"""
import os

HERE       = os.path.dirname(os.path.abspath(__file__))  # 02_scripts/
ROOT       = os.path.dirname(HERE)                       # project root
DATA       = os.path.join(ROOT, "01_data")
FIGURES    = os.path.join(ROOT, "03_figures")
MANUSCRIPT = os.path.join(ROOT, "04_manuscript")
MODELS     = os.path.join(ROOT, "06_models")        # trained surrogates
