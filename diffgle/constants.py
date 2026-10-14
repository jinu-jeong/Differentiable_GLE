"""Units and paths.

Units follow torchmd: length in Angstrom, mass in amu, energy in kcal/mol.
Internal time unit = TIMEFACTOR fs, so a 1 fs step is dt = 1 / TIMEFACTOR.
"""

import os

TIMEFACTOR = 48.88821   # fs per internal time unit
BOLTZMAN = 0.001987191  # kcal/mol/K
TEMPERATURE = 300.0     # K, all systems
DT = 1 / TIMEFACTOR     # 1 fs time step in internal units

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "Result")
