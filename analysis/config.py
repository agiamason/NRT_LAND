"""Input and output locations of the paper analysis.

Each path can be overridden with an environment variable of the same name.
Directory paths end with '/' because the analysis code appends file names.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(name, default):
    return os.environ.get(name, default)


def _dir(name, default):
    path = os.environ.get(name, default)
    return path if path.endswith('/') else path + '/'


# TRENDY S2 monthly NBP of all models in one file (dimensions DGVM, time, lat, lon; 0.5 deg; kg m-2 s-1)
TRENDY_V13_NBP = _path('TRENDY_V13_NBP', os.path.join(ROOT, 'data', 'analysis', 'Trendyv13_S2_nbp.nc'))
TRENDY_V14_NBP = _path('TRENDY_V14_NBP', os.path.join(ROOT, 'data', 'analysis', 'Trendyv14_S2_nbp.nc'))

# Emulator output written by scripts/reproduce_paper.sh (one CSV per model, <exp_id>.csv)
EMULATOR_EXP1_DIR = _dir('EMULATOR_EXP1_DIR', os.path.join(ROOT, 'result', 'exp1_recursive'))
EMULATOR_EXP1_ONESTEP_DIR = _dir('EMULATOR_EXP1_ONESTEP_DIR', os.path.join(ROOT, 'result', 'exp1_onestep'))
EMULATOR_EXP2_DIR = _dir('EMULATOR_EXP2_DIR', os.path.join(ROOT, 'result', 'exp2_recursive'))

# RECCAP2 land regions on the 0.5 deg grid (variable 'mask', ids 1-25)
RECCAP_MASK = _path('RECCAP_MASK', os.path.join(ROOT, 'data', 'analysis', 'Land_reccap_regions_mask_05x05.nc'))

# the 36 excluded cells of western Venezuela, made by analysis/event_mask/make_event_mask.py
EVENT_MASK = _path('EVENT_MASK', os.path.join(ROOT, 'data', 'analysis', 'crujra_v2.4_2021_precip_event_mask.nc'))

# ERA5 monthly t2m and tp on the 0.5 deg grid (for the ERA5 regression baseline)
ERA5_FILE = _path('ERA5_FILE', os.path.join(ROOT, 'data', 'analysis', 'ERA5.nc'))

# Emulator input table (the Feather file used for training; ERA5 columns are read)
DATA_FEATHER = _path('DATA_FEATHER', os.path.join(ROOT, 'data', 'data25_0616.feather'))

# All figures, tables and text outputs
OUT_DIR = _dir('OUT_DIR', os.path.join(ROOT, 'output'))

NAMES = ['TRENDY_V13_NBP', 'TRENDY_V14_NBP', 'EMULATOR_EXP1_DIR', 'EMULATOR_EXP1_ONESTEP_DIR', 'EMULATOR_EXP2_DIR',
         'RECCAP_MASK', 'EVENT_MASK', 'ERA5_FILE', 'DATA_FEATHER', 'OUT_DIR']
