# NRT_LAND

NRT_LAND is a neural emulator for monthly land carbon fluxes. It trains a
spatial deep-learning model from gridded meteorological variables, dynamic
vegetation information, and a target land-flux field, then predicts future
months recursively by feeding each predicted target back as history for the
next month.

This repository contains the model, the training and evaluation code and
their launchers, and the code used for the paper: the emulator runs
([Reproducing the paper](#reproducing-the-paper)), the analysis behind the
figures and tables (`analysis/`) and the construction of the input tables
(`preprocessing/`). It does **not** contain data, checkpoints or results.

## Contents

- [How the pipeline works](#how-the-pipeline-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Prepare the data](#prepare-the-data)
- [Data format](#data-format)
- [Quick start](#quick-start)
- [Training](#training)
- [One-step prediction](#one-step-prediction)
- [Recursive evaluation](#recursive-evaluation)
- [Reproducing the paper](#reproducing-the-paper)
- [Outputs and checkpoint layout](#outputs-and-checkpoint-layout)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

## How the pipeline works

Each training sample is an `18 x 18` spatial patch with a temporal window.
The model receives:

1. four coordinate/time fields: `year`, `month`, `latitude`, and `longitude`;
2. nine meteorological fields: `sp`, `ssrd`, `strd`, `t2m`, `tp`, `u10`, `v10`,
   `q`, and `co2`;
3. the target's previous value as a history channel; and
4. optional columns whose names contain `PFT`.

During recursive evaluation, the target values in the requested prediction
period are replaced month by month with the model prediction. The next month
therefore uses the preceding prediction as history. Target observations from
the prediction period are not used as input history.

The evaluators use a CNN with a 13-month input window and 18 x 18 patches and
load the five fold checkpoints of the selected experiment; the prediction is the
mean of the five folds.

Note on the learning rate: `run_epoch()` never calls `scheduler.step()`, so the
learning rate stays at the start value of the warm-up scheduler,
0.01 x 3.2e-3 = 3.2e-5 (AdamW, weight decay 1e-3), for the whole training. The
published checkpoints were trained this way, and the code is kept unchanged so
that retraining reproduces them.

## Requirements

- Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation
- Python 3.10 (recommended)
- PyTorch 2.3.0 with CUDA 12.1, or a compatible PyTorch build
- `torchrun` available on `PATH`
- Enough memory for the full gridded Feather table and one training batch

The entry points initialize the NCCL distributed backend and are intended for
GPU execution. CPU-only execution is not supported by the supplied launchers.

## Installation

```bash
git clone https://github.com/agiamason/NRT_LAND.git
cd NRT_LAND

conda create -n nrt_land python=3.10 -y
conda activate nrt_land

# Installs PyTorch 2.3.0 (CUDA 12.1) and the Python dependencies.
bash install_dep_pkgs.sh
```

If PyTorch is already installed, install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

Check the environment before starting a long run:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
```

## Prepare the data

The repository expects **preprocessed Feather files**. Raw NetCDF, CSV, and
PFT source files must be converted before they are passed to the launchers;
see `preprocessing/` for the scripts that build them.

Use this directory layout (the names can be changed through the launcher
variables described below):

```text
NRT_LAND/
├── data/
│   ├── data25_0616.feather       # --data_name data25_0616
│   └── PFT/
│       ├── ORCHIDEE_nbp.feather
│       ├── CABLE-POP_nbp.feather
│       └── ...
├── checkpoint/
├── normalizers/
└── result/
```

`DATA_NAME=data25_0616` means that the loader reads
`$DATA_DIR/data25_0616.feather`; it is a file stem, not a path.

## Data format

### Main Feather table

The main file is a long table with **one row per `(latitude, longitude, year,
month)`**. It must contain these columns:

| Column | Type | Required content |
| --- | --- | --- |
| `year` | integer | Calendar year; the loader keeps years from 2000 onward |
| `month` | integer | Month number from 1 to 12 |
| `latitude` | integer | Grid-row index from 0 to 359 for the 0.5° global grid |
| `longitude` | integer | Grid-column index from 0 to 719 for the 0.5° global grid |
| `<TARGET>` | float | The target column, for example `ORCHIDEE_nbp` |
| `sp` | float | Surface pressure |
| `ssrd` | float | Surface shortwave radiation |
| `strd` | float | Surface thermal radiation |
| `t2m` | float | 2-m air temperature |
| `tp` | float | Total precipitation |
| `u10` | float | 10-m eastward wind |
| `v10` | float | 10-m northward wind |
| `q` | float | Specific humidity |
| `co2` | float | Atmospheric CO₂ input |

`<TARGET>` is replaced by the value passed through `--target`. For example,
when `TARGET=ORCHIDEE_nbp`, the table must contain a column named
`ORCHIDEE_nbp`.

The coordinate fields are used as embedding indices by the models. Convert
physical latitude/longitude values to the integer grid indices expected by the
code before writing the Feather file. Keep one complete, regular time series
for every grid cell and sort rows by:

```text
latitude, longitude, year, month
```

The loader sorts the main table again, but the original order still matters
when the PFT table is attached. Missing meteorological and PFT values are
filled with zero during preprocessing. Missing target values are retained so
that the training loss can mask them.

A minimal schema check is:

```python
import pandas as pd

target = "ORCHIDEE_nbp"
required = {
    "year", "month", "latitude", "longitude", target,
    "sp", "ssrd", "strd", "t2m", "tp", "u10", "v10", "q", "co2",
}

df = pd.read_feather("data/data25_0616.feather")
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"missing columns: {sorted(missing)}")
if not df[["year", "month", "latitude", "longitude"]].notna().all().all():
    raise ValueError("coordinate/time columns must not contain NaN")
print(df.shape)
print(df[["year", "month", "latitude", "longitude"]].head())
```

### PFT Feather table

If PFT features are used, create one file per target at
`$PFT_DIR/<TARGET>.feather`. The file must:

- have one row per row of the main table with `year >= 2000`, in the same
  order, with the same `year`, `month`, `latitude` and `longitude` values (the
  loader checks this; extra rows at the end are dropped);
- contain feature columns whose names include `PFT` (for example `PFT_1`,
  `PFT_2`, ...).

PFT values may be static or repeated for each month. A static 2-D map must be
expanded/repeated to the main table row count before it is saved. The loader
copies all columns containing `PFT` into the main table. A missing PFT file is allowed by
the loader, but a checkpoint trained with PFT features will not be compatible
with a run that omits them.

### Missing-data mask of the predictions

The recursive evaluator sets predictions to NaN where the target column of the
main table is missing in the same month (only the missingness is read, never the
target values). In a prediction year without target values, pass
`--mask_year` (`MASK_YEAR`): months after that year use the mask of the same
month of that year. `--pft_freeze_year` (`PFT_FREEZE_YEAR`) holds the PFT
fractions of later years at that year.

### Units and normalization

The code does not convert physical units. Use the same units and variable
definitions used to train the checkpoint. Continuous target and meteorological
columns are scaled with `RobustScaler`; the fitted normalizers are saved under
`normalizers/` during evaluation and must be kept with the corresponding data
version and experiment.

## Quick start

After placing a compatible Feather table, PFT file, and completed checkpoint
under the layout above, run a one-GPU recursive evaluation:

```bash
GPUS=1 \
TARGET=ORCHIDEE_nbp \
EXP_ID=101 \
DATA_NAME=data25_0616 \
START_YEAR=2022 START_MONTH=1 \
END_YEAR=2023 END_MONTH=12 \
bash scripts/evaluate.sh
```

The result is written to `result/101.csv` (mean of the five folds) and
`result/101_<fold>.csv`. The evaluator creates a normalizer in `normalizers/` if
one does not already exist; use the training years of the checkpoint
(`NORMALIZER_START_YEAR`, `NORMALIZER_END_YEAR`).

## Training

Train a five-fold model (13-month window and 18 x 18 patches, the launcher
defaults) on the years `KFOLD_START_YEAR`-`KFOLD_END_YEAR`:

```bash
GPUS=1 \
TARGET=ORCHIDEE_nbp \
EXP_ID=101 \
DATA_NAME=data25_0616 \
KFOLD_START_YEAR=2000 KFOLD_END_YEAR=2021 \
bash scripts/train.sh
```

For multiple GPUs, set `GPUS` to the number of visible GPUs (the paper used
four GPUs with 128 samples per GPU):

```bash
GPUS=4 TARGET=ORCHIDEE_nbp EXP_ID=101 KFOLD_END_YEAR=2021 bash scripts/train.sh
```

Each fold writes a completed checkpoint under
`checkpoint/<EXP_ID>/<fold>/finish_*.pth`. When several runs share a machine,
give each a different `MASTER_PORT`.

## One-step prediction

`scripts/onestep.sh` (`kfold_prediction_fixed_scaler.py`) predicts every month
from `START_YEAR` to `END_YEAR` (default 2010-2023) from the target history in
the main table, with the normalizer fitted on the training years
(`NORMALIZER_START_YEAR`-`NORMALIZER_END_YEAR`, default 2000-2021):

```bash
TARGET=ORCHIDEE_nbp EXP_ID=101 RESULT_DIR=result/exp1_onestep bash scripts/onestep.sh
```

Its CSVs are also the history files of the recursive evaluation (`HISTORY_DIR`).

## Recursive evaluation

The evaluator loads every `finish_*.pth` file under the selected experiment and
predicts each month in chronological order from the 13 months ending at that
month; the prediction replaces the target value, so the next month uses it as
history. Each month is appended to `<EXP_ID>.csv` and `<EXP_ID>_<fold>.csv`.
With `HISTORY_DIR`, these files start as copies of the one-step files of the
same experiment, cut just before `START_YEAR`. The launcher exposes:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TARGET` | required | Target column and PFT filename stem |
| `EXP_ID` | required | Checkpoint experiment identifier |
| `DATA_NAME` | `data25_0616` | Main Feather file stem |
| `START_YEAR` | `2022` | First prediction year |
| `START_MONTH` | `1` | First prediction month |
| `END_YEAR` | `2023` | Last prediction year |
| `END_MONTH` | `12` | Last prediction month |
| `GPUS` | `1` | Number of processes/GPU devices |
| `DATA_DIR` | `./data` | Directory containing the Feather inputs |
| `PFT_DIR` | `$DATA_DIR/PFT` | Directory containing target PFT Feather files |
| `CHECKPOINT_DIR` | `./checkpoint` | Root checkpoint directory |
| `RESULT_DIR` | `./result` | Result output directory |
| `NORMALIZER_DIR` | `./normalizers` | Saved normalizer directory |
| `NORMALIZER_MODE` | `auto` | `auto` fits a missing normalizer, `load` requires it |
| `NORMALIZER_START_YEAR`, `NORMALIZER_END_YEAR` | `2000`, `2021` | Normalizer fit years (the training years) |
| `HISTORY_DIR` | empty | One-step result directory continued by this run |
| `MASK_YEAR` | `0` (off) | Missing-data mask year for years without targets |
| `PFT_FREEZE_YEAR` | `0` (off) | Year at which the PFT fractions are held fixed |
| `MASTER_PORT` | `29500` | torchrun port; use a different one per concurrent run |

## Reproducing the paper

The paper evaluates emulators of the 19 TRENDY v13 S2 models (experiment ids
101-119, see `scripts/reproduce_paper.sh`) in two experiments:

| | Training years | One-step | Recursive | Other settings |
| --- | --- | --- | --- | --- |
| `exp1` | 2000-2021 | 2010-2023 | 2022-01 to 2023-12 | |
| `exp2` | 2000-2023 | 2010-2023 | 2024-01 to 2024-12 | `PFT_FREEZE_YEAR=2023`, `MASK_YEAR=2023` |

Inputs: `data/data25_0616.feather` (main table) and the PFT tables of the 12
models with land-cover input (`preprocessing/build_pft_feather.py`). With the
checkpoints of an experiment in `CHECKPOINT_DIR`:

```bash
# training (4 GPUs in the paper), one-step and recursive prediction
GPUS=4 CHECKPOINT_DIR=checkpoint/exp1 bash scripts/reproduce_paper.sh exp1 train onestep recursive
GPUS=4 CHECKPOINT_DIR=checkpoint/exp2 bash scripts/reproduce_paper.sh exp2 train onestep recursive
```

Without the `train` stage, the script runs the one-step and recursive
predictions from existing checkpoints. Results are written to
`result/<exp>_onestep/` and `result/<exp>_recursive/`. The figures, tables and
numbers of the paper are then produced by `analysis/` (see
[analysis/README.md](analysis/README.md)).

## Outputs and checkpoint layout

```text
checkpoint/
└── <EXP_ID>/
    ├── 0/finish_<timestamp>_<metric>.pth
    ├── 1/finish_<timestamp>_<metric>.pth
    ├── 2/finish_<timestamp>_<metric>.pth
    ├── 3/finish_<timestamp>_<metric>.pth
    └── 4/finish_<timestamp>_<metric>.pth

normalizers/
├── normalizer_land_<TARGET>_<DATA_NAME>_<start>_<end>.pkl
└── normalizer_land_<TARGET>_<DATA_NAME>_<start>_<end>.json

result/
├── <EXP_ID>.csv
└── <EXP_ID>_<fold>.csv
```

Result CSV files contain the prediction coordinates and the model output. Do
not commit checkpoints, normalizers, results, logs, or input data to the
public repository.

## Troubleshooting

### `PFT data is shorter than the selected data` or `PFT rows are not aligned`

The PFT file does not have one row per row of the main table with
`year >= 2000`, in the same order. Regenerate it from the same main table
(`preprocessing/build_pft_feather.py`).

### `Missing normalizer` or a normalizer mismatch

Use the same target, data file, and normalizer year range that were used for
the checkpoint. Delete an accidentally generated normalizer only after
confirming that it is not used by another experiment.

### No `finish_*.pth` files found

Check that `CHECKPOINT_DIR/<EXP_ID>/<fold>/` contains completed checkpoints
and that `EXP_ID` is the numeric identifier used during training.

### CUDA/NCCL startup errors

Confirm that the selected GPUs are visible (`CUDA_VISIBLE_DEVICES`), that
`torch.cuda.is_available()` is `True`, and that `GPUS` does not exceed the
number of visible devices.

## Repository layout

```text
model/                           Neural-network definitions
process_data.py                  Feather loading and sample construction
data_utils.py                    Dataset utilities kept for compatibility
pipeline.py                      Training and non-recursive evaluation entry point
kfold_prediction_fixed_scaler.py One-step prediction with a fixed normalizer
multi_prediction_evaluation.py   Recursive month-by-month evaluation
scripts/train.sh                 Training launcher
scripts/onestep.sh               One-step prediction launcher
scripts/evaluate.sh              Recursive evaluation launcher
scripts/reproduce_paper.sh       The paper's two experiments for the 19 models
preprocessing/                   Construction of the input tables
analysis/                        Figures, tables and numbers of the paper
install_dep_pkgs.sh              PyTorch and dependency installation
requirements.txt                 Python dependencies other than PyTorch
```

## License

This project is licensed under the [MIT License](LICENSE).
