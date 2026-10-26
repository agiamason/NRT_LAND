# NRT_LAND

NRT_LAND is a neural emulator for monthly land carbon fluxes. It trains a
spatial deep-learning model from gridded meteorological variables, dynamic
vegetation information, and a target land-flux field, then predicts future
months recursively by feeding each predicted target back as history for the
next month.

This repository is the minimal runnable package for training and recursive
evaluation. It contains the model definitions, data loader, training and
evaluation entry points, and example launchers. It does **not** contain raw
data, PFT files, checkpoints, generated results, notebooks, or credentials.

## Contents

- [How the pipeline works](#how-the-pipeline-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Prepare the data](#prepare-the-data)
- [Data format](#data-format)
- [Quick start](#quick-start)
- [Training](#training)
- [Recursive evaluation](#recursive-evaluation)
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

The current recursive evaluator uses a CNN with a 13-month input window and
loads all completed fold checkpoints for the selected experiment.

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
the conversion scripts are intentionally kept outside this minimal public
package because they are specific to each data release.

Use this directory layout (the names can be changed through the launcher
variables described below):

```text
NRT_LAND/
├── data/
│   ├── data25_0616.feather       # --data_name data25_0616
│   ├── Trendyv14_S2_nbp.nc       # DGVM masks for recursive evaluation
│   ├── average_nbp.nc            # optional average target mask
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

- have exactly the same number of rows as the main Feather table;
- use the same row order as the main table before preprocessing; and
- contain feature columns whose names include `PFT` (for example `PFT_1`,
  `PFT_2`, ...).

PFT values may be static or repeated for each month. A static 2-D map must be
expanded/repeated to the main table row count before it is saved. The loader
copies all columns containing `PFT` into the main table. A missing PFT file is allowed by
the loader, but a checkpoint trained with PFT features will not be compatible
with a run that omits them.

### Evaluation NetCDF files

Recursive evaluation also reads the DGVM reference file supplied by
`--dgvm_path` (default `data/Trendyv14_S2_nbp.nc`). It expects an `nbp`
data variable and a `DGVM` coordinate/value list containing the selected target
name. For the special target `average_nbp`, the evaluator uses
`--dgvm_average_path` (default `data/average_nbp.nc`). These files are used to
identify invalid target cells and are separate from the Feather training table.

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

The final result is written to `result/101.csv`. The evaluator creates a
normalizer in `normalizers/` if one does not already exist. The first run can
be slow because every requested month is evaluated sequentially.

## Training

Train a five-fold model with the 13-month context expected by the recursive
evaluator:

```bash
GPUS=1 \
TARGET=ORCHIDEE_nbp \
EXP_ID=101 \
DATA_NAME=data25_0616 \
MODEL=cnn WINDOW=13 \
bash scripts/train.sh
```

For multiple GPUs, set `GPUS` to the number of visible GPUs:

```bash
GPUS=4 TARGET=ORCHIDEE_nbp EXP_ID=101 WINDOW=13 bash scripts/train.sh
```

Training uses five folds by default. Each fold writes a completed checkpoint
under `checkpoint/<EXP_ID>/<fold>/finish_*.pth`.

## Recursive evaluation

The supplied evaluator loads every `finish_*.pth` file under the selected
experiment, predicts each month in chronological order, and writes the final
ensemble result at the last requested month. The standard launcher exposes:

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
| `DATA_DIR` | `./data` | Directory containing Feather and NetCDF inputs |
| `PFT_DIR` | `$DATA_DIR/PFT` | Directory containing target PFT Feather files |
| `CHECKPOINT_DIR` | `./checkpoint` | Root checkpoint directory |
| `RESULT_DIR` | `./result` | Result output directory |
| `NORMALIZER_DIR` | `./normalizers` | Saved normalizer directory |

For a fixed normalizer range or a custom DGVM file, call
`multi_prediction_evaluation.py` directly and add its corresponding
`--normalizer_*` or `--dgvm_*` arguments. For example:

```bash
torchrun --nproc-per-node=1 multi_prediction_evaluation.py \
  --target ORCHIDEE_nbp --exp_id 101 \
  --data_name data25_0616 \
  --start_year 2022 --start_month 1 \
  --end_year 2023 --end_month 12 \
  --normalizer_start_year 2000 \
  --normalizer_end_year 2021
```

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
└── <EXP_ID>.csv
```

Result CSV files contain the prediction coordinates and the model output. Do
not commit checkpoints, normalizers, results, logs, or input data to the
public repository.

## Troubleshooting

### `PFT data length should equal to data length`

The PFT file does not have the same number of rows as the main Feather table.
Regenerate it from the same grid and time range, and verify that its row order
matches the main table.

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
model/                         Neural-network definitions
process_data.py                Feather loading and sample construction
data_utils.py                  Dataset utilities kept for compatibility
pipeline.py                    Training and non-recursive evaluation entry point
multi_prediction_evaluation.py Recursive month-by-month evaluation
scripts/train.sh               Training launcher
scripts/evaluate.sh            Recursive evaluation launcher
install_dep_pkgs.sh            PyTorch and dependency installation
requirements.txt               Python dependencies other than PyTorch
```

The public package is intentionally small. Data conversion, cluster job
templates, experiment-specific configuration, and private source files should
remain in the corresponding internal project.
