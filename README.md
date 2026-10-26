# NRT_LAND minimal package

This is the small hand-off version of the NRT_LAND emulator. It contains only
the model definitions, data loading/preprocessing, training/evaluation entry
points, dependency files, and two generic launcher examples. It does not
include data, PFT files, checkpoints, generated results, logs, notebooks, or
experiment-specific job templates.

## Layout

```text
model/                         neural-network definitions
process_data.py               load training/evaluation samples
pipeline.py                   training entry point
multi_prediction_evaluation.py recursive evaluation entry point
scripts/train.sh               configurable training example
scripts/evaluate.sh            configurable evaluation example
```

## Setup

```bash
conda create -n nrt_land python=3.10 -y
conda activate nrt_land
pip install -r requirements.txt
```

Put the prepared files on the compute machine as follows (or pass explicit
paths to the launchers):

```text
data/<data_name>.feather
data/Trendyv14_S2_nbp.nc
data/average_nbp.nc
data/PFT/*.feather
checkpoint/<experiment>/...
```

The source repository contains no credentials or large binary inputs. The
default paths are relative to the repository; use `DATA_DIR`, `PFT_DIR`,
`CHECKPOINT_DIR`, `RESULT_DIR`, and `NORMALIZER_DIR` to point to local storage.

## Run training

```bash
TARGET=ORCHIDEE_nbp EXP_ID=101 DATA_NAME=data25_0616 \
  GPUS=1 bash scripts/train.sh
```

The launcher uses `torchrun` and can be changed to multiple GPUs with
`GPUS=4`. Additional model options can be supplied through the variables
documented in `scripts/train.sh`.

## Run recursive evaluation

```bash
TARGET=ORCHIDEE_nbp EXP_ID=101 DATA_NAME=data25_0616 \
  START_YEAR=2022 START_MONTH=1 END_YEAR=2023 END_MONTH=12 \
  bash scripts/evaluate.sh
```

Keep normalizers and outputs outside version control. Record the commit,
command line, data version, checkpoint version, and Python environment for
each experiment.
