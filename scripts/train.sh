#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN="${TORCHRUN:-torchrun}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"   # use a different port for each run on the same machine
TARGET="${TARGET:?Set TARGET, for example ORCHIDEE_nbp}"
EXP_ID="${EXP_ID:?Set EXP_ID, for example 101}"
DATA_NAME="${DATA_NAME:-data25_0616}"
MODEL="${MODEL:-cnn}"
# The evaluators use a 13-month window and 18 x 18 patches, so train with the same values.
WINDOW="${WINDOW:-13}"
PATCH="${PATCH:-18}"
# Years used for the five-fold training split (and for the scaler fitted during training).
# The paper used 2000-2021 for the 2022-2023 test and 2000-2023 for the 2024 prediction.
KFOLD_START_YEAR="${KFOLD_START_YEAR:-2000}"
KFOLD_END_YEAR="${KFOLD_END_YEAR:-2023}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
PFT_DIR="${PFT_DIR:-$DATA_DIR/PFT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoint}"
RESULT_DIR="${RESULT_DIR:-$ROOT/result}"

exec "$TORCHRUN" --nproc-per-node="$GPUS" --master-port="$MASTER_PORT" "$ROOT/pipeline.py" \
  --stage train --target "$TARGET" --exp_id "$EXP_ID" \
  --model "$MODEL" --window "$WINDOW" --patch "$PATCH" --data_name "$DATA_NAME" \
  --kfold_start_year "$KFOLD_START_YEAR" --kfold_end_year "$KFOLD_END_YEAR" \
  --dump_dir "$CHECKPOINT_DIR" --result_dir "$RESULT_DIR" \
  --preprocessed_data_dir "$DATA_DIR" --pft_data_dir "$PFT_DIR"
