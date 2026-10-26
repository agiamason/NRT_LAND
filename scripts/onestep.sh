#!/usr/bin/env bash
# One-step (non-recursive) monthly prediction with a fixed normalizer
# (kfold_prediction_fixed_scaler.py). Its CSVs are also the history files for
# scripts/evaluate.sh (HISTORY_DIR).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN="${TORCHRUN:-torchrun}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"   # use a different port for each run on the same machine
TARGET="${TARGET:?Set TARGET, for example ORCHIDEE_nbp}"
EXP_ID="${EXP_ID:?Set EXP_ID, for example 101}"
DATA_NAME="${DATA_NAME:-data25_0616}"
START_YEAR="${START_YEAR:-2010}"
END_YEAR="${END_YEAR:-2023}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
PFT_DIR="${PFT_DIR:-$DATA_DIR/PFT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoint}"
RESULT_DIR="${RESULT_DIR:-$ROOT/result/onestep}"
NORMALIZER_DIR="${NORMALIZER_DIR:-$ROOT/normalizers}"
NORMALIZER_MODE="${NORMALIZER_MODE:-auto}"
NORMALIZER_START_YEAR="${NORMALIZER_START_YEAR:-2000}"
NORMALIZER_END_YEAR="${NORMALIZER_END_YEAR:-2021}"
PFT_FREEZE_YEAR="${PFT_FREEZE_YEAR:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

exec "$TORCHRUN" --nproc-per-node="$GPUS" --master-port="$MASTER_PORT" "$ROOT/kfold_prediction_fixed_scaler.py" \
  --target "$TARGET" --exp_id "$EXP_ID" --data_name "$DATA_NAME" \
  --start_year "$START_YEAR" --end_year "$END_YEAR" \
  --dump_dir "$CHECKPOINT_DIR" --result_dir "$RESULT_DIR" \
  --preprocessed_data_dir "$DATA_DIR" --pft_data_dir "$PFT_DIR" \
  --normalizer_mode "$NORMALIZER_MODE" --normalizer_dir "$NORMALIZER_DIR" \
  --normalizer_start_year "$NORMALIZER_START_YEAR" --normalizer_end_year "$NORMALIZER_END_YEAR" \
  --pft_freeze_year "$PFT_FREEZE_YEAR" --num_workers "$NUM_WORKERS"
