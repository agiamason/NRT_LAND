#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN="${TORCHRUN:-torchrun}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"   # use a different port for each run on the same machine
TARGET="${TARGET:?Set TARGET, for example ORCHIDEE_nbp}"
EXP_ID="${EXP_ID:?Set EXP_ID, for example 101}"
DATA_NAME="${DATA_NAME:-data25_0616}"
START_YEAR="${START_YEAR:-2022}"
START_MONTH="${START_MONTH:-1}"
END_YEAR="${END_YEAR:-2023}"
END_MONTH="${END_MONTH:-12}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
PFT_DIR="${PFT_DIR:-$DATA_DIR/PFT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoint}"
RESULT_DIR="${RESULT_DIR:-$ROOT/result}"
NORMALIZER_DIR="${NORMALIZER_DIR:-$ROOT/normalizers}"
NORMALIZER_MODE="${NORMALIZER_MODE:-auto}"
# Use the training years of the checkpoint (2000-2021 or 2000-2023 in the paper).
NORMALIZER_START_YEAR="${NORMALIZER_START_YEAR:-2000}"
NORMALIZER_END_YEAR="${NORMALIZER_END_YEAR:-2021}"
# Optional: one-step result directory whose CSVs are continued (see scripts/onestep.sh).
HISTORY_DIR="${HISTORY_DIR:-}"
# Optional (0 = off): missing-data mask year for prediction years without target values,
# and the year at which PFT fractions are held fixed.
MASK_YEAR="${MASK_YEAR:-0}"
PFT_FREEZE_YEAR="${PFT_FREEZE_YEAR:-0}"

exec "$TORCHRUN" --nproc-per-node="$GPUS" --master-port="$MASTER_PORT" "$ROOT/multi_prediction_evaluation.py" \
  --target "$TARGET" --exp_id "$EXP_ID" --data_name "$DATA_NAME" \
  --start_year "$START_YEAR" --start_month "$START_MONTH" \
  --end_year "$END_YEAR" --end_month "$END_MONTH" \
  --dump_dir "$CHECKPOINT_DIR" --result_dir "$RESULT_DIR" \
  --preprocessed_data_dir "$DATA_DIR" --pft_data_dir "$PFT_DIR" \
  --normalizer_mode "$NORMALIZER_MODE" --normalizer_dir "$NORMALIZER_DIR" \
  --normalizer_start_year "$NORMALIZER_START_YEAR" --normalizer_end_year "$NORMALIZER_END_YEAR" \
  --historical_result_dir "$HISTORY_DIR" --mask_year "$MASK_YEAR" --pft_freeze_year "$PFT_FREEZE_YEAR"
