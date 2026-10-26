#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN="${TORCHRUN:-torchrun}"
GPUS="${GPUS:-1}"
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

exec "$TORCHRUN" --nproc-per-node="$GPUS" "$ROOT/multi_prediction_evaluation.py" \
  --target "$TARGET" --exp_id "$EXP_ID" --data_name "$DATA_NAME" \
  --start_year "$START_YEAR" --start_month "$START_MONTH" \
  --end_year "$END_YEAR" --end_month "$END_MONTH" \
  --dump_dir "$CHECKPOINT_DIR" --result_dir "$RESULT_DIR" \
  --preprocessed_data_dir "$DATA_DIR" --pft_data_dir "$PFT_DIR" \
  --normalizer_dir "$NORMALIZER_DIR"
