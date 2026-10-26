#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN="${TORCHRUN:-torchrun}"
GPUS="${GPUS:-1}"
TARGET="${TARGET:?Set TARGET, for example ORCHIDEE_nbp}"
EXP_ID="${EXP_ID:?Set EXP_ID, for example 101}"
DATA_NAME="${DATA_NAME:-data25_0616}"
MODEL="${MODEL:-cnn}"
WINDOW="${WINDOW:-1}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
PFT_DIR="${PFT_DIR:-$DATA_DIR/PFT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoint}"
RESULT_DIR="${RESULT_DIR:-$ROOT/result}"

exec "$TORCHRUN" --nproc-per-node="$GPUS" "$ROOT/pipeline.py" \
  --stage train --target "$TARGET" --exp_id "$EXP_ID" \
  --model "$MODEL" --window "$WINDOW" --data_name "$DATA_NAME" \
  --dump_dir "$CHECKPOINT_DIR" --result_dir "$RESULT_DIR" \
  --preprocessed_data_dir "$DATA_DIR" --pft_data_dir "$PFT_DIR"
