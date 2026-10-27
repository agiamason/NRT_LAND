#!/usr/bin/env bash
# Emulator runs behind the paper, for the 19 TRENDY v14 S2 models.
#
#   Experiment 1 (exp1): train on 2000-2021; one-step 2010-2023; recursive 2022-01..2023-12.
#   Experiment 2 (exp2): train on 2000-2023; one-step 2010-2023; recursive 2024-01..2024-12,
#                        PFT held at 2023 and the 2023 missing-data mask used for 2024.
#
# Usage: scripts/reproduce_paper.sh exp1|exp2 [train] [onestep] [recursive]
#        (no stage given = onestep and recursive, i.e. from existing checkpoints)
# Paths are taken from the environment as in the other launchers; each experiment
# needs its own CHECKPOINT_DIR. Results go to $RESULT_ROOT/<exp>_onestep and
# $RESULT_ROOT/<exp>_recursive. The paper trained on 4 GPUs (GPUS=4, batch 128 per GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${1:?Usage: reproduce_paper.sh exp1|exp2 [train] [onestep] [recursive]}"
shift
STAGES="${*:-onestep recursive}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
export DATA_NAME="${DATA_NAME:-data25_0616}"

case "$EXP" in
  exp1) TRAIN_END=2021; REC_START=2022; REC_END=2023; FREEZE=0; MASK=0 ;;
  exp2) TRAIN_END=2023; REC_START=2024; REC_END=2024; FREEZE=2023; MASK=2023 ;;
  *) echo "unknown experiment: $EXP" >&2; exit 1 ;;
esac

# Experiment id -> target column (ids as used for the published checkpoints).
IDS=(101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119)
TARGETS=(ORCHIDEE_nbp CABLE-POP_nbp CLASSIC_nbp IBIS_nbp JSBACH_nbp JULES_nbp LPX-Bern_nbp
         OCN_nbp VISIT_nbp iMAPLE_nbp ISBA-CTRIP_nbp CLM6.0_nbp LPJwsl_nbp EDv3_nbp ISAM_nbp
         E3SM_nbp CARDAMOM_nbp SDGVM_nbp visit-ut_nbp)

for i in "${!IDS[@]}"; do
  export EXP_ID="${IDS[$i]}" TARGET="${TARGETS[$i]}"
  for stage in $STAGES; do
    echo "== $EXP $stage $EXP_ID $TARGET"
    case "$stage" in
      train)
        KFOLD_START_YEAR=2000 KFOLD_END_YEAR=$TRAIN_END \
          "$ROOT/scripts/train.sh" ;;
      onestep)
        START_YEAR=2010 END_YEAR=2023 \
        NORMALIZER_START_YEAR=2000 NORMALIZER_END_YEAR=$TRAIN_END PFT_FREEZE_YEAR=$FREEZE \
        RESULT_DIR="$RESULT_ROOT/${EXP}_onestep" \
          "$ROOT/scripts/onestep.sh" ;;
      recursive)
        START_YEAR=$REC_START START_MONTH=1 END_YEAR=$REC_END END_MONTH=12 \
        NORMALIZER_MODE=load NORMALIZER_START_YEAR=2000 NORMALIZER_END_YEAR=$TRAIN_END \
        PFT_FREEZE_YEAR=$FREEZE MASK_YEAR=$MASK HISTORY_DIR="$RESULT_ROOT/${EXP}_onestep" \
        RESULT_DIR="$RESULT_ROOT/${EXP}_recursive" \
          "$ROOT/scripts/evaluate.sh" ;;
      *) echo "unknown stage: $stage" >&2; exit 1 ;;
    esac
  done
done
