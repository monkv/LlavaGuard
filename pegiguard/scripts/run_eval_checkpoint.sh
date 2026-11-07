#!/usr/bin/env bash
set -u

BASE_MODEL_DIR="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/models/QwenGuard3.6/rating/v3"
CHECKPOINTS=(240) # 120 180 240 300

DATA_PATH="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/datasets/PEGI-Guard-DS/age_rating/v2/test.json"
SAVE_BASE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/evaluation/age_rating/pegi_v2"
EVAL_FILE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/LlavaGuard/pegiguard/evaluation/eval_rating.py"
LOG_DIR="${SAVE_BASE}/logs"

mkdir -p "$SAVE_BASE" "$LOG_DIR"

for CKPT in "${CHECKPOINTS[@]}"; do

  MODEL_DIR="${BASE_MODEL_DIR}/merged_checkpoint_${CKPT}"
  OUT_DIR="${SAVE_BASE}/checkpoint_${CKPT}/test_results_2.json"
  RESPONSES_PATH="${SAVE_BASE}/checkpoint_${CKPT}/responses.json"
  LOG_FILE="${LOG_DIR}/checkpoint_${CKPT}_eval_only.log"

  mkdir -p "$(dirname "$OUT_DIR")"

  echo "==================================================" | tee -a "$LOG_FILE"
  echo "Starting evaluation for checkpoint ${CKPT} at $(date)" | tee -a "$LOG_FILE"
  echo "Model: ${MODEL_DIR}" | tee -a "$LOG_FILE"
  echo "Responses: ${RESPONSES_PATH}" | tee -a "$LOG_FILE"
  echo "Output: ${OUT_DIR}" | tee -a "$LOG_FILE"

  if [[ ! -f "$RESPONSES_PATH" ]]; then
    echo "ERROR: responses file not found: ${RESPONSES_PATH}" | tee -a "$LOG_FILE"
    continue
  fi

  python "$EVAL_FILE" \
    --model_dir "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --responses_path "$RESPONSES_PATH" \
    --label_mode "age_rating" \
    >> "$LOG_FILE" 2>&1

  EXIT_CODE=$?

  if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "ERROR: evaluation failed for checkpoint ${CKPT} with exit code ${EXIT_CODE}" | tee -a "$LOG_FILE"
  else
    echo "SUCCESS: checkpoint ${CKPT} evaluation finished" | tee -a "$LOG_FILE"
  fi

  echo "Finished checkpoint ${CKPT} at $(date)" | tee -a "$LOG_FILE"
  echo "Log saved to ${LOG_FILE}"
done