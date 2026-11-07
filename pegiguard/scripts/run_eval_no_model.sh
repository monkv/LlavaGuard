#!/usr/bin/env bash
set -u


DATA_PATH="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/datasets/PEGI-Guard-DS/age_rating/v2/test.json"
SAVE_BASE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/evaluation/age_rating/base_models"
EVAL_FILE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/LlavaGuard/pegiguard/evaluation/eval_rating.py"
LOG_DIR="${SAVE_BASE}/logs"


MODELS=(
  "google/gemma-3-12b-it"
  "OpenGVLab/InternVL3-9B"
  "OpenGVLab/InternVL3_5-14B"
  "Qwen/Qwen3.6-35B-A3B"
)

#"moonshotai/Kimi-VL-A3B-Instruct"
#"moonshotai/Kimi-VL-A3B-Instruct"
#internlm/internlm3-8b-instruct
#"google/gemma-4-E2B-it"
#"google/gemma-4-12B-it"
#"OpenGVLab/InternVL3-1B-hf"

mkdir -p "$SAVE_BASE" "$LOG_DIR"

sanitize_name() {
  echo "$1" | sed 's#[/:]#_#g'
}

for MODEL in "${MODELS[@]}"; do
  MODEL_SAFE=$(sanitize_name "$MODEL")
  

  OUT_DIR="${SAVE_BASE}/${MODEL_SAFE}/test_results.json"
  RESPONSES_PATH="${SAVE_BASE}/${MODEL_SAFE}/responses.json"
  LOG_FILE="${LOG_DIR}/${MODEL_SAFE}.log"

  mkdir -p "$(dirname "$OUT_DIR")"
  echo "==================================================" | tee -a "$LOG_FILE"
  echo "Starting evaluation for model ${MODEL} at $(date)" | tee -a "$LOG_FILE"
  echo "Model: ${MODEL}" | tee -a "$LOG_FILE"
  echo "Responses: ${RESPONSES_PATH}" | tee -a "$LOG_FILE"
  echo "Output: ${OUT_DIR}" | tee -a "$LOG_FILE"

  if [[ ! -f "$RESPONSES_PATH" ]]; then
    echo "ERROR: responses file not found: ${RESPONSES_PATH}" | tee -a "$LOG_FILE"
    continue
  fi

  python "$EVAL_FILE" \
    --model_dir "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --responses_path "$RESPONSES_PATH" \
    --label_mode "age_rating" \
    >> "$LOG_FILE" 2>&1

  EXIT_CODE=$?

  if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "ERROR: evaluation failed for model ${MODEL} with exit code ${EXIT_CODE}" | tee -a "$LOG_FILE"
  else
    echo "SUCCESS: model ${MODEL} evaluation finished" | tee -a "$LOG_FILE"
  fi

  echo "Finished model ${MODEL} at $(date)" | tee -a "$LOG_FILE"
  echo "Log saved to ${LOG_FILE}"
done
