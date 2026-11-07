#!/usr/bin/env bash
set -u

MODELS=(
  #"Qwen/Qwen3.6-35B-A3B" #
  "google/gemma-4-E2B-it"
  "google/gemma-4-12B-it"
  "moonshotai/Kimi-VL-A3B-Instruct"
  #"OpenGVLab/InternVL3-1B-hf"
)

TP=2
PORT=8000

SAVE_BASE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/evaluation/age_rating/base_models""

DATA_PATH="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/datasets/PEGI-Guard-DS/age_rating/v2/test.json"
EVAL_FILE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/LlavaGuard/pegiguard/evaluation/eval_rating.py"

LOG_DIR="${SAVE_BASE}/logs" 

mkdir -p "$SAVE_BASE" "$LOG_DIR"

sanitize_name() {
  echo "$1" | sed 's#[/:]#_#g'
}

cleanup_old_vllm() {
  echo "Cleaning old vLLM processes..."
  pkill -9 -f "vllm serve" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  sleep 15
}

for MODEL in "${MODELS[@]}"; do
  MODEL_SAFE=$(sanitize_name "$MODEL")
  SERVED_MODEL_NAME="$MODEL_SAFE"

  OUT_DIR="${SAVE_BASE}/${MODEL_SAFE}/test_results.json"
  RESPONSES_PATH="${SAVE_BASE}/${MODEL_SAFE}/responses.json"
  LOG_FILE="${LOG_DIR}/${MODEL_SAFE}.log"

  mkdir -p "$(dirname "$OUT_DIR")"

  echo "==================================================" | tee -a "$LOG_FILE"
  echo "Starting model ${MODEL} at $(date)" | tee -a "$LOG_FILE"
  echo "Served model name: ${SERVED_MODEL_NAME}" | tee -a "$LOG_FILE"

  cleanup_old_vllm

  vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP" \
    --load-format auto \
    --gpu-memory-utilization 0.80 \
    --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &

  VLLM_PID=$!

  cleanup() {
    echo "Stopping vLLM..." | tee -a "$LOG_FILE"
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 10
    kill -9 "$VLLM_PID" 2>/dev/null || true
    pkill -9 -P "$VLLM_PID" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
    sleep 20
  }

  echo "Waiting for vLLM server..." | tee -a "$LOG_FILE"

  READY=0
  for i in {1..180}; do
    if python - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/health", timeout=2).read()
PY
    then
      READY=1
      break
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM died while starting ${MODEL}" | tee -a "$LOG_FILE"
      tail -n 100 "$LOG_FILE"
      READY=0
      break
    fi

    sleep 5
  done

  if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: vLLM did not start for ${MODEL}" | tee -a "$LOG_FILE"
    cleanup
    continue
  fi

  echo "vLLM ready. Served models:" | tee -a "$LOG_FILE"
  python - <<PY 2>&1 | tee -a "$LOG_FILE"
import urllib.request, json
data = json.load(urllib.request.urlopen("http://127.0.0.1:${PORT}/v1/models"))
print(json.dumps(data, indent=2))
PY

  echo "Running evaluation..." | tee -a "$LOG_FILE"

  python "$EVAL_FILE" \
    --model_dir "$SERVED_MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --responses_path "$RESPONSES_PATH" \
    --label_mode "age_rating" \
    >> "$LOG_FILE" 2>&1

  EXIT_CODE=$?

  if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "ERROR: evaluation failed for ${MODEL} with exit code ${EXIT_CODE}" | tee -a "$LOG_FILE"
  else
    echo "SUCCESS: ${MODEL} finished" | tee -a "$LOG_FILE"
  fi

  cleanup

  echo "Finished model ${MODEL} at $(date)" | tee -a "$LOG_FILE"
  echo "Log saved to ${LOG_FILE}"
done