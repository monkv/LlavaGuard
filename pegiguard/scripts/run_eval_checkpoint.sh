#!/usr/bin/env bash
set -u

BASE_MODEL_DIR="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/models/QwenGuard3.6/rating/v3"
CHECKPOINTS=(480 688) #120 180 240 300

TP=2
PORT=8000

DATA_PATH="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/datasets/PEGI-Guard-DS/age_rating/v2/test.json"
SAVE_BASE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/data/evaluation/age_rating/pegi_v2"
EVAL_FILE="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/LlavaGuard/pegiguard/evaluation/eval_rating.py"
LOG_DIR="${SAVE_BASE}/logs"

mkdir -p "$SAVE_BASE" "$LOG_DIR"

for CKPT in "${CHECKPOINTS[@]}"; do

  MODEL_DIR="${BASE_MODEL_DIR}/merged_checkpoint_${CKPT}"
  OUT_DIR="${SAVE_BASE}/checkpoint_${CKPT}/test_results.json"
  RESPONSES_PATH="${SAVE_BASE}/checkpoint_${CKPT}/responses.json"
  LOG_FILE="${LOG_DIR}/checkpoint_${CKPT}.log"

  mkdir -p "$(dirname "$OUT_DIR")"

  echo "==================================================" | tee -a "$LOG_FILE"
  echo "Starting checkpoint ${CKPT} at $(date)" | tee -a "$LOG_FILE"
  echo "Model: ${MODEL_DIR}" | tee -a "$LOG_FILE"

  vllm serve "$MODEL_DIR" \
    --tensor-parallel-size "$TP" \
    --load-format safetensors \
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
for i in {1..120}; do
  if python - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/health", timeout=2).read()
PY
  then
    READY=1
    break
  fi

  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died while starting checkpoint ${CKPT}" | tee -a "$LOG_FILE"
    tail -n 80 "$LOG_FILE"
    READY=0
    break
  fi

  sleep 5
done

  if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: vLLM did not start for checkpoint ${CKPT}" | tee -a "$LOG_FILE"
    cleanup
    continue
  fi

  echo "vLLM ready. Running evaluation..." | tee -a "$LOG_FILE"

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
    echo "SUCCESS: checkpoint ${CKPT} finished" | tee -a "$LOG_FILE"
  fi

  cleanup

  echo "Finished checkpoint ${CKPT} at $(date)" | tee -a "$LOG_FILE"
  echo "Log saved to ${LOG_FILE}"
done