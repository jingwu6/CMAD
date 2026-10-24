#!/bin/bash
# End-to-end CMAD run: local tests -> data prep -> SFT loop + eval -> plot.
# Install dependencies first (see README). Configure via environment variables:
#   DATASET           gsm8k | math                  (default: gsm8k)
#   TRAIN_POOL_SIZE   training questions            (default: 300)
#   EVAL_SIZE         eval questions                (default: 500)
#   NUM_ITERATIONS    SFT iterations L              (default: 4)
#   CONCURRENCY       concurrent questions          (default: 16)
#   VLLM_GPU          GPU for the vLLM server       (default: 0)
#   TRAIN_GPU         GPU for LoRA fine-tuning      (default: 1)
#   SKIP_BASELINES=1  skip cot/vanilla_mad/dmad
#   EXTRA_RUN_CMAD_FLAGS  extra flags passed to scripts/run_cmad_full.py
set -euo pipefail

TRAIN_POOL_SIZE="${TRAIN_POOL_SIZE:-300}"
EVAL_SIZE="${EVAL_SIZE:-500}"
NUM_ITERATIONS="${NUM_ITERATIONS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
DATASET="${DATASET:-gsm8k}"
VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
SKIP_BASELINES_FLAG=""
if [ "${SKIP_BASELINES:-0}" = "1" ]; then
  SKIP_BASELINES_FLAG="--skip-baselines"
fi
EXTRA_RUN_CMAD_FLAGS="${EXTRA_RUN_CMAD_FLAGS:-}"

echo "=== Local tests ==="
for f in tests/test_*.py; do
  python3 "$f"
done

echo "=== Preparing $DATASET (train_pool=$TRAIN_POOL_SIZE, eval=$EVAL_SIZE) ==="
python3 "scripts/prepare_${DATASET}.py" --out-dir data_full --eval-size "$EVAL_SIZE" --train-pool-size "$TRAIN_POOL_SIZE"

echo "=== Running CMAD (dataset=$DATASET, L=$NUM_ITERATIONS) ==="
SECONDS=0
python3 scripts/run_cmad_full.py \
    --train-file data_full/train_pool.jsonl --eval-file data_full/eval.jsonl \
    --base-model Qwen/Qwen2.5-7B-Instruct --num-iterations "$NUM_ITERATIONS" \
    --concurrency "$CONCURRENCY" --adapter-dir run_adapters --results-dir results \
    --dataset "$DATASET" --vllm-gpu "$VLLM_GPU" --train-gpu "$TRAIN_GPU" \
    $SKIP_BASELINES_FLAG $EXTRA_RUN_CMAD_FLAGS
echo "run_cmad_full took ${SECONDS}s"

python3 scripts/make_results_plot.py --results-dir results --num-iterations "$NUM_ITERATIONS" --out results/plot.png
echo "=== Done. Results in results/, adapters in run_adapters/ ==="
