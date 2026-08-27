#!/usr/bin/env bash
set -euo pipefail

# Evaluate the replacement eight-sample Part16 diving set on the four stage-1
# step-800 checkpoints. Two GPUs are assigned to each checkpoint; each worker
# generates a disjoint four-sample slice with identical sampling settings.

WORKSPACE=/workspace/code/controlsft_teacher
REPO="$WORKSPACE/Controlsft"
PYTHON="$WORKSPACE/.venv/bin/python"
INFER="$REPO/packages/ltx-trainer/scripts/infer_part16_precomputed.py"
BASE="$WORKSPACE/checkpoints/ltx-2.3/ltx-2.3-22b-dev.safetensors"
MANIFEST=/workspace/code/datasets/part16_eval8_source_aligned_960x544/test_inference.jsonl
OUTPUT_ROOT="$WORKSPACE/outputs/validation_new_diving8_step_00800"

declare -a NAMES=(baseline mlp3 mlp5 mlp8)
declare -a CHECKPOINTS=(
  "$WORKSPACE/outputs/ablation_stage1_baseline/checkpoints/model_weights_step_00800.safetensors"
  "$WORKSPACE/outputs/ablation_stage1_sra_mlp3_layer16/checkpoints/model_weights_step_00800.safetensors"
  "$WORKSPACE/outputs/ablation_stage1_sra_mlp5_layer16/checkpoints/model_weights_step_00800.safetensors"
  "$WORKSPACE/outputs/ablation_stage1_sra_mlp8_layer16/checkpoints/model_weights_step_00800.safetensors"
)

mkdir -p "$OUTPUT_ROOT/logs"
pids=()
for model_index in "${!NAMES[@]}"; do
  for half in 0 1; do
    gpu=$((model_index * 2 + half))
    start=$((half * 4))
    name="${NAMES[$model_index]}"
    log="$OUTPUT_ROOT/logs/${name}_${start}_$((start + 3)).log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$INFER" \
      --base-checkpoint "$BASE" \
      --trained-checkpoint "${CHECKPOINTS[$model_index]}" \
      --manifest-path "$MANIFEST" \
      --output-dir "$OUTPUT_ROOT/$name" \
      --start-index "$start" \
      --num-samples 4 \
      --inference-steps 30 \
      --guidance-scale 1.0 \
      --stg-scale 1.0 \
      --stg-blocks 29 \
      --seed 42 \
      --disable-progress-bars \
      >"$log" 2>&1 &
    pids+=("$!")
    echo "GPU $gpu: $name samples $start..$((start + 3)), log=$log"
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if ((failed)); then
  echo "One or more inference workers failed. Inspect $OUTPUT_ROOT/logs." >&2
  exit 1
fi

echo "All four checkpoint evaluations completed: $OUTPUT_ROOT"
