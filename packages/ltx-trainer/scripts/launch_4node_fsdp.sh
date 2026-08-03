#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ID TRAIN_CONFIG" >&2
  exit 2
fi

run_id="$1"
train_config="$2"
node_count=4
gpus_per_node=8
world_size=$((node_count * gpus_per_node))
master_port="${MASTER_PORT:-2345}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
trainer_root="${repo_root}/packages/ltx-trainer"
venv_root="${repo_root}/../.venv"
control_root="${repo_root}/../.multinode-rendezvous"
run_dir="${control_root}/${run_id}"

if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
fi
if [[ ! -f "$train_config" ]]; then
  echo "Training config does not exist: ${train_config}" >&2
  exit 2
fi
if [[ ! -x "${venv_root}/bin/accelerate" ]]; then
  echo "Accelerate executable does not exist: ${venv_root}/bin/accelerate" >&2
  exit 2
fi

mkdir -p "$run_dir"

machine_rank=""
current_hostname="$(hostname)"
for candidate in $(seq 0 $((node_count - 1))); do
  claim_dir="${run_dir}/rank_${candidate}.claim"
  if mkdir "$claim_dir" 2>/dev/null; then
    machine_rank="$candidate"
    printf '%s\n' "$current_hostname" > "${claim_dir}/hostname"
    hostname -I | awk '{print $1}' > "${claim_dir}/ip"
    break
  fi
  if [[ -f "${claim_dir}/hostname" ]] && [[ "$(cat "${claim_dir}/hostname")" == "$current_hostname" ]]; then
    machine_rank="$candidate"
    hostname -I | awk '{print $1}' > "${claim_dir}/ip"
    break
  fi
done

if [[ -z "$machine_rank" ]]; then
  echo "No free machine rank in ${run_dir}. Use a new unique RUN_ID." >&2
  exit 1
fi

if [[ "$machine_rank" == "0" ]]; then
  master_addr="$(cat "${run_dir}/rank_0.claim/ip")"
  tmp_master="${run_dir}/master.env.tmp"
  {
    echo "MASTER_ADDR=${master_addr}"
    echo "MASTER_PORT=${master_port}"
  } > "$tmp_master"
  mv "$tmp_master" "${run_dir}/master.env"
fi

deadline=$((SECONDS + 300))
while [[ ! -f "${run_dir}/master.env" ]]; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for rank 0 to publish master address." >&2
    exit 1
  fi
  sleep 1
done

while true; do
  registered="$(find "$run_dir" -maxdepth 1 -type d -name 'rank_*.claim' | wc -l)"
  if [[ "$registered" == "$node_count" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${node_count} nodes; found ${registered}." >&2
    exit 1
  fi
  sleep 1
done

source "${run_dir}/master.env"
export MASTER_ADDR MASTER_PORT

echo "RUN_ID=${run_id} machine_rank=${machine_rank}/${node_count} master=${MASTER_ADDR}:${MASTER_PORT}"

cd "$repo_root"
exec "${venv_root}/bin/accelerate" launch \
  --config_file "${trainer_root}/configs/accelerate/fsdp_4node_32gpu.yaml" \
  --num_machines "$node_count" \
  --num_processes "$world_size" \
  --machine_rank "$machine_rank" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  "${trainer_root}/scripts/train.py" \
  "$train_config"
