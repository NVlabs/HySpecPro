#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run HySpecPro.py on all Titan23 designs and collect cut size / runtime.
#
# Optional environment overrides:
#   PY=/path/to/python
#   DESIGN_ROOT=/path/to/Titan23_benchmark/
#   RESULT_ROOT=./results/
#   TAG=my_run
#   UB=0.02
#   KWAY=2
#   N_CMA_ITE=5
#   DEVICES_OVERRIDE="cuda:0 cuda:1 cuda:2 cuda:3"

set -u
cd "$(dirname "$0")"

PY=${PY:-python}
DESIGN_ROOT=${DESIGN_ROOT:-benchmarks/L_HG_benchmark/}
RESULT_ROOT=${RESULT_ROOT:-./results/}
TAG=${TAG:-HySpecPro_th10}
UB=${UB:-0.1}
KWAY=${KWAY:-2}
N_CMA_ITE=${N_CMA_ITE:-6}
DEVICES=(${DEVICES_OVERRIDE:-cuda:0 cuda:1 cuda:2 cuda:3})

DESIGNS=(
  Bump_2911.mtx CurlCurl_4.mtx Ga41As41H72.mtx Geo_1438.mtx HV15R.mtx StocF-1465.mtx circuit5M.mtx dgreen.mtx
)

mkdir -p "$RESULT_ROOT"
LOG_DIR="logs_${TAG}"
mkdir -p "$LOG_DIR"

CSV="LHG_results_${TAG}.csv"
LOCK="${CSV}.lock"
echo "timestamp,design,device,cut_size,runtime_seconds,status,exit_code,solution_file,score_file,log_file" > "$CSV"

run_design() {
  local design="$1"
  local device="$2"
  local idx="$3"
  local total="$4"

  local start_sec end_sec runtime start_ts exit_code status cut_size
  local solution_file score_file log_file

  start_sec=$(date +%s)
  start_ts=$(date '+%Y-%m-%d %H:%M:%S')
  solution_file="${RESULT_ROOT%/}/${TAG}_${design}_best_solution.pt"
  score_file="${RESULT_ROOT%/}/${TAG}_${design}_best_score.pt"
  log_file="${LOG_DIR}/${TAG}_${design}.log"

  echo "[$(date)] Starting ${design} on ${device} (${idx}/${total})"
  "$PY" HySpecPro.py \
    --design_root "$DESIGN_ROOT" \
    --result_root "$RESULT_ROOT" \
    --design "$design" \
    --device "$device" \
    --tag "$TAG" \
    --N_CMA_ITE "$N_CMA_ITE" \
    --KWAY "$KWAY" \
    --UB "$UB" \
    > "$log_file" 2>&1
  exit_code=$?

  end_sec=$(date +%s)
  runtime=$((end_sec - start_sec))
  if [ "$exit_code" -eq 0 ]; then
    status="SUCCESS"
  else
    status="FAILED"
  fi

  cut_size=""
  if [ -f "$score_file" ]; then
    cut_size=$("$PY" -c 'import sys, torch; x=torch.load(sys.argv[1], map_location="cpu"); print(float(x))' "$score_file" 2>/dev/null || true)
  fi

  (
    flock -x 200
    echo "${start_ts},${design},${device},${cut_size},${runtime},${status},${exit_code},${solution_file},${score_file},${log_file}" >> "$CSV"
  ) 200>"$LOCK"

  echo "[$(date)] Finished ${design}: status=${status}, cut=${cut_size}, runtime=${runtime}s"
  return "$exit_code"
}

export -f run_design
export PY DESIGN_ROOT RESULT_ROOT TAG UB KWAY N_CMA_ITE CSV LOCK LOG_DIR

total=${#DESIGNS[@]}
gpu_idx=0
failed=0
pids=()

for i in "${!DESIGNS[@]}"; do
  design="${DESIGNS[$i]}"
  device="${DEVICES[$((gpu_idx % ${#DEVICES[@]}))]}"
  run_design "$design" "$device" "$((i + 1))" "$total" &
  pids+=($!)
  gpu_idx=$((gpu_idx + 1))

  if [ "${#pids[@]}" -ge "${#DEVICES[@]}" ]; then
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    pids=()
  fi
done

if [ "${#pids[@]}" -gt 0 ]; then
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
fi

rm -f "$LOCK"
echo "Wrote $CSV"
exit "$failed"
