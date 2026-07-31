#!/bin/bash
set -euo pipefail

nn_dir=/home/rbaba/tauspin-ss/NN
snapshot_dir=/home/rbaba/tauspin-ss/MakeNtuple/snapshots/fixed-partial-v3-20260730-2100
matching_dir="$nn_dir/outputs/data-preparation/fixed-partial-v3-20260730-2100-ptmatched20"
processed_dir=/tmp/rbaba-tauspin-fixed-partial-v3-20260730-2100/processed
study_dir="$nn_dir/outputs/hpo/fixed-partial-v3-20260730-2100-high-throughput-hpo-v1"
selection_dir="$study_dir/validation_selection_v1"
hpo_figures_dir="$study_dir/hpo_figures_v1"
retrain_dir="$study_dir/final_50epoch_run_v1"
final_dir="$study_dir/final_evaluation_v1"
final_figures_dir="$study_dir/final_figures_v1"
completion_audit="$study_dir/completion_audit.json"
primary_hpo_complete="$study_dir/controller_complete.json"
extended_hpo_complete="$study_dir/hpo_extension_complete.json"
python="$nn_dir/.venv-gpu/bin/python"

export PYTHONWARNINGS="ignore:networkx backend defined more than once:RuntimeWarning"
ulimit -n 65536
echo "[$(date --iso-8601=seconds)] open-file soft limit $(ulimit -n)"

echo "[$(date --iso-8601=seconds)] waiting for HPO controller"
while [ ! -f "$primary_hpo_complete" ]; do
    sleep 60
done

"$python" - "$primary_hpo_complete" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1]))
counts = document["trial_state_counts"]
if counts.get("RUNNING", 0) or counts.get("WAITING", 0):
    raise RuntimeError(f"HPO controller is not terminal: {counts}")
if counts.get("COMPLETE", 0) < 7:
    raise RuntimeError(f"Too few completed HPO trials: {counts}")
print(json.dumps(counts, indent=2))
PY

if [ ! -f "$extended_hpo_complete" ]; then
    echo "[$(date --iso-8601=seconds)] extending HPO until the deadline gate"
    cd "$nn_dir"
    "$python" -u final_hpo_controller.py \
        --processed-dir "$processed_dir" \
        --event-selection-manifest "$matching_dir/pt_matching_manifest.json" \
        --snapshot-manifest "$snapshot_dir/snapshot_manifest.json" \
        --study-name fixed-partial-v3-20260730-2100-high-throughput-hpo-v1 \
        --output-root "$nn_dir/outputs/hpo" \
        --worker-script "$nn_dir/final_hpo_worker.py" \
        --gpus 0,1,2,3,4 \
        --stop-new-at 2026-07-31T06:30:00+09:00 \
        --max-epochs 32 \
        --objective-start-epoch 8 \
        --early-stop-start-epoch 20 \
        --target-total-trials 200 \
        --poll-seconds 15 \
        --resume-existing \
        --status-file hpo_extension_status.json \
        --completion-file hpo_extension_complete.json
fi

"$python" - "$extended_hpo_complete" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1]))
counts = document["trial_state_counts"]
if counts.get("RUNNING", 0) or counts.get("WAITING", 0):
    raise RuntimeError(f"Extended HPO is not terminal: {counts}")
if counts.get("COMPLETE", 0) < 7:
    raise RuntimeError(f"Too few completed HPO trials: {counts}")
print(json.dumps(counts, indent=2))
PY

cd "$nn_dir"
if [ ! -e "$selection_dir" ]; then
    "$python" select_v3_hpo.py \
        --study-dir "$study_dir" \
        --output-dir "$selection_dir"
else
    echo "[$(date --iso-8601=seconds)] reusing existing validation selection"
fi

if [ ! -e "$hpo_figures_dir" ]; then
    "$python" plot_v3_hpo.py \
        --study-dir "$study_dir" \
        --selection-dir "$selection_dir" \
        --output-dir "$hpo_figures_dir"
else
    echo "[$(date --iso-8601=seconds)] reusing existing HPO figures"
fi

gpu=""
echo "[$(date --iso-8601=seconds)] waiting for one idle GPU"
while [ -z "$gpu" ]; do
    for candidate in 0 1 2 3 4 5 6 7; do
        memory_used=$(nvidia-smi \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits \
            -i "$candidate")
        utilization=$(nvidia-smi \
            --query-gpu=utilization.gpu \
            --format=csv,noheader,nounits \
            -i "$candidate")
        if [ "$memory_used" -le 5 ] && [ "$utilization" -eq 0 ]; then
            gpu="$candidate"
            break
        fi
    done
    if [ -z "$gpu" ]; then
        sleep 60
    fi
done
echo "[$(date --iso-8601=seconds)] selected idle physical GPU $gpu"

test ! -e "$retrain_dir"
CUDA_VISIBLE_DEVICES="$gpu" "$python" -u v3_retrain_50epoch.py \
    --processed-dir "$processed_dir" \
    --selection-json "$selection_dir/selected_parameters.json" \
    --event-selection-manifest "$matching_dir/pt_matching_manifest.json" \
    --snapshot-manifest "$snapshot_dir/snapshot_manifest.json" \
    --output-dir "$retrain_dir"

test ! -e "$final_dir"
CUDA_VISIBLE_DEVICES="$gpu" "$python" -u finalize_v3_50epoch.py \
    --processed-dir "$processed_dir" \
    --selection-json "$selection_dir/selected_parameters.json" \
    --retrain-dir "$retrain_dir" \
    --event-selection-manifest "$matching_dir/pt_matching_manifest.json" \
    --snapshot-manifest "$snapshot_dir/snapshot_manifest.json" \
    --output-dir "$final_dir"

test ! -e "$final_figures_dir"
"$python" plot_v3_final_figures.py \
    --final-dir "$final_dir" \
    --retrain-dir "$retrain_dir" \
    --matching-dir "$matching_dir" \
    --processed-dir "$processed_dir" \
    --output-dir "$final_figures_dir"

test ! -e "$completion_audit"
"$python" audit_v3_completion.py \
    --snapshot-dir "$snapshot_dir" \
    --matching-dir "$matching_dir" \
    --processed-dir "$processed_dir" \
    --study-dir "$study_dir" \
    --selection-dir "$selection_dir" \
    --retrain-dir "$retrain_dir" \
    --final-dir "$final_dir" \
    --hpo-figures-dir "$hpo_figures_dir" \
    --final-figures-dir "$final_figures_dir" \
    --report "$completion_audit"

echo "[$(date --iso-8601=seconds)] fixed-partial-v3 continuation complete"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader
