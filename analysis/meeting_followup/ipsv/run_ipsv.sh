#!/bin/bash
# Queue the simple-vs-ATLAS IP/SV point-h arms on lxgpu02 (at most MAXJ of our jobs, only idle GPUs).
set -uo pipefail
D=/home/rbaba/meeting-followup-20260930/ipsv
C=/home/rbaba/meeting-followup-20260930/code/analysis/meeting_followup/ipsv/train_ipsv.py
PY=/home/rbaba/tauspin-ss/NN/.venv-gpu/bin/python
MAXJ=${MAXJ:-6}
cd $D; mkdir -p runs logs
L=$D/logs/queue.log
echo "START $(date +%s) $(hostname)" >> $L
JOBS=""
for seed in 42 43; do for s in simple atlas; do for k in none legacy lab local local_shuffle; do
  [ $seed = 43 ] && case $k in legacy|local_shuffle) continue;; esac
  JOBS="$JOBS $s:$k:$seed"; done; done; done
PIDS=()
running() { local n=0; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && n=$((n+1)); done
  # jobs started by an earlier instance of this queue (listed in queue.log without result.json)
  for d in $(awk '/^LAUNCH/{print $2}' $L | sort -u); do [ -f runs/$d/result.json ] || { pgrep -f "output-dir runs/$d\$" >/dev/null && n=$((n+1)); }; done
  echo $n; }
for j in $JOBS; do
  IFS=: read s k seed <<< "$j"; name=${s}_${k}_s${seed}
  [ -f runs/$name/result.json ] && continue
  [ -d runs/$name ] && continue
  while true; do
    mine=$(running)
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<500{print $1; exit}')
    [ "$mine" -lt "$MAXJ" ] && [ -n "$gpu" ] && break
    sleep 60
  done
  G=$D/${s}_inputs.npz; [ $s = atlas ] && G=$D/atlas_geometry.npz
  SAMPLE=$s SIMPLE_INPUTS=$D/simple_inputs.npz GEO_FEATURES=$G GEO_KEY=$k SEED=$seed OMP_NUM_THREADS=2 \
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u $C --arm geo16 --output-dir runs/$name > logs/$name.log 2>&1 &
  PIDS+=($!)
  echo "LAUNCH $name gpu=$gpu $(date +%s)" >> $L
  sleep 90
done
wait
echo "QUEUE_DONE $(date +%s)" >> $L
