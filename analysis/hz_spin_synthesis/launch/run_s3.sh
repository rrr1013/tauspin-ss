#!/usr/bin/env bash
# usage: run_s3.sh TAG NCHUNK ROWS [s3 args...]   (row-chunked, one process per chunk, then merge)
set -u
TAG=$1; N=$2; ROWS=$3; shift 3
cd ~/hz-spin-synthesis-20260923
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
OUT=artifacts/s3/$TAG; mkdir -p $OUT logs/s3
STEP=$(( (ROWS + N - 1) / N ))
for ((i=0; i<N; i++)); do
  S=$((i*STEP)); E=$(( (i+1)*STEP )); [ $E -gt $ROWS ] && E=$ROWS
  $PY -u scripts/s3_reco_nullspace.py "$@" --row-start $S --row-stop $E --output $OUT/chunk_r$S.npz > logs/s3/${TAG}_r$S.log 2>&1 &
done
wait
$PY scripts/s4_merge.py --inputs $OUT/chunk_r*.npz --output $OUT/merged.npz --scores-dir artifacts/s3_scores --tag $TAG > logs/s3/${TAG}_merge.log 2>&1
touch $OUT/DONE
