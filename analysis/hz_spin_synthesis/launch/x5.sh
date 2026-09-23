#!/usr/bin/env bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 GEO_DIM=22 GEO_KEY=full
export GEO_FEATURES=$HOME/hz-beyond-ceiling-20260923/artifacts/geo_full22.npz
cd ~/hz-spin-synthesis-20260923
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
T=~/hz-beyond-ceiling-20260923/scripts/p11_train_full_geometry.py
mkdir -p artifacts/p11
(SEED=44 CUDA_VISIBLE_DEVICES=0 $PY -u $T --arm geo16 --output-dir artifacts/p11/full22_s44 > logs/p11_full22_s44.log 2>&1) &
(SEED=45 CUDA_VISIBLE_DEVICES=1 $PY -u $T --arm geo16 --output-dir artifacts/p11/full22_s45 > logs/p11_full22_s45.log 2>&1) &
wait
touch artifacts/p11/DONE_X5
