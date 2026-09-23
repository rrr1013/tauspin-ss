#!/usr/bin/env bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 GEO_DIM=31 GEO_KEY=full
export GEO_FEATURES=$HOME/hz-spin-synthesis-20260923/artifacts/geo_v3.npz
cd ~/hz-spin-synthesis-20260923
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
T=~/hz-beyond-ceiling-20260923/scripts/p11_train_full_geometry.py
(SEED=42 CUDA_VISIBLE_DEVICES=2 $PY -u $T --arm geo16 --output-dir artifacts/p11/v3_s42 > logs/p11_v3_s42.log 2>&1) &
(SEED=43 CUDA_VISIBLE_DEVICES=3 $PY -u $T --arm geo16 --output-dir artifacts/p11/v3_s43 > logs/p11_v3_s43.log 2>&1) &
wait
touch artifacts/p11/DONE_X8
