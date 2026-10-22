#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
$PY -u $P --output out/val_track_dl_orient --tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --ip-orient --workers 24 > logs/val_track_dl_orient.log 2>&1
echo "DONE $(hostname) $(date +%s)" > out/val5_DONE
