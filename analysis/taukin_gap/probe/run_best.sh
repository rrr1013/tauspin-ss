#!/usr/bin/env bash
# combination of the arms that each helped (decided from the paired deltas above): track + decay length + signed-orientation
# IP response + TRAIN parent-mass prior
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
$PY -u $P --output out/val_best --tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --ip-orient --mass-prior train --mass-sigma 0.5 --M 512 --block 10 --workers 44 > logs/val_best.log 2>&1
echo "DONE $(hostname) $(date +%s)" > out/best_DONE
