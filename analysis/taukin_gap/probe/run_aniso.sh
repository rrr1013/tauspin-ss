#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
until [ -f out/sub_DONE ]; do sleep 20; done
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
$PY -u $P --output out/val_track_dl_aniso --tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --ip-aniso --workers 24 > logs/val_track_dl_aniso.log 2>&1
echo "DONE $(hostname) $(date +%s)" > out/val4_DONE
