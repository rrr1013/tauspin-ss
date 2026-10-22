#!/usr/bin/env bash
# IP response arms on top of track + decay length (decided after the batch-1 side-quality diagnosis, before any of these
# arms' validation AUC was seen)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
C="--tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length"
run() { n=$1; shift; $PY -u $P --output out/val_$n $C "$@" > logs/val_$n.log 2>&1; }
run track_dl_fine --ip-fine --workers 12 &
run track_dl_joint --ip-joint --workers 12 &
wait
echo "DONE $(hostname) $(date +%s)" > out/val3_DONE
