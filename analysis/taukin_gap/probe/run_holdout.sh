#!/usr/bin/env bash
# TRAIN holdout (event_number % 5 == 0, excluded from every response table): classical-vs-classical deltas of the IP arms
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
C="--tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --split train --stride 3 --workers 11"
run() { n=$1; shift; $PY -u $P --output out/ho_$n $C "$@" > logs/ho_$n.log 2>&1; }
run base &
run track_dl --vis track --decay-length &
run fine --vis track --decay-length --ip-fine &
run orient --vis track --decay-length --ip-orient &
wait
run aniso --vis track --decay-length --ip-aniso
echo "DONE $(hostname) $(date +%s)" > out/ho_DONE
