#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
C="--tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --rows 3000"
run() { n=$1; shift; $PY -u $P --output out/pilot_$n $C "$@" > logs/pilot_$n.log 2>&1; }
run base --workers 11 &
run track --vis track --workers 11 &
run ptbin --met-model ptbin --workers 11 &
run dl --decay-length --workers 11 &
wait
run m1 --mass-sigma 1 --M 512 --block 10 --workers 11 &
run m05 --mass-sigma 0.5 --M 512 --block 10 --workers 11 &
run mt05 --mass-prior train --mass-sigma 0.5 --M 512 --block 10 --workers 11 &
run exact --met-model exact --workers 11 &
wait
echo "DONE $(hostname) $(date +%s)" > out/pilot_DONE
