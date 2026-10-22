#!/usr/bin/env bash
# full validation arms; decided before any validation AUC was seen (pilot: H-only rows, side quality only)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
C="--tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz"
run() { n=$1; shift; $PY -u $P --output out/val_$n $C "$@" > logs/val_$n.log 2>&1; }
run base --workers 14 &
run track --vis track --workers 14 &
run track_dl --vis track --decay-length --workers 14 &
wait
echo "DONE $(hostname) $(date +%s)" > out/val1_DONE
run track_dl_mt --vis track --decay-length --mass-prior train --mass-sigma 0.5 --M 512 --block 10 --workers 22 &
run all --vis track --decay-length --mass-prior train --mass-sigma 0.5 --M 512 --block 10 --met-model ptbin --workers 22 &
wait
echo "DONE $(hostname) $(date +%s)" > out/val2_DONE
