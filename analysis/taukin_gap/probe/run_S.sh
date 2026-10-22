#!/usr/bin/env bash
# sampler-precision check (computation, not a method change): stride-10 subset, S and M increased
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
C="--tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --stride 10"
run() { n=$1; shift; $PY -u $P --output out/sub_$n $C "$@" > logs/sub_$n.log 2>&1; }
run S16k --workers 8 &
run S64k --S 65536 --block 20 --workers 16 &
wait
run S64kM512 --S 65536 --M 512 --block 5 --workers 24
echo "DONE $(hostname) $(date +%s)" > out/sub_DONE
