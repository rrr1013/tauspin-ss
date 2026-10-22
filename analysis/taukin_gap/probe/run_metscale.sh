#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
C="--tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --stride 10 --workers 8"
for f in 1.5 3 30; do $PY -u $P --output out/sub_met$f $C --met-scale $f > logs/sub_met$f.log 2>&1 & done
wait
echo "DONE $(hostname) $(date +%s)" > out/metscale_DONE
