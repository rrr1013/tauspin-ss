#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/taukin-gap-20261023
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/taukin_gap/posterior_kin.py
C="--tables out/kin_tables.npz --response $HOME/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --stride 10 --workers 6"
$PY -u $P --output out/sub_vm2 $C --ip-vm 2 > logs/sub_vm2.log 2>&1 &
$PY -u $P --output out/sub_vm6 $C --ip-vm 6 > logs/sub_vm6.log 2>&1 &
$PY -u $P --output out/sub_vm20 $C --ip-vm 20 > logs/sub_vm20.log 2>&1 &
$PY -u $P --output out/sub_truthpsi $C --ip-vm 2000 --ip-truth-psi > logs/sub_truthpsi.log 2>&1 &
wait
echo "DONE $(hostname) $(date +%s)" > out/vm_DONE
