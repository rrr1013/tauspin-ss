#!/usr/bin/env bash
set -euo pipefail

cd /home/rbaba/rho-gap-20261024
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/rho_gap/posterior_rho_oracle.py
C="--tables out/kin_tables.npz --response /home/rbaba/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --ip-orient --seed 20261017"

# Arm 1: Symlink baseline from taukin-gap-20261023
if [ ! -d out/val_reco_pi0_reco_psi ]; then
    echo "Symlinking baseline val_reco_pi0_reco_psi..."
    ln -sfn /home/rbaba/taukin-gap-20261023/out/val_track_dl_orient out/val_reco_pi0_reco_psi
fi

echo "=== Batch 1: reco_pi0_truth_psi and truth_pi0_reco_psi ==="
$PY -u $P --output out/val_reco_pi0_truth_psi $C --rho-pi0 marg --rho-psi truth --rho-psi-scope rho_only --workers 22 > logs/val_reco_pi0_truth_psi.log 2>&1 &
$PY -u $P --output out/val_truth_pi0_reco_psi $C --rho-pi0 truth --rho-psi reco --workers 22 > logs/val_truth_pi0_reco_psi.log 2>&1 &
wait
echo "Batch 1 finished at $(date)"

echo "=== Batch 2: truth_pi0_truth_psi and truthdir_pi0_reco_psi ==="
$PY -u $P --output out/val_truth_pi0_truth_psi $C --rho-pi0 truth --rho-psi truth --rho-psi-scope rho_only --workers 22 > logs/val_truth_pi0_truth_psi.log 2>&1 &
$PY -u $P --output out/val_truthdir_pi0_reco_psi $C --rho-pi0 truthdir --rho-psi reco --workers 22 > logs/val_truthdir_pi0_reco_psi.log 2>&1 &
wait
echo "Batch 2 finished at $(date)"

echo "=== Batch 3: truthe_pi0_reco_psi ==="
$PY -u $P --output out/val_truthe_pi0_reco_psi $C --rho-pi0 truthe --rho-psi reco --workers 24 > logs/val_truthe_pi0_reco_psi.log 2>&1 &
wait
echo "Batch 3 finished at $(date)"

echo "=== All Full Validation Arms Finished ==="
