#!/usr/bin/env bash
set -euo pipefail

cd /home/rbaba/rho-gap-20261024
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=~/tauspin-ss/NN/.venv-gpu/bin/python
P=code/analysis/rho_gap/posterior_rho_oracle.py
C="--tables out/kin_tables.npz --response /home/rbaba/rho-pi0-marginal-20261022/out/response.npz --vis track --decay-length --ip-orient --seed 20261017 --stride 10"

echo "=== Running Stride-10 Probes (5,939 events each) ==="

# 1. reco_pi0_truth_psi
echo "Starting probe_reco_pi0_truth_psi..."
$PY -u $P --output out/probe_reco_pi0_truth_psi $C --rho-pi0 marg --rho-psi truth --rho-psi-scope rho_only --workers 10 > logs/probe_reco_pi0_truth_psi.log 2>&1 &

# 2. truth_pi0_reco_psi
echo "Starting probe_truth_pi0_reco_psi..."
$PY -u $P --output out/probe_truth_pi0_reco_psi $C --rho-pi0 truth --rho-psi reco --workers 10 > logs/probe_truth_pi0_reco_psi.log 2>&1 &

# 3. truth_pi0_truth_psi
echo "Starting probe_truth_pi0_truth_psi..."
$PY -u $P --output out/probe_truth_pi0_truth_psi $C --rho-pi0 truth --rho-psi truth --rho-psi-scope rho_only --workers 10 > logs/probe_truth_pi0_truth_psi.log 2>&1 &

# 4. truthdir_pi0_reco_psi
echo "Starting probe_truthdir_pi0_reco_psi..."
$PY -u $P --output out/probe_truthdir_pi0_reco_psi $C --rho-pi0 truthdir --rho-psi reco --workers 9 > logs/probe_truthdir_pi0_reco_psi.log 2>&1 &

# 5. truthe_pi0_reco_psi
echo "Starting probe_truthe_pi0_reco_psi..."
$PY -u $P --output out/probe_truthe_pi0_reco_psi $C --rho-pi0 truthe --rho-psi reco --workers 9 > logs/probe_truthe_pi0_reco_psi.log 2>&1 &

wait
echo "=== All Stride-10 Probes Finished ==="
