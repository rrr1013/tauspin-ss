#!/bin/bash
# Spin-flat H(125)+jet for the entanglement feasibility:
# reco (no b-tag) -> dataset (MMC table of the regressor training sample) -> regressor (nominal and no-IP/SV ablation).
set -e
GPU=${1:-3}
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
D=$HOME/dihiggs-spin-20261006/code/analysis/dihiggs_spin
V1=$HOME/dihiggs-spin-20261006/runs/v1; E=$HOME/dihiggs-spin-20261006/runs/ent; mkdir -p $E/reco
cd $D
ls $HOME/dihiggs-spin-20261006/train_shower_U/trainHU_run_*.npz | grep -v _tmp_ | \
  xargs -P 12 -I{} sh -c 'b=$(basename {}); [ -f '$E'/reco/$b ] || nice -n 10 python3 reco.py {} --no-btag --outdir '$E'/reco > /dev/null'
echo "reco done $(date)"
OMP_NUM_THREADS=16 nice -n 10 python3 build_dataset.py --recodir $E/reco --procs trainHU --table-from $V1/train_dataset.npz --out $E/dataset_HU.npz
echo "dataset done $(date)"
PYG="env -i HOME=$HOME PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES=$GPU $HOME/tauspin-ss/NN/.venv-gpu/bin/python"
$PYG train_h.py --data $V1/train_dataset.npz --load $V1/hpred_partial_model.pt --apply $E/dataset_HU.npz --out $E/hpred_HU.npz
[ -f $E/hpred_noipsv_model.pt ] || $PYG train_h.py --data $V1/train_dataset.npz --drop ip_,sv_ --apply $E/dataset_HU.npz --out $E/hpred_noipsv.npz
echo "ent pipeline done $(date)"
