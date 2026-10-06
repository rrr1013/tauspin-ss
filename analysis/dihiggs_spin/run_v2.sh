#!/bin/bash
# Spin-flat HH pipeline: reco -> dataset (v1 MMC table) -> regressor apply -> K score.
set -e
GPU=${1:-3}
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
cd $(dirname "$0")
V1=$HOME/dihiggs-spin-20261006/runs/v1; V2=$HOME/dihiggs-spin-20261006/runs/v2; mkdir -p $V2/reco_hhU
ls $HOME/dihiggs-spin-20261006/shower_hhU/hhU_run_*.npz | grep -v _tmp_ | \
  xargs -P 12 -I{} sh -c 'b=$(basename {}); [ -f '$V2'/reco_hhU/$b ] || nice -n 10 python3 reco.py {} --outdir '$V2'/reco_hhU > /dev/null'
echo "reco done $(date)"
OMP_NUM_THREADS=16 nice -n 10 python3 build_dataset.py --recodir $V2/reco_hhU --procs hhU --table-from $V1/dataset.npz --out $V2/dataset_hhU.npz
OMP_NUM_THREADS=16 nice -n 10 python3 build_dataset.py --recodir $V1/reco --procs hh --table-from $V1/dataset.npz --out $V2/dataset_hhreal.npz
echo "datasets done $(date)"
for t in hhU hhreal; do
  env -i HOME=$HOME PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES=$GPU $HOME/tauspin-ss/NN/.venv-gpu/bin/python train_h.py \
    --data $V1/train_dataset.npz --load $V1/hpred_partial_model.pt --apply $V2/dataset_$t.npz --out $V2/hpred_$t.npz
done
python3 k_model.py --data $V1/dataset.npz --apply $V2/dataset_hhU.npz $V2/dataset_hhreal.npz --model $V2/k_model.json
echo "v2 done $(date)"
