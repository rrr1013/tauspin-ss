#!/bin/bash
# reco -> dataset (MMC, features) -> h regressor (GPU) -> classifiers, on lxgpu02.
# usage: run_analysis.sh <tag> [gpu_index]
set -e
TAG=$1; GPU=${2:-3}
CODE=$(cd "$(dirname "$0")" && pwd)
SH=/tmp/rbaba-dihiggs/shower/prod
OUT=$HOME/dihiggs-spin-20261006/runs/$TAG; mkdir -p $OUT/reco
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
cd $CODE
ls $SH/*.npz | grep -v test | xargs -P 12 -I{} sh -c 'b=$(basename {}); [ -f '$OUT'/reco/$b ] || nice -n 10 python3 reco.py {} --outdir '$OUT'/reco > /dev/null'
echo "reco done $(date)"
OMP_NUM_THREADS=16 nice -n 10 python3 build_dataset.py --recodir $OUT/reco --out $OUT/dataset.npz
echo "dataset done $(date)"
CUDA_VISIBLE_DEVICES=$GPU $HOME/tauspin-ss/NN/.venv-gpu/bin/python train_h.py --data $OUT/dataset.npz --out $OUT/hpred.npz
echo "h regressor done $(date)"
for seed in 0 1 2; do
  nice -n 10 python3 classify.py --data $OUT/dataset.npz --hpred $OUT/hpred.npz --out $OUT/classify_s$seed.json --seed $seed
done
echo "analysis done $(date)"
