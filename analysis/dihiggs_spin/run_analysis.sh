#!/bin/bash
# reco -> dataset (MMC, features) -> h regressor (GPU, trained on the equal-mass
# H/Z(125) sample, applied to the analysis sample) -> classifiers, on lxgpu02.
# usage: run_analysis.sh <tag> [gpu_index] [stage]   stage: all | reco | rest
set -e
TAG=$1; GPU=${2:-3}; STAGE=${3:-all}
CODE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/dihiggs-spin-20261006/runs/$TAG; mkdir -p $OUT/reco $OUT/train_reco
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
cd $CODE
if [ $STAGE = all ] || [ $STAGE = reco ]; then
  ls /tmp/rbaba-dihiggs/shower/prod/*.npz $HOME/dihiggs-spin-20261006/shower/*_run_*.npz 2>/dev/null | grep -v _tmp_ | \
    xargs -P 12 -I{} sh -c 'b=$(basename {}); [ -f '$OUT'/reco/$b ] || nice -n 10 python3 reco.py {} --outdir '$OUT'/reco > /dev/null'
  ls $HOME/dihiggs-spin-20261006/train_shower/*_run_*.npz 2>/dev/null | grep -v _tmp_ | \
    xargs -P 12 -I{} sh -c 'b=$(basename {}); [ -f '$OUT'/train_reco/$b ] || nice -n 10 python3 reco.py {} --no-btag --outdir '$OUT'/train_reco > /dev/null'
  echo "reco done $(date)"
fi
[ $STAGE = reco ] && exit 0
OMP_NUM_THREADS=16 nice -n 10 python3 build_dataset.py --recodir $OUT/reco --out $OUT/dataset.npz
OMP_NUM_THREADS=16 nice -n 10 python3 build_dataset.py --recodir $OUT/train_reco --procs trainH,trainZ --out $OUT/train_dataset.npz
echo "datasets done $(date)"
env -i HOME=$HOME PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES=$GPU $HOME/tauspin-ss/NN/.venv-gpu/bin/python \
  train_h.py --data $OUT/train_dataset.npz --apply $OUT/dataset.npz --out $OUT/hpred.npz
echo "h regressor done $(date)"
for seed in 0 1 2; do
  nice -n 10 python3 classify.py --data $OUT/dataset.npz --hpred $OUT/hpred.npz --out $OUT/classify_s$seed.json --seed $seed
done
python3 projection.py $OUT
echo "analysis done $(date)"
