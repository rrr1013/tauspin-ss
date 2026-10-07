#!/bin/bash
# tau_lep tau_had: after all shower_lh files exist -> reco -> dataset -> classifiers (lxgpu02, nice 10).
S=$HOME/dihiggs-spin-20261006/shower_lh; L=$HOME/dihiggs-spin-20261006/runs/lh
D=$HOME/dihiggs-spin-20261006/code/analysis/dihiggs_spin
until [ $(ls $S | grep -v _tmp_ | grep -c npz) -ge 342 ]; do sleep 120; done
echo "showers complete $(date)"
bash $D/run_lephad_reco.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
cd $D
OMP_NUM_THREADS=16 nice -n 10 python3 build_lephad.py --recodir $L/reco --out $L/dataset.npz
echo "dataset done $(date)"
OMP_NUM_THREADS=16 nice -n 10 python3 classify_lephad.py --data $L/dataset.npz --out $L/cls.json
echo "classify done $(date)"
