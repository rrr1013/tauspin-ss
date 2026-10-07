#!/bin/bash
# tau_lep tau_had: reco every shower_lh file not yet processed (lxgpu02, nice 10, 12 parallel).
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
D=$HOME/dihiggs-spin-20261006/code/analysis/dihiggs_spin
S=$HOME/dihiggs-spin-20261006/shower_lh; O=$HOME/dihiggs-spin-20261006/runs/lh/reco
mkdir -p $O
cd $D
ls $S/*.npz | grep -v _tmp_ | while read f; do [ -f $O/$(basename $f) ] || echo $f; done | \
  xargs -P 12 -I{} sh -c 'nice -n 10 python3 reco_lephad.py {} --outdir '$O' > /dev/null 2>> '$O'/../reco_err.log'
echo "reco pass done $(date)"
