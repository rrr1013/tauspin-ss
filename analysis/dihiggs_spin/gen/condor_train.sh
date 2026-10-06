#!/bin/bash
# HTCondor wrapper: gen_train.py chunk; args: proc n seed outdir
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
export LD_PRELOAD=$(gfortran -print-file-name=libgfortran.so.5)
cd $HOME/dihiggs-spin-20261006/code/gen
$HOME/dihiggs-spin-20261006/venv/bin/python gen_train.py --proc $1 --n $2 --seed $3 --out _tmp_$1_$3.npz && mv _tmp_$1_$3.npz $4/$1_run_$3.npz
