#!/bin/bash
# HTCondor wrapper: shower one LHE chunk; args: proc lhe out seed
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
export LD_PRELOAD=$(gfortran -print-file-name=libgfortran.so.5)
cd $HOME/dihiggs-spin-20261006/code/gen
fk=""; [ "$1" = "ttlj" ] && fk="--fakes"
tmp=$(dirname $3)/_tmp_$(basename $3)
sf=""; [ "$1" = "hhU" ] && sf="--spinflat"
$HOME/dihiggs-spin-20261006/venv/bin/python pythia_shower.py --lhe $2 --proc $1 --out $tmp --seed $4 $fk $sf && mv $tmp $3
