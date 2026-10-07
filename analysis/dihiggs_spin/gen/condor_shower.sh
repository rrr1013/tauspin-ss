#!/bin/bash
# HTCondor wrapper: shower one LHE chunk; args: proc lhe out seed
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
export LD_PRELOAD=$(gfortran -print-file-name=libgfortran.so.5)
cd $HOME/dihiggs-spin-20261006/code/gen
fk=""; case "$1" in ttlj|ttljp|tWlj) fk="--fakes";; esac
tmp=$(dirname $3)/_tmp_$(basename $3)
sf=""; [ "$1" = "hhU" ] && sf="--spinflat"
# hhC: the spin-flat HH LHE showered with the full spin correlations (TauDecays:mode = 4)
[ "$1" = "hhC" ] && set -- hh "$2" "$3" "$4"
$HOME/dihiggs-spin-20261006/venv/bin/python pythia_shower.py --lhe $2 --proc $1 --out $tmp --seed $4 $fk $sf && mv $tmp $3
