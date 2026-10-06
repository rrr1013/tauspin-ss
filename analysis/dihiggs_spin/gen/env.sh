source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
export LD_PRELOAD=$(gfortran -print-file-name=libgfortran.so.5)
PY=~/dihiggs-spin-20261006/venv/bin/python
