#!/bin/bash
# Extract H (v2), Z (v2 two-step) and Z (v3 full ME) from the *filtered* LHE (same vis-pT>18/|eta|<2.6 filter)
# for the CP-angle / transverse-correlation distributions.  100 subjobs each.
set -uo pipefail
D=/home/rbaba/meeting-followup-20260930/cpangle; mkdir -p $D/out; cd $D
PY=/home/rbaba/tauspin-ss/NN/.venv-gpu/bin/python
V2=/gpfs/fs5001/chen/mySamples/gen_higgs_91p18_v2; V3=/gpfs/fs5001/chen/mySamples/gen_higgs_91p18_v3
for tag in h_v2:$V2:999801 z_v2:$V2:999802 z_v3:$V3:999802; do
  IFS=: read name src dsid <<< "$tag"
  ls -d $src/test_${dsid}_sub*_n2000 | sort -V | head -100 > dirs_$name.txt
  split -n l/4 -d dirs_$name.txt chunk_${name}_
  for c in chunk_${name}_0*; do
    files=$(for d in $(cat $c); do ls $d/mc.TXT._00001.tar.gz; done)
    nice -n 10 $PY extract_lhe.py out/$c.npz $files > out/$c.log 2>&1 &
  done
done
wait; cat out/*.log; echo EXTRACT_DONE
