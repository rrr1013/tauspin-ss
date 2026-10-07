#!/bin/bash
# tau_lep tau_had extension: create and generate the new background processes (lxgpu02, nice 10).
set -e
CODE=$(cd "$(dirname "$0")" && pwd)
LOG=/tmp/rbaba-dihiggs/gen_lephad.log
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
for p in ttlt ttljp tWll tWlj; do bash $CODE/make_proc.sh $p >> $LOG 2>&1; done
bash $CODE/gen_events.sh ttlt 8 500000 7100 >> $LOG 2>&1
bash $CODE/gen_events.sh ttljp 6 500000 7200 >> $LOG 2>&1
bash $CODE/gen_events.sh tWll 2 500000 7300 >> $LOG 2>&1
bash $CODE/gen_events.sh tWlj 4 500000 7400 >> $LOG 2>&1
echo "$(date) all lephad generation done" >> $LOG
