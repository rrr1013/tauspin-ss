#!/bin/bash
# split HH LHE runs (args: run names) onto NFS as hhU chunks and write a condor submit file
source "$(dirname "$0")/env.sh"
CODE=$(cd "$(dirname "$0")" && pwd)
L=$HOME/dihiggs-spin-20261006/lhe; S=$HOME/dihiggs-spin-20261006/shower_hhU; mkdir -p $L $S
sub=$CODE/condor_hhU_$(date +%H%M%S).sub
cat > $sub <<EOS
universe   = vanilla
executable = $CODE/condor_shower.sh
arguments  = \$(proc) \$(lhe) \$(out) \$(seed)
output     = $HOME/dihiggs-spin-20261006/logs/hhU_\$(Cluster)_\$(Process).out
error      = $HOME/dihiggs-spin-20261006/logs/hhU_\$(Cluster)_\$(Process).err
log        = $HOME/dihiggs-spin-20261006/logs/hhU.log
request_cpus = 1
request_memory = 3000
queue proc, lhe, out, seed from (
EOS
for r in "$@"; do
  $PY $CODE/split_lhe.py /tmp/rbaba-dihiggs/gen/hh/Events/$r/unweighted_events.lhe.gz $L/hhU_$r 50000
  for f in $L/hhU_${r}_*.lhe.gz; do b=$(basename $f .lhe.gz); n=${b//[^0-9]/}; echo "hhU, $f, $S/$b.npz, 9${n: -5}" >> $sub; done
done
echo ")" >> $sub
echo $sub
