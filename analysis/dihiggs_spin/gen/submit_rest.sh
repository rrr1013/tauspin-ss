#!/bin/bash
# After MadGraph finishes: split the remaining LHE runs onto NFS and submit one HTCondor job per chunk.
source "$(dirname "$0")/env.sh"
CODE=$(cd "$(dirname "$0")" && pwd)
GEN=/tmp/rbaba-dihiggs/gen; L=$HOME/dihiggs-spin-20261006/lhe; S=$HOME/dihiggs-spin-20261006/shower
mkdir -p $L $S
until grep -q "done ttlj run 1" /tmp/rbaba-dihiggs/gen_all.log; do sleep 60; done
sub=$CODE/condor_shower.sub
cat > $sub <<EOS
universe   = vanilla
executable = $CODE/condor_shower.sh
arguments  = \$(proc) \$(lhe) \$(out) \$(seed)
output     = $HOME/dihiggs-spin-20261006/logs/shower_\$(Cluster)_\$(Process).out
error      = $HOME/dihiggs-spin-20261006/logs/shower_\$(Cluster)_\$(Process).err
log        = $HOME/dihiggs-spin-20261006/logs/shower.log
request_cpus = 1
request_memory = 3000
queue proc, lhe, out, seed from (
EOS
i=0
for proc in zbb zh tth ttlj; do
  for run in $GEN/$proc/Events/run_*; do
    r=$(basename $run)
    $PY $CODE/split_lhe.py $run/unweighted_events.lhe.gz $L/${proc}_${r} 50000
    for f in $L/${proc}_${r}_*.lhe.gz; do
      b=$(basename $f .lhe.gz); echo "$proc, $f, $S/$b.npz, $((5000+i))" >> $sub; i=$((i+1))
    done
  done
done
echo ")" >> $sub
echo "READY $sub"
echo "submitted $i jobs $(date)"
