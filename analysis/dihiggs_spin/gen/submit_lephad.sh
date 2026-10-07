#!/bin/bash
# tau_lep tau_had extension: shower all processes into shower_lh (tau-lepton flavour stored),
# one HTCondor job per 50k-event LHE chunk.  usage: submit_lephad.sh existing|new
source "$(dirname "$0")/env.sh"
CODE=$(cd "$(dirname "$0")" && pwd)
GEN=/tmp/rbaba-dihiggs/gen; L=$HOME/dihiggs-spin-20261006/lhe; S=$HOME/dihiggs-spin-20261006/shower_lh
mkdir -p $L $S $HOME/dihiggs-spin-20261006/logs
sub=$CODE/condor_shower_lh_$1.sub
cat > $sub <<EOS
universe   = vanilla
executable = $CODE/condor_shower.sh
arguments  = \$(proc) \$(lhe) \$(out) \$(seed)
output     = $HOME/dihiggs-spin-20261006/logs/shlh_\$(Cluster)_\$(Process).out
error      = $HOME/dihiggs-spin-20261006/logs/shlh_\$(Cluster)_\$(Process).err
log        = $HOME/dihiggs-spin-20261006/logs/shower_lh.log
request_cpus = 1
request_memory = 3000
queue proc, lhe, out, seed from (
EOS
i=0
add() { echo "$1, $2, $S/$(basename $2 .lhe.gz | sed "s/^hhU_/hhC_/").npz, $((9000+i+$3))" >> $sub; i=$((i+1)); }
if [ "$1" = existing ]; then
  for run in $GEN/ttll/Events/run_0[012]; do
    [ -f $L/ttll_$(basename $run)_000.lhe.gz ] || $PY $CODE/split_lhe.py $run/unweighted_events.lhe.gz $L/ttll_$(basename $run) 50000
  done
  for f in $L/hhU_run_*.lhe.gz; do add hhC $f 0; done
  for p in zbb ttll ttlj tth zh; do for f in $L/${p}_run_*.lhe.gz; do add $p $f 0; done; done
else
  until grep -q "all lephad generation done" /tmp/rbaba-dihiggs/gen_lephad.log; do sleep 120; done
  for p in ttlt ttljp tWll tWlj; do
    for run in $GEN/$p/Events/run_*; do
      $PY $CODE/split_lhe.py $run/unweighted_events.lhe.gz $L/${p}_$(basename $run) 50000
    done
    for f in $L/${p}_run_*.lhe.gz; do add $p $f 1000; done
  done
fi
echo ")" >> $sub
condor_submit $sub
