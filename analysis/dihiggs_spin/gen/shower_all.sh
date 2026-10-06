#!/bin/bash
# Split every finished LHE run into 100k-event chunks and shower them, 16 at a time.
source "$(dirname "$0")/env.sh"
CODE=$(cd "$(dirname "$0")" && pwd)
GEN=/tmp/rbaba-dihiggs/gen; SH=/tmp/rbaba-dihiggs/shower/prod; mkdir -p $SH/lhe
for proc in "$@"; do
  for run in $GEN/$proc/Events/run_*; do
    r=$(basename $run)
    [ -f $run/unweighted_events.lhe.gz ] || continue
    ls $SH/lhe/${proc}_${r}_*.lhe.gz >/dev/null 2>&1 || $PY $CODE/split_lhe.py $run/unweighted_events.lhe.gz $SH/lhe/${proc}_${r} 100000
  done
done
jobs_list=()
for proc in "$@"; do for f in $SH/lhe/${proc}_run_*.lhe.gz; do jobs_list+=("$proc $f"); done; done
i=0
for item in "${jobs_list[@]}"; do
  set -- $item; proc=$1; f=$2; b=$(basename $f .lhe.gz)
  [ -f $SH/$b.npz ] && continue
  fk=""; [ "$proc" = "ttlj" ] && fk="--fakes"
  nice -n 10 $PY $CODE/pythia_shower.py --lhe $f --proc $proc --out $SH/$b.npz --seed $((1000+i)) $fk > $SH/$b.log 2>&1 &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge ${NPAR:-12} ]; do sleep 10; done
done
wait
echo "all showers done $(date)"
