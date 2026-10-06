#!/bin/bash
# usage: gen_events.sh <name> <nruns> <nevents_per_run> <seed0>
set -e
name=$1; nruns=$2; nev=$3; seed0=$4
GEN=/tmp/rbaba-dihiggs/gen
CODE=$(cd "$(dirname "$0")" && pwd)
cd $GEN/$name
for i in $(seq 0 $((nruns-1))); do
  seed=$((seed0+i))
  python3 $CODE/run_cards.py Cards/run_card.dat $name $nev $seed
  sed -i 's/^\s*#\?\s*nb_core\s*=.*/nb_core = 16/; s/^\s*#\?\s*run_mode\s*=.*/run_mode = 2/' Cards/me5_configuration.txt
  ./bin/generate_events run_$(printf %02d $i) -f > $GEN/gen_${name}_$i.log 2>&1
  echo "$(date) done $name run $i"
done
