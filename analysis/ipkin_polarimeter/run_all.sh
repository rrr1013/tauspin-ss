#!/bin/bash
# Posterior polarimeter for all 158,584 events of the spin-flat H(125)+jet dataset, four variants,
# in 4096-event chunks on lxgpu02 CPUs (nice, single-threaded workers).
#   ip        IP/SV + MET + unpolarised decay prior
#   ip_m125   same + Gaussian m_tautau prior (125, 3 GeV)
#   noip      MET + prior only
#   noip_m125 MET + prior + m_tautau prior
set -e
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
R=$HOME/ipkin-polarimeter-20261016
cd $R/code/analysis/ipkin_polarimeter
P=${1:-16}
N=158584
mkdir -p $R/out/chunks
for v in ip ip_m125 noip noip_m125; do
  for a in $(seq 0 4096 $((N-1))); do echo "$v $a"; done
done | xargs -P $P -L 1 sh -c '
  v=$0; a=$1; o='$R'/out/chunks/${v}_$(printf %06d $a).npz
  [ -f $o ] && exit 0
  opt=""; case $v in noip*) opt="--no-ip";; esac
  case $v in *_m125) opt="$opt --mass-prior 125,3";; esac
  nice -n 10 python3 posterior_h.py --start $a --stop $((a+4096)) --met-sigma 20.7 --S-side 32768 --M 96 --K 256 \
     $opt --seed 1 --out $o.tmp.npz > '$R'/logs/${v}_$a.log 2>&1 && mv $o.tmp.npz $o'
echo "all chunks done $(date)"
