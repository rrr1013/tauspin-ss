#!/usr/bin/env bash
export OMP_NUM_THREADS=2
cd ~/hz-spin-synthesis-20260923
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
B=~/hz-beyond-ceiling-20260923/artifacts/p11
A=artifacts/p11
O=artifacts/x10_readout
while [ ! -f $A/DONE_X5 ]; do sleep 60; done
run() { CUDA_VISIBLE_DEVICES=$1 $PY -u scripts/s1_readout.py --name $2 --target $3 --output-dir $O --pred-dirs ${@:4} > logs/x10_$2.log 2>&1; }
(run 0 f22s44_hard hard $A/full22_s44; run 0 f22ens4_hard hard $B/full22_s42 $B/full22_s43 $A/full22_s44 $A/full22_s45) &
(run 1 f22s45_hard hard $A/full22_s45; run 1 f22ens4_soft soft $B/full22_s42 $B/full22_s43 $A/full22_s44 $A/full22_s45) &
wait
touch $O/DONE_A
while [ ! -f $A/DONE_X8 ]; do sleep 60; done
(run 0 v3s42_hard hard $A/v3_s42; run 0 v3ens_hard hard $A/v3_s42 $A/v3_s43) &
(run 1 v3s43_hard hard $A/v3_s43; run 1 all6_hard hard $B/full22_s42 $B/full22_s43 $A/full22_s44 $A/full22_s45 $A/v3_s42 $A/v3_s43) &
wait
touch $O/DONE_B
