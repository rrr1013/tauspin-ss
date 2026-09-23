#!/usr/bin/env bash
export OMP_NUM_THREADS=2
cd ~/hz-spin-synthesis-20260923
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
B=~/hz-beyond-ceiling-20260923/artifacts
O=artifacts/x1_readout
run() { CUDA_VISIBLE_DEVICES=$1 $PY -u scripts/s1_readout.py --name $2 --target $3 --output-dir $O --pred-dirs ${@:4} > logs/x1_$2.log 2>&1; }
(run 0 base_hard hard $B/p9/baseline_current; run 0 base_soft soft $B/p9/baseline_current) &
(run 1 f22s42_hard hard $B/p11/full22_s42; run 1 f22s42_soft soft $B/p11/full22_s42) &
(run 2 f22ens_hard hard $B/p11/full22_s42 $B/p11/full22_s43; run 2 f22ens_soft soft $B/p11/full22_s42 $B/p11/full22_s43) &
(run 3 f22s43_soft soft $B/p11/full22_s43; run 3 ideal_soft soft $B/p11/idealip22_s42) &
wait
touch $O/DONE
