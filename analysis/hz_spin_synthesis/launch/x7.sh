#!/usr/bin/env bash
export OMP_NUM_THREADS=2
cd ~/hz-spin-synthesis-20260923/scripts
PY=~/tauspin-ss/NN/.venv-gpu/bin/python
B=~/hz-beyond-ceiling-20260923/artifacts
O=../artifacts/x7_readout
(CUDA_VISIBLE_DEVICES=2 $PY -u s7_additive_readout.py --source exact --model additive --name exact_add --output-dir $O > ../logs/x7_exact_add.log 2>&1;
 CUDA_VISIBLE_DEVICES=2 $PY -u s7_additive_readout.py --source exact --model full --name exact_full --output-dir $O > ../logs/x7_exact_full.log 2>&1) &
(CUDA_VISIBLE_DEVICES=3 $PY -u s7_additive_readout.py --source pred --pred-dirs $B/p9/baseline_current --name base_add --output-dir $O > ../logs/x7_base_add.log 2>&1;
 CUDA_VISIBLE_DEVICES=3 $PY -u s7_additive_readout.py --source pred --pred-dirs $B/p11/full22_s42 $B/p11/full22_s43 --name f22ens_add --output-dir $O > ../logs/x7_f22_add.log 2>&1) &
wait
touch $O/DONE
