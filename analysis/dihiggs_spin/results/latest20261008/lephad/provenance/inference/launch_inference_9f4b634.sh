#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
nice -n 10 /home/rbaba/dihiggs-spin-20261006/venv/bin/python -u /home/rbaba/dihiggs-latest-20261008/reevaluate_9f4b634.py
result_code=$?
if [ "$result_code" = 0 ]; then
 nice -n 10 /home/rbaba/dihiggs-spin-20261006/venv/bin/python /home/rbaba/dihiggs-latest-20261008/source-inference-9f4b634/analysis/dihiggs_spin/latest_lephad_plots.py --prepared /home/rbaba/dihiggs-latest-20261008/prepared_latest_slt_hilo/prepared.npz --run /home/rbaba/dihiggs-latest-20261008/lephad-inference-9f4b634 --out /home/rbaba/dihiggs-latest-20261008/lephad-inference-9f4b634/figures
 result_code=$?
fi
printf "%s\n" "$result_code" > /home/rbaba/dihiggs-latest-20261008/lephad-inference-9f4b634/exit_code.txt
date -u +%Y-%m-%dT%H:%M:%SZ > /home/rbaba/dihiggs-latest-20261008/lephad-inference-9f4b634/finished_utc.txt
exit "$result_code"
