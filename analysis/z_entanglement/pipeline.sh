#!/bin/zsh
# wait for ICEPP, extract raw v3 LHE on lxgpu02, copy back, run the analysis. Log: data/pipeline.log
set -u
HERE=${0:A:h}; cd $HERE; mkdir -p data results
exec >> data/pipeline.log 2>&1
echo "START $(date -u)"
for i in $(seq 1 180); do
  ssh -o BatchMode=yes -o ConnectTimeout=10 icepp-gpu true 2>/dev/null && break
  ssh -o BatchMode=yes -o ConnectTimeout=10 icepp true 2>/dev/null && break
  sleep 120
done
HOST=icepp-gpu; ssh -o BatchMode=yes -o ConnectTimeout=10 icepp-gpu true 2>/dev/null || HOST=icepp
echo "UP via $HOST $(date -u)"
RD=/home/rbaba/z-entanglement-20260927
SRC=/gpfs/fs5001/chen/mySamples/gen_higgs_91p18_v3
ssh $HOST "mkdir -p $RD/out" && scp -q extract_lhe.py $HOST:$RD/ || { echo FAIL_COPY; exit 1; }
ssh $HOST "bash -s" <<REMOTE
cd $RD; uptime; nproc
PY=""
for p in python3 \$HOME/tauspin*/.venv*/bin/python \$HOME/*/.venv*/bin/python; do
  if [ -x "\$(command -v \$p)" ] && \$p -c 'import numpy' 2>/dev/null; then PY=\$p; break; fi
done
echo "PY=\$PY"; [ -n "\$PY" ] || exit 2
ls -d $SRC/test_999802_sub*_n2000 | sort -V | head -160 > dirs.txt
wc -l dirs.txt; ls $SRC/test_999802_sub1_n2000/ > listing_sub1.txt
cp \$(ls $SRC/test_999802_sub1_n2000/*banner* 2>/dev/null | head -1) banner_sub1.txt 2>/dev/null
split -n l/8 -d dirs.txt chunk_
for c in chunk_0*; do
  files=\$(for d in \$(cat \$c); do ls \$d/*raw*.lhe.gz 2>/dev/null | head -1; done)
  nice -n 10 \$PY extract_lhe.py out/\$c.npz \$files > out/\$c.log 2>&1 &
done
wait; cat out/*.log; ls -la out
REMOTE
echo "REMOTE rc=$? $(date -u)"
scp -q "$HOST:$RD/out/*.npz" data/ && scp -q "$HOST:$RD/{listing_sub1.txt,banner_sub1.txt}" data/ 2>/dev/null
ls -la data
python3 run_analysis.py results/v3 data/chunk_0*.npz > results/v3_stdout.txt 2>&1; echo "ANALYSIS rc=$?"
echo "DONE $(date -u)"
