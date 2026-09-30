#!/bin/bash
# Mass-width control for v3 Z: rescale Z (v2 and v3) tau-pair mass to the H value (91.1872 GeV),
# rerun Chen's truth/smear chain and prepare_simple (same code/seeds as the 2026-09-21 v3 rerun),
# then retrain the four arms whose inputs can form m_tautau.  Log: logs/zmass.log
set -uo pipefail
D=/home/rbaba/meeting-followup-20260930/zmass
PY=/home/rbaba/tauspin-ss/NN/.venv-gpu/bin/python
C=/home/rbaba/tauspin-zspin-v3-20260921/NN/zspin_v3_20260921/chain
W=/home/rbaba/tauspin-zspin-v3-20260921/NN/simple_smearing_20260910
V2=/gpfs/fs5001/chen/mySamples/gen_higgs_91p18_v2
V3=/gpfs/fs5001/chen/mySamples/gen_higgs_91p18_v3
export PYTHONPATH=/home/rbaba/spin-statistics-20260906-python-extra OMP_NUM_THREADS=4
cd $D; mkdir -p logs training
L=$D/logs/zmass.log
echo "START $(date +%s) $(hostname)" >> $L
for tag in v2r v3r; do
  src=$V2; [ $tag = v3r ] && src=$V3
  $PY $D/lhe_mass_rescale.py --src $src --dst $D/lhe_$tag --dsid 999802 --mass 91.1872 --workers 24 > logs/rescale_$tag.log 2>&1
  echo "RESCALE_DONE $tag rc=$? $(date +%s)" >> $L
  mkdir -p chain_$tag
  ( cd chain_$tag && LHE_SRC_H=$V2 LHE_SRC_Z=$D/lhe_$tag TR_DIR=$D/chain_$tag $PY $C/make_truth_ntuple_from_lhe.py && \
    TR_DIR=$D/chain_$tag SRC_H=h_lhetruth.root SRC_Z=z_lhetruth.root OUT_TAG=_lhe $PY $C/make_smeared_ntuple.py ) > logs/chain_$tag.log 2>&1
  echo "CHAIN_DONE $tag rc=$? $(date +%s)" >> $L
  SIMPLE_ROOT_DIR=$D/chain_$tag SIMPLE_LHE_ROOT_H=$V2 SIMPLE_LHE_ROOT_Z=$D/lhe_$tag \
    $PY $W/prepare_simple.py --output-dir $D/data_$tag > logs/prepare_$tag.log 2>&1
  echo "PREPARE_DONE $tag rc=$? $(date +%s)" >> $L
done
cd $W
arm() { # gpu tag arm
  CUDA_VISIBLE_DEVICES=$1 $PY train_campaign.py --arm $3 --data-dir $D/data_$2 --output-dir $D/training/$2/$3 > $D/logs/$2_$3.log 2>&1
  echo "ARM_DONE $2 $3 rc=$? $(date +%s)" >> $L
}
( arm 0 v3r truth_transformer; arm 0 v3r reco_baseline ) &
arm 1 v3r truth_nu_transformer &
arm 2 v3r reco_transformer &
( arm 3 v2r truth_transformer; arm 3 v2r reco_baseline ) &
arm 4 v2r truth_nu_transformer &
arm 5 v2r reco_transformer &
wait
echo "ZMASS_DONE $(date +%s)" >> $L
