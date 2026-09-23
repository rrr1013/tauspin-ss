#!/usr/bin/env bash
cd ~/hz-spin-synthesis-20260923
R=56396
./scripts/run_s3.sh truth_exact 8 $R --visible truth --met exact &
./scripts/run_s3.sh reco_exact2 8 $R --visible reco --met exact --met-sigma 2 &
./scripts/run_s3.sh truth_recomet 8 $R --visible truth --met reco &
./scripts/run_s3.sh reco_recomet 8 $R --visible reco --met reco &
wait
./scripts/run_s3.sh truth_exact2 8 $R --visible truth --met exact --met-sigma 2 &
./scripts/run_s3.sh recopi_exact2 8 $R --visible reco_pi --met exact --met-sigma 2 &
./scripts/run_s3.sh recorho_exact2 8 $R --visible reco_rho --met exact --met-sigma 2 &
./scripts/run_s3.sh reco3pi_exact2 8 $R --visible reco_3pi --met exact --met-sigma 2 &
wait
touch artifacts/s3/DONE_X2
