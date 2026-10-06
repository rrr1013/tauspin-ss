# MadGraph process definitions for the di-Higgs spin projection (14 TeV, LO, sm).
# Taus are left undecayed and decayed by Pythia 8 with spin correlations from
# their mother (h, Z or W in the event record).
declare -A PROC
PROC[hh]="generate g g > h h [noborn=QCD]"
PROC[ttll]="generate p p > t t~, (t > b w+, w+ > ta+ vt), (t~ > b~ w-, w- > ta- vt~)"
PROC[ttlj]="generate p p > t t~, (t > b w+, w+ > ta+ vt), (t~ > b~ w-, w- > j j)
add process p p > t t~, (t > b w+, w+ > j j), (t~ > b~ w-, w- > ta- vt~)"
PROC[zbb]="generate p p > z b b~, z > ta+ ta-"
PROC[zh]="generate p p > z h, z > b b~"
PROC[tth]="generate p p > t t~ h"
