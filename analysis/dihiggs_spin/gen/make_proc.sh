#!/bin/bash
# usage: make_proc.sh <name> ; creates process directory under $GEN
set -e
source "$(dirname "$0")/procs.sh"
name=$1
MG=/tmp/rbaba-dihiggs/MG5/bin/mg5_aMC
GEN=/tmp/rbaba-dihiggs/gen
mkdir -p $GEN && cd $GEN
cat > proc_$name.mg5 <<EOC
set automatic_html_opening False
set nb_core 16
import model sm
define p = g u c d s b u~ c~ d~ s~ b~
define j = g u c d s b u~ c~ d~ s~ b~
${PROC[$name]}
output $name -f
EOC
$MG proc_$name.mg5 > make_$name.log 2>&1
echo "made $name"
