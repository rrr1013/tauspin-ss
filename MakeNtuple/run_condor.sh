#!/bin/bash
set -o pipefail
sample="$1"
input_list="$2"
output_group="${3:-}"
project_dir="/home/rbaba/tauspin-ss/MakeNtuple"
chunk_name=$(basename "$input_list")
chunk_id=${chunk_name%.txt}
work_dir="$project_dir/condor_work/${sample}_${chunk_id}_${RANDOM}"
if [ -n "$output_group" ]; then
    output_dir="$project_dir/outputs/$output_group/$sample"
else
    output_dir="$project_dir/outputs/$sample"
fi
mkdir -p "$work_dir" "$output_dir"
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source "$ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh" --quiet
cd "$project_dir"
asetup AnalysisBase,25.2.32
source build/x86_64-el9-gcc13-opt/setup.sh
set -e
cd "$work_dir"
python "$project_dir/source/TauSpinNtuple/share/runTauSpin.py" --input-list "$input_list" --max-events -1 --submit-dir submitDir
test -f submitDir/data-ANALYSIS/input.root
mv submitDir/data-ANALYSIS/input.root "$output_dir/${chunk_id}.root"
cd "$project_dir"
rm -rf "$work_dir"
