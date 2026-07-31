#!/bin/bash
set -u

study=/home/rbaba/tauspin-ss/NN/outputs/hpo/fixed-partial-v3-20260730-2100-high-throughput-hpo-v1
audit="$study/completion_audit.json"
stderr_log=/home/rbaba/tauspin-ss/MakeNtuple/snapshots/fixed-partial-v3-20260730-2100/after_hpo.stderr.log

while true; do
    if [ -f "$audit" ]; then
        echo "COMPLETE $(date --iso-8601=seconds)"
        exit 0
    fi
    if ! tmux has-session -t tauspin_v3_after_hpo 2>/dev/null; then
        echo "AFTER_HPO_STOPPED_WITHOUT_AUDIT $(date --iso-8601=seconds)"
        tail -n 80 "$stderr_log"
        exit 1
    fi
    python3 -c '
import datetime
import json
from pathlib import Path

path = Path("/home/rbaba/tauspin-ss/NN/outputs/hpo/fixed-partial-v3-20260730-2100-high-throughput-hpo-v1/controller_status.json")
if path.exists():
    document = json.loads(path.read_text())
    print(
        datetime.datetime.now().astimezone().isoformat(),
        document["trial_state_counts"],
        "active",
        [item["trial_number"] for item in document["active"]],
        flush=True,
    )
else:
    print("waiting for controller status", flush=True)
'
    sleep 60
done
