# fixed-partial-v3 handoff — 2026-07-30 21:29 JST

> [!IMPORTANT]
> This safe-stop handoff was fully executed by 2026-07-31 08:31 JST. The
> historical state below is retained as the restart record. The completed
> pipeline produced 94 COMPLETE and 2 FAIL HPO trials, selected trial 67
> using validation only, retrained from scratch for 50 epochs, selected
> epoch 31, evaluated the test split once, and generated all final figures.
> Final test AUC is 0.609141 and test loss is 0.677500. The authoritative
> completion audit is
> `NN/outputs/hpo/fixed-partial-v3-20260730-2100-high-throughput-hpo-v1/completion_audit.json`
> with SHA-256
> `c008fb355199bdb502d44f02d4a2809296371f5b409e634c26ddc43ef7bd4f81`
> and `approved: true`.

## Safe-stop state

- Work intentionally stopped after the H/Z HTCondor ntuple smoke.
- No full ntuple jobs were submitted.
- No dataset build, HPO, 50-epoch training, test evaluation, or final plotting was started.
- `condor_q rbaba`: zero jobs.
- `lxgpu02`: GPUs 0–7 all 0% utilization, 1 MiB used; no compute processes.
- No matching audit/preflight/ntuple/NN processes remain.
- No commit or push was made.
- Existing uncommitted high-throughput work and the user's `.gitignore` change were preserved.

## Repository state

Repository: `/home/rbaba/tauspin-ss`

HEAD: `4b7708d13bcd52b9e8abea8dab8381330e3f2f42`

Known working-tree state at safe stop:

```text
 M .gitignore
 M NN/dataset.py
 M NN/final_hpo_controller.py
 M NN/final_hpo_worker.py
 M NN/hpo_utils.py
 M NN/model.py
 M NN/train.py
?? MakeNtuple/snapshots/
?? NN/finalize_v3_50epoch.py
?? NN/select_v3_hpo.py
?? NN/tests/
?? NN/v3_retrain_50epoch.py
```

Do not discard or reset these changes. Do not commit or push unless the user changes the instruction.

## Frozen input snapshot

Snapshot: `fixed-partial-v3-20260730-2100`

Directory:

```text
/home/rbaba/tauspin-ss/MakeNtuple/snapshots/fixed-partial-v3-20260730-2100
```

The candidate set is explicit and does not use a later glob:

- H: 499 DAOD files, sub1–297 and sub299–500; sub298 excluded because its internal EVNT→HITS failed and no DAOD existed at cutoff.
- Z: 489 DAOD files, sub1–489; sub490–500 excluded because their production jobs were still running at the 21:00 cutoff.
- H raw entries: 997,943.
- Z raw entries: 977,940.
- Total raw entries: 1,975,883.
- Total input size: H 35,911,802,478 bytes; Z 32,932,550,749 bytes.
- All 988 candidates passed ROOT open, zombie, `CollectionTree`, production `jobReport.json`, reported size/entry/GUID, and SHA-256 guards.

Authoritative artifacts:

```text
snapshot_manifest.json  SHA-256 832bef0ec3f69fd52dcbec91927adcc7cb2de9925f6bd6efbba688502a568019
audit_report.json       SHA-256 efcda6927c51e588f4d0704d3a414ed26b675ed42e8a30a23a4ea7be73b445a4
preflight_report.json   SHA-256 bfa90e9f41cdb924927fb57949860367ac13cf8a30420167d2031875acef430d
```

`preflight_report.json` passed 988/988 immediately before the smoke. Repeat preflight into a new report immediately before full submission:

```sh
cd /home/rbaba/tauspin-ss
python3 MakeNtuple/snapshots/fixed-partial-v3-20260730-2100/preflight_snapshot.py \
  --manifest MakeNtuple/snapshots/fixed-partial-v3-20260730-2100/snapshot_manifest.json \
  --output MakeNtuple/snapshots/fixed-partial-v3-20260730-2100/preflight_full_report.json
```

Proceed only if `submission_allowed` is true and failed count is zero.

## Completed HTCondor smoke

Clusters:

- H: `11557695.0`, exit 0, EventLoop real time 88 s, stderr 0 bytes.
- Z: `11557696.0`, exit 0, EventLoop real time 82 s, stderr 0 bytes.

Audit:

- H output: 10,002 `tauspin` entries, 84 branches, `truth_boson_pdgId={25}`, no non-finite float branch.
- Z output: 10,175 `tauspin` entries, 84 branches, `truth_boson_pdgId={23}`, no non-finite float branch.
- H/Z branch schema is identical.
- Full machine-readable result: `smoke_audit.json`.

Smoke outputs are isolated under:

```text
/home/rbaba/tauspin-ss/MakeNtuple/outputs/fixed-partial-v3-20260730-2100-smoke
```

## Full ntuple continuation

After a fresh passing preflight:

```sh
cd /home/rbaba/tauspin-ss/MakeNtuple/snapshots/fixed-partial-v3-20260730-2100
condor_submit submit_H.sub
condor_submit submit_Z.sub
```

This submits H 50 chunks and Z 49 chunks, 10 DAOD files per chunk except the final partial chunks. Outputs go to:

```text
/home/rbaba/tauspin-ss/MakeNtuple/outputs/fixed-partial-v3-20260730-2100/H
/home/rbaba/tauspin-ss/MakeNtuple/outputs/fixed-partial-v3-20260730-2100/Z
```

After completion, audit all 99 ROOT outputs before dataset construction: open/zombie, `tauspin`, identical 84-branch schema, H/Z PDG ID, finite values and valid shapes, input-chunk to output one-to-one coverage, output SHA-256, no held/failed jobs, and stderr/exit codes. Record any exclusion explicitly; do not silently drop a chunk.

## Dataset continuation

Use the CPU environment for uproot:

```sh
cd /home/rbaba/tauspin-ss/NN
.venv/bin/python make_pt_matching_manifest.py \
  --h-pattern '../MakeNtuple/outputs/fixed-partial-v3-20260730-2100/H/*.root' \
  --z-pattern '../MakeNtuple/outputs/fixed-partial-v3-20260730-2100/Z/*.root' \
  --output-dir outputs/data-preparation/fixed-partial-v3-20260730-2100-ptmatched20 \
  --bin-width 20 --overflow-edge 1000 --matching-seed 42

.venv/bin/python build_dataset.py \
  --h-pattern '../MakeNtuple/outputs/fixed-partial-v3-20260730-2100/H/*.root' \
  --z-pattern '../MakeNtuple/outputs/fixed-partial-v3-20260730-2100/Z/*.root' \
  --output-dir processed/fixed-partial-v3-20260730-2100-ptmatched20-relative-v3 \
  --selection-manifest outputs/data-preparation/fixed-partial-v3-20260730-2100-ptmatched20/pt_matching_manifest.json \
  --feature-set absolute-plus-parent-relative-v3
```

The implemented split identity is `(sample, ntuple chunk basename, file-local entry index)` with BLAKE2b seed 42; it is not physical `eventNumber`. Matching is performed after split, independently within each split, in 20 GeV bins with `>=1000 GeV` overflow and matching seed 42.

Before HPO, audit feature dimensions EVENT 13, TAU 10, TRACK 19, PFO 10; train-only normalization; row-identity disjointness; matching before/after counts; retention; pT-only AUC; shard counts; labels; offsets; sides; decay modes; and all finite values.

## New HPO / final-50 implementation

Implemented but not run on a real v3 dataset:

- `NN/final_hpo_controller.py`: requires absent/empty study directory, `load_if_exists=False`, binds processed metadata, selection manifest, and snapshot manifest SHA-256.
- `NN/final_hpo_worker.py`: rechecks the three hashes before trial-directory creation and records them in config/checkpoint/result.
- `NN/select_v3_hpo.py`: validation-only ranking, AUC gap `<0.001`, then minimum validation loss, canonical parameters, trial number; never opens test.
- `NN/v3_retrain_50epoch.py`: starts a new high-throughput run, forces 50 epochs, objective start 2, early stopping disabled past the horizon, and verifies all 50 epochs/test-unread.
- `NN/finalize_v3_50epoch.py`: validates the selected center checkpoint and reload AUC/loss first, then creates the test loader once; batch 512, 12 workers, event partition, TF32/compile. Produces 50-epoch curves and test ROC.
- `NN/tests/test_v3_selection.py`: two selection tests.

Verified without GPU:

- Remote Python AST and all five `--help` invocations pass.
- Selection unit tests: 2 passed.
- Non-empty study directory is rejected.
- A hash mismatch aborts before worker trial-directory creation.

Still required before real HPO:

1. Independently review the new v3 scripts, especially finalizer artifact schema and test-late-open invariant.
2. Run the existing high-throughput regression suite.
3. Stage the new processed dataset on `lxgpu02` local storage and run a one-GPU 3-epoch smoke.
4. Recompute the deadline gate from measured epoch time.
5. Check `nvidia-smi` immediately before allocation and use only truly idle GPUs, at most five.
6. Create a unique absent study/output directory. Do not use any fixed-partial-v2 Optuna DB, trial, or checkpoint.

## Vault / review state

The current Analysis Note is:

```text
$ATHENA_VAULT/10_Projects/HE/ATLAS/tauspin/10_Note.md
```

Section: `fixed-partial-v3解析計画`

It records the plan, Evidence Ledger, assumptions, independent review and Open Majors. The plan was `Pilot-ready / Execution-readyではない` before the smoke. Update it with the passing manifest/preflight/smoke evidence and only mark Execution-ready after the new HPO/test-isolation code receives an independent review and the one-GPU dataset smoke/time gate passes.

No final result has been written to `00_Home`, the daily Log, Transformer reference, or Tasks because the requested analysis is not complete.
