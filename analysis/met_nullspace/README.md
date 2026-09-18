# MET-constrained null space of tau-pair reconstruction

Given both visible four-momenta, both tau mass shells and the neutrino
transverse sum, the neutrino system still has two free parameters.  This
directory builds that solution set exactly and asks whether the pair statistic
`T = h-_n h+_n + h-_r h+_r - h-_k h+_k` survives marginalising it better than a
per-side point estimate does, and how well any estimator could possibly do.

Run note: `AthenaVault/10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-met-nullspace-pair-observable-20260919/`.

## Files

| file | what it is |
|---|---|
| `metspace.py` | the solution set: allowed region, sheets, the two measures, the estimators |
| `test_metspace.py` | kinematic closure; needs no ROOT and no cohort |
| `run_m1_moments.py` | per-event polarimeter moments on the solution set (ICEPP) |
| `run_m2_report.py` | estimators, AUCs, bootstrap, MET-resolution curve |
| `run_m3_figures.py` | figures |

`nullspace.py` from `../azimuth_nullspace` is imported unmodified, and so is the
existing `HybridPolarimeter`.  Nothing here re-trains or re-simulates anything.

## Reproducing

On ICEPP, with `nullspace.py` and this directory's scripts side by side:

```sh
S=/home/rbaba/azimuth-nullspace-20260918/artifacts/truth_surface.npz
C="--surface $S --keep 128 --proposal-lattice 64 --row-chunk 1500"
python3 run_m1_moments.py $C --output artifacts/met   --tag met
python3 run_m1_moments.py $C --output artifacts/indep --tag indep --independent
for SIG in 5 10 15 20 30; do
  python3 run_m1_moments.py $C --output artifacts/sigma$SIG --tag sigma$SIG --met-sigma $SIG
done
python3 run_m1_moments.py --surface $S --keep 512 --proposal-lattice 96 \
        --row-chunk 750 --output artifacts/met_hi --tag met_hi
```

Then locally, with the moment files under `<moments>/<tag>/<tag>_moments.npz`:

```sh
python3 run_m2_report.py --moments <moments> --output <out> --bootstrap 2000
python3 run_m3_figures.py --moments <moments> --surface truth_surface.npz \
        --report <out>/report.json --output <out>
```

The seed fixes the draws, so re-running with extra accumulators reproduces the
existing numbers exactly; that was checked (max absolute difference `0.0` on
every shared array).

## Results

`analysis/outputs/met-nullspace-20260919/`: `report.json` and seven figures.
The per-event moment arrays stay on ICEPP under `~/met-nullspace-20260919/`.
