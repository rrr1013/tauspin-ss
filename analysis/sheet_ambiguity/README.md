# The discrete part of the tau-pair null space

Above every allowed transverse neutrino momentum, each side's tau mass shell is
a quadratic with two roots, so the MET-constrained solution set built in
`../met_nullspace` is four sheets rather than one.  That run found that about
40% of the variance of the pair statistic `T` sits *between* those sheets.  This
one asks what resolving the discrete choice is worth, how it compares with
resolving the continuous coordinate, and how well the tau flight direction would
have to be measured to do either.

Run note: `AthenaVault/10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-sheet-ambiguity-20260920/`.

## Files

| file | what it is |
|---|---|
| `sheetspace.py` | the sheet label, the tau direction of a hypothesis, the direction likelihood |
| `test_sheetspace.py` | kinematic closure; needs no ROOT and no cohort |
| `run_s1_moments.py` | per-event moments under every conditioning and every resolution (ICEPP) |
| `run_s2_report.py` | the oracle ladder, headroom shares, resolution curves |
| `run_s3_figures.py` | figures |
| `run_s4_vertex.py` | decay length and the direction resolution a vertex measurement gives |
| `run_s5_sensitivity.py` | root-separation cuts, side symmetry, gain against geometry |

`nullspace.py` from `../azimuth_nullspace`, `metspace.py` from `../met_nullspace`
and the existing `HybridPolarimeter` are imported unmodified.  Nothing here
re-trains or re-simulates anything, and the test split is never opened.

## The conditionings

| symbol | what is known | what is left free |
|---|---|---|
| `O0` | nothing | the continuous 2D coordinate and the four sheets |
| `O1` | the correct mass-shell root on both sides | the continuous coordinate |
| `O2a` / `O2b` | the correct root on one side | the continuous coordinate and two sheets |
| `O3` | the truth transverse neutrino momentum | the four sheets |
| `O4` | the truth neutrinos | nothing |

`O1` and `O3` are complementary: knowing both is `O4`, knowing neither is `O0`.
They are decomposition devices, not measurement proposals.

## Two resolution curves

`run_s1_moments.py` reweights every hypothesis by an idealised tau-direction
measurement, which uses the measurement for everything it is worth.  That is
exact Bayes but it is limited by how finely the Monte Carlo draws sample the
region, so each point carries its effective sample size.

`run_s2_report.py` also builds the posterior over the four sheets alone and
mixes the sheet-conditional moments.  Moments of a mixture are mixtures of
moments, so that curve is exact at any resolution.  Its `sigma -> 0` limit must
reproduce `O1` and its `sigma -> infinity` limit must reproduce `O0`; both are
checked and reported.

The difference between the two curves is what a direction measurement buys
beyond choosing the root.

## Reproducing

On ICEPP, with all scripts side by side in one directory:

```sh
S=/home/rbaba/azimuth-nullspace-20260918/artifacts/truth_surface.npz
A=~/sheet-ambiguity-20260920/artifacts
python3 run_s1_moments.py --surface $S --keep 128 --proposal-lattice 64 \
        --row-chunk 1500 --output $A/base --tag base
python3 run_s1_moments.py --surface $S --keep 512 --proposal-lattice 96 \
        --row-chunk 500  --output $A/hi   --tag hi
python3 run_s2_report.py --moments $A --tag hi --cross-tag base \
        --output <out> --bootstrap 2000
python3 run_s4_vertex.py --surface $S --output <out>
```

The seed is the previous run's, and the direction noise comes from its own
generator, so the `O0` bank reproduces `met_nullspace` exactly: on the same
loose cohort of `59,344` events the likelihood ratio, joint and marginal AUCs
come back as `0.6206146`, `0.6187704`, `0.6180718` against that run's `0.62061`,
`0.61877`, `0.61807`, and the same `46` events have no valid hypothesis.

Then locally, with matplotlib:

```sh
python3 run_s3_figures.py --output <out>
python3 run_s5_sensitivity.py --output <out> --bootstrap 2000
```

## Results

`analysis/outputs/sheet-ambiguity-20260920/`.  Per-event moment arrays stay on
ICEPP under `~/sheet-ambiguity-20260920/`.
