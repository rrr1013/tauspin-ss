# Z -> tau tau spin state: generator closure and entanglement (v3 full-ME Z)

Run note: `AthenaVault/10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-z-entanglement-20260927/`.

* `ztheory.py` - tau- tau+ spin state (N, B-, B+, C) from a Cartesian Z spin
  density R by explicit Dirac traces (exact in m_tau); R from the tau direction
  moments; concurrence / PPT / Horodecki m12.
* `polarimeter.py` - copied unchanged from `analysis/mode_pair_auc_origin`
  (branch `ariadne/mode-pair-auc-origin-20260924`): exact h with the generator
  3pi current and the project's canonical frames.
* `extract_lhe.py` - raw (pre-filter) LHE -> flat arrays (runs on ICEPP).
* `zent.py` - exact h in pair-frame Cartesian axes, Collins-Soper axes,
  predicted and measured states.
* `toy_closure.py` - closure of the whole chain on a toy with known R.
