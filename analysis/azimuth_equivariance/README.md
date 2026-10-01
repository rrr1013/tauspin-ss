# Azimuth-response audit

Autonomous ARIADNE run `tauspin-azimuth-equivariance-20261002`.

The fixed seed-42 `none`, `lab`, and `local` point-h pipelines are evaluated on
global rotations about the beam axis.  The primary orbit is C16; a fixed C17
offset grid probes angles outside C16.  Geometry is always rebuilt from the
saved raw validation IP, track-direction, SV, and reco-basis arrays.  In
particular, already-clipped Cartesian geometry is never rotated.

Run order on ICEPP:

1. `freeze_readout_remote.py` once per arm, using only unrotated train
   predictions.  This reproduces and then freezes the 9/30 H/Z readout.
2. `truth_rotation_check_remote.py` once, independently rebuilding exact h
   from truth four-vectors under the same rotations.
3. `rotation_remote.py` once per arm.  It writes validation predictions and
   scores for C16 and C17 plus numerical and clipping controls.
4. `analyze.py` and `make_figures.py` combine the three fixed-arm outputs.

The test split is never loaded.  Cross-arm differences are conditional on the
particular trained checkpoints; `lab` and `local` are not information-equivalent
coordinate encodings.
