"""Evaluate posterior_rho_oracle.py arms: H/Z AUC by truth pair, per-side h quality, and paired bootstrap.
Wrapper around taukin_gap/eval_kin.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'taukin_gap'))
import eval_kin  # noqa: E402

if __name__ == '__main__':
    eval_kin.main()
