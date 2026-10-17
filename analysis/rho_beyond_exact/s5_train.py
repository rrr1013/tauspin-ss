"""Run gen3pi_teacher/train_point_h.py unchanged, with the dataset's oracle truth-input file taken from
TRUTH_INPUTS_DIR (the row-aligned subset written by s5_make_subset_targets.py).  Same env as train_point_h.py."""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, '/home/rbaba/tauspin-ip-sv-common-v2/NN/ip_sv_ablation')
sys.path.insert(0, '/home/rbaba/tauspin-ip-sv-common-v2/NN')
import reco_data_original_geometry as R  # noqa: E402

R.TRUTH_INPUTS_DIR = Path(os.environ['TRUTH_INPUTS_DIR'])
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'gen3pi_teacher' / 'train_point_h.py'), run_name='__main__')
