from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

H_INPUT_FILES = (
    "/home/rbaba/tauspin-ss/MakeNtuple/outputs/H/H_chunk_*.root"
)
Z_INPUT_FILES = (
    "/home/rbaba/tauspin-ss/MakeNtuple/outputs/Z/Z_chunk_*.root"
)
TREE_NAME = "tauspin"

Z_LABEL = 0
H_LABEL = 1

TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1
TEST_FRACTION = 0.1
SPLIT_SEED = 42

# These bounds limit memory during the one-time ROOT -> PyTorch conversion.
ROOT_STEP_SIZE = "200 MB"
EVENTS_PER_SHARD = 50_000

EVENT_FEATURES = (
    "met_et",
    "met_phi",
    "met_sumet",
)

TAU_CONTINUOUS_FEATURES = (
    "tau_pt",
    "tau_eta",
    "tau_phi",
    "tau_m",
    "tau_nTracks",
    "tau_nIsolatedTracks",
    "tau_rnnJetScoreSigTrans",
    "tau_gntauScoreSigTrans_v0",
    "tau_vertexDeltaZ",
)

TAU_CATEGORICAL_FEATURES = ("tau_decayMode",)

TRACK_FEATURES = (
    "track_pt",
    "track_eta",
    "track_phi",
    "track_charge",
    "track_d0",
    "track_z0SinTheta",
    "track_isCore",
    "track_isIsolation",
    "track_isConversion",
    "track_isFake",
    "track_passTrkSelector",
    "track_numberOfPixelHits",
    "track_numberOfSCTHits",
    "track_numberOfTRTHits",
)

PFO_FEATURES = (
    "pfo_pt",
    "pfo_eta",
    "pfo_phi",
    "pfo_e",
    "pfo_isPi0",
)

EVENT_NUMBER_BRANCH = "eventNumber"
TRACK_TAU_INDEX_BRANCH = "track_tauIndex"
PFO_TAU_INDEX_BRANCH = "pfo_tauIndex"

TAU_MINUS_INDEX = 0
TAU_PLUS_INDEX = 1

EVENT_OUTPUT_FEATURES = (
    "log1p_met_et",
    "sin_met_phi",
    "cos_met_phi",
    "log1p_met_sumet",
)

TAU_OUTPUT_FEATURES = (
    "log1p_tau_pt",
    "tau_eta",
    "sin_tau_phi",
    "cos_tau_phi",
    "log1p_tau_m",
    "tau_nTracks",
    "tau_nIsolatedTracks",
    "tau_rnnJetScoreSigTrans",
    "tau_gntauScoreSigTrans_v0",
    "tau_vertexDeltaZ",
)

TRACK_OUTPUT_FEATURES = (
    "log1p_track_pt",
    "track_eta",
    "sin_track_phi",
    "cos_track_phi",
    "track_charge",
    "track_d0",
    "track_z0SinTheta",
    "track_isCore",
    "track_isIsolation",
    "track_isConversion",
    "track_isFake",
    "track_passTrkSelector",
    "track_numberOfPixelHits",
    "track_numberOfSCTHits",
    "track_numberOfTRTHits",
)

PFO_OUTPUT_FEATURES = (
    "log1p_pfo_pt",
    "pfo_eta",
    "sin_pfo_phi",
    "cos_pfo_phi",
    "log1p_pfo_e",
    "pfo_isPi0",
)

# The remaining output features are standardised with train-only statistics.
EVENT_UNSCALED_FEATURES = ("sin_met_phi", "cos_met_phi")
TAU_UNSCALED_FEATURES = ("sin_tau_phi", "cos_tau_phi")
TRACK_UNSCALED_FEATURES = (
    "sin_track_phi",
    "cos_track_phi",
    "track_charge",
    "track_isCore",
    "track_isIsolation",
    "track_isConversion",
    "track_isFake",
    "track_passTrkSelector",
)
PFO_UNSCALED_FEATURES = (
    "sin_pfo_phi",
    "cos_pfo_phi",
    "pfo_isPi0",
)

# Selected optional branches were complete in the test ntuples. The production
# builder fails clearly instead of silently learning sentinel values.
MISSING_SENTINELS = (-999.0, -9999.0)

# Transformer defaults agreed for the first model.
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
RANDOM_SEED = 42

if abs(
    TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0
) > 1.0e-12:
    raise ValueError("Dataset split fractions must sum to 1.")
