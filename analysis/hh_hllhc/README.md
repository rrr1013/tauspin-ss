# HL-LHC HH→bbττ sensitivity with TauSpin information

Likelihood-level estimate of how much TauSpin information (regressed τ polarimeter ĥ; PV-referenced
muon |d0|/σ in τℓτh) raises the ATLAS HL-LHC SM HH expected significance (ATL-PHYS-PUB-2025-006:
4.26σ combination, bbττ 3.54σ). Human-readable report: Vault
`10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-dihiggs-hllhc-sensitivity-20261008/00_Run.md`.

Outputs, inputs and the Python venv live on ICEPP: `lxgpu02:/home/rbaba/dihiggs-hllhc-20261008/`
(`outputs/{spin,lifetime,latest,calib,study,figures}`, `lit/data` = downloaded primary sources).

| file | role |
|---|---|
| `hepdata_inputs.py` | Run-2 legacy (2209.10910) per-bin yields from HEPData (with the LTT Z+HF fix); loader for latest (2607.26879) yields |
| `extract_latest.py` | recovers the latest analysis' 12 SR × bin × process yields from the vector PDFs of Fig. 7/8 |
| `spin_response.py` | spin-hypothesis classifiers on the spin-flat HH cohort (out-of-fold probabilities; arms textbook / tauspin / exact) |
| `lifetime.py` | lepton \|d0\|/σ templates per lepton origin and bin window; τ→ℓ shares of background groups |
| `projection.py` | pyhf model: baseline bins × spin sub-bins (× d0 bins in τℓτh), Asimov discovery significance |
| `calibrate.py` | systematic-model calibration to PUB-2024-016 Table 4 |
| `study.py` | all variants; gain ratios R and transfer to ATLAS numbers |
| `make_figures.py` | figures 1–7 |
| `lit/` | literature survey (GPT-6.1 Sol) and HEPData yaml |
| `review/`, `review_packet.md` | validity and skeptical reviews |

Run (on lxgpu02, `HH_OUT=/home/rbaba/dihiggs-hllhc-20261008/outputs`, `HH_LIT=.../lit/data`):
`spin_response.py --seed {0,1,2}` (+ `--c_t 0`), `extract_latest.py`, `lifetime.py`, `calibrate.py`,
`study.py <variant>`, `make_figures.py`.
