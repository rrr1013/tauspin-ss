"""Write run_card settings for each process (14 TeV)."""
import re, sys
from pathlib import Path

COMMON = {"ebeam1": "7000.0", "ebeam2": "7000.0", "use_syst": "False",
          "systematics_program": "none", "nevents": None, "iseed": None}
CUTS = {
    "hh": {},
    "ttll": {},
    "ttlj": {},
    "zbb": {"ptb": "20.0", "etab": "2.7", "drbb": "0.2", "ptl": "15.0", "etal": "2.7", "drll": "0.2"},
    "zh": {"ptb": "20.0", "etab": "2.7", "drbb": "0.2"},
    "tth": {},
    "ttlt": {}, "ttljp": {}, "tWll": {}, "tWlj": {},
}

def patch(card, name, nevents, seed):
    txt = Path(card).read_text()
    opts = dict(COMMON, nevents=str(nevents), iseed=str(seed))
    opts.update(CUTS[name])
    for k, v in opts.items():
        txt, n = re.subn(rf"^(\s*)\S+(\s*=\s*{k}\b)", rf"\g<1>{v}\g<2>", txt, flags=re.M)
        if n == 0:
            print(f"warning: {k} not in run_card", file=sys.stderr)
    Path(card).write_text(txt)

if __name__ == "__main__":
    patch(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
