"""Standalone Pythia check of tau spin handling: gg -> H -> tau tau and
ffbar -> W -> tau nu, stored with the same truth layout as pythia_shower."""
import sys
import numpy as np
import fastjet
import pythia8
from pythia_shower import empty, process_event

proc, nev, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
extra = sys.argv[4:]
py = pythia8.Pythia("", False)
cmds = ["Beams:eCM = 14000", "111:mayDecay = off", "Next:numberCount = 0",
        ]
cmds += extra
if proc == "H":
    cmds += ["HiggsSM:gg2H = on", "25:onMode = off", "25:onIfAny = 15"]
elif proc == "Z":
    cmds += ["WeakSingleBoson:ffbar2gmZ = on", "23:onMode = off", "23:onIfAny = 15",
             "WeakZ0:gmZmode = 2"]
else:
    cmds += ["WeakDoubleBoson:ffbar2WW = on", "24:onMode = off", "24:onIfAny = 15"]
for c in cmds:
    py.readString(c)
py.init()
o = empty(nev)
jd = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
k = 0
while k < nev:
    if not py.next():
        continue
    process_event(py.event, o, k, False, jd)
    k += 1
np.savez_compressed(out, **{a: b for a, b in o.items() if not a.startswith("fk_")})
