"""Assemble the background-knowledge scan of ent_robust.py --aux-scan from its log, and rerun the
specs that did not return (a fit with all starting points non-finite raises in ent_robust.test and
aborts the pool).  Reruns use all tetrahedron starting points (as ent_robust does for drift specs);
a spec that still fails is recorded as failed instead of being dropped silently.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ip_tau_reco"))
import ent_robust as er  # noqa: E402

LINE = re.compile(r"^(\S+) (tau[0-9.]+(?:_lumi6000)?) ([0-9.]+) \[")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    got = {}
    for ln in open(a.log):
        m = LINE.match(ln)
        if m:
            got.setdefault(m.group(1), {})[m.group(2)] = {"Z": float(m.group(3)), "source": "log"}
    specs = [(l, f"tau{t:g}", {"tau": t}) for l in a.levels.split(",") for t in (0, 0.3, 1, 3, 10, 100)]
    specs += [(l, f"tau{t:g}_lumi6000", {"tau": t, "scale": 2.0}) for l in a.full.split(",") for t in (1, 10)]
    er.load_prep(a.cache)
    orig = er.TET_STARTS
    for lev, name, kw in specs:
        if name in got.get(lev, {}):
            continue
        er.TET_STARTS = orig + [er.np.log(er.np.array(w) + 1e-3) for w in ([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])]
        try:
            kw = dict(kw, c_true=er.TRUE_C)   # job() then uses all tetrahedron starts; the Asimov truth is unchanged
            _, r = er.job((lev, name, kw))
            r["source"] = "rerun_all_starts"
        except Exception as e:  # noqa: BLE001
            r = {"Z": None, "source": f"failed: {e!r}"}
        got.setdefault(lev, {})[name] = r
        print(lev, name, r.get("Z"), r["source"], flush=True)
    json.dump({"config": vars(a), "levels": got}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
