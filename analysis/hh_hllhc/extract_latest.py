"""Recover the bin-by-bin, process-by-process post-fit yields of the latest ATLAS
HH->bbtautau analysis (arXiv:2607.26879, Figures 7 and 8: final classifier score
distributions of all 12 signal regions) from the vector graphics of the published PDFs.

Each stacked process is drawn as one filled polygon giving the cumulative stack height
per bin on a log axis; the ggF/VBF HH x20 overlays are stroked polylines.  The log axis
is calibrated from its tick marks (major ticks are the longer ones; the lowest decade is
read from the tick labels).  No digitising by eye: the numbers are the PDF coordinates.
Closure: per-region process totals are compared with the auxiliary yield tables
(tabaux01: tau_had tau_had, tabaux02: tau_lep tau_had).
"""
import json
import os
from pathlib import Path

import fitz
import numpy as np

DATA = Path(os.environ.get("HH_LIT", Path(__file__).resolve().parent / "lit" / "data"))
OUT = Path(os.environ.get("HH_OUT", Path(__file__).resolve().parent / "outputs")) / "latest"

HADHAD = [((0.8, 0.0, 0.0), "hh_postfit"), ((0.6, 0.8, 1.0), "fake_mj"), ((1.0, 0.8, 0.2), "ttbar"),
          ((1.0, 0.4, 0.4), "fake_tt"), ((0.6, 1.0, 0.6), "zhf"), ((0.8, 0.4, 0.0), "tW"),
          ((0.2, 0.2, 0.6), "other"), ((1.0, 1.0, 0.6), "single_h")]
LEPHAD = [((0.8, 0.0, 0.0), "hh_postfit"), ((0.6, 0.8, 1.0), "fake"), ((1.0, 0.8, 0.2), "ttbar"),
          ((0.6, 1.0, 0.6), "zhf"), ((0.8, 0.4, 0.0), "tW"), ((0.2, 0.2, 0.6), "other"),
          ((1.0, 1.0, 0.6), "single_h")]
LINES = {(0.6, 0.2, 0.2): "ggF_x20", (0.0, 0.4, 1.0): "VBF_x20"}
FIGS = {"07a": ("run2", "hadhad", "Lo"), "07b": ("run2", "lephad", "Lo"), "07c": ("run2", "hadhad", "Hi"),
        "07d": ("run2", "lephad", "Hi"), "07e": ("run2", "hadhad", "VBF"), "07f": ("run2", "lephad", "VBF"),
        "08a": ("run3", "hadhad", "Lo"), "08b": ("run3", "lephad", "Lo"), "08c": ("run3", "hadhad", "Hi"),
        "08d": ("run3", "lephad", "Hi"), "08e": ("run3", "hadhad", "VBF"), "08f": ("run3", "lephad", "VBF")}


def rnd(c):
    return tuple(round(x, 2) for x in c)


def segments(d):
    out = []
    for it in d["items"]:
        if it[0] == "l":
            out.append((it[1].x, it[1].y, it[2].x, it[2].y))
    return out


def extract(path):
    page = fitz.open(path)[0]
    dr = page.get_drawings()
    words = page.get_text("words")
    hadhad = any(w[4] == "(MJ)" for w in words)
    order = HADHAD if hadhad else LEPHAD
    polys = {}
    for d in dr:
        f = d.get("fill")
        if f and len(d["items"]) > 10 and rnd(f) in dict(order):
            polys[dict(order)[rnd(f)]] = d
    frame = polys["hh_postfit"]["rect"]
    x_axis = frame.x1                      # page x of the y-axis minimum (rotated page)
    y_lo, y_hi = frame.y0, frame.y1        # bins run along page y
    # bin edges from the stack polygon vertices
    ys = sorted({round(s[1], 2) for s in segments(polys["hh_postfit"])} |
                {round(s[3], 2) for s in segments(polys["hh_postfit"])})
    edges = np.array([y for y in ys if y_lo - 0.01 <= y <= y_hi + 0.01])
    # log-axis calibration from the ticks drawn on the frame edge at page y = y_lo
    ticks = []
    for d in dr:
        if d.get("fill") is None and d.get("color") == (0.0, 0.0, 0.0):
            for x0, y0, x1, y1 in segments(d):
                if abs(x0 - x1) < 0.01 and min(y0, y1) == y_lo and abs(y1 - y0) < 20 and x0 < x_axis - 0.01:
                    ticks.append((round(x0, 3), round(abs(y1 - y0), 3)))
    ticks = sorted(set(ticks))
    L = max(t[1] for t in ticks)
    majors = sorted([t[0] for t in ticks if t[1] > 0.75 * L], reverse=True)  # ascending value
    D = np.median(-np.diff(majors))
    txt = " ".join(w[4] for w in words)
    lowest = -1 if txt.startswith("1 − 10") else 0
    # value = 10^(lowest + (majors[0] - x) / D)
    to_val = lambda x: 10 ** (lowest + (majors[0] - x) / D)
    nb = len(edges) - 1
    centers = 0.5 * (edges[1:] + edges[:-1])

    def level(d):
        """x of the vertical (constant-x) polygon/line segment covering each bin centre."""
        out = np.full(nb, np.nan)
        for x0, y0, x1, y1 in segments(d):
            if abs(x0 - x1) < 1e-6:
                lo, hi = sorted((y0, y1))
                for i, c in enumerate(centers):
                    if lo - 1e-3 <= c <= hi + 1e-3 and hi - lo > 0.5 * (edges[1] - edges[0]):
                        out[i] = x0
        return out

    cum = {name: to_val(level(polys[name])) for _, name in order}
    names = [n for _, n in order]
    yields = {}
    for i, n in enumerate(names):
        below = cum[names[i + 1]] if i + 1 < len(names) else 0.0
        yields[n] = cum[n] - below
    for d in dr:
        c = d.get("color")
        if d.get("fill") is None and c and rnd(c) in LINES and len(d["items"]) > 5:
            lv = level(d)
            v = to_val(lv)
            v[~np.isfinite(lv) | (lv >= x_axis - 0.05)] = 0.0  # clipped at the axis minimum
            yields[LINES[rnd(c)]] = v
    # bins are listed from the lowest score (page y = y_hi) to the highest
    yields = {k: v[::-1].tolist() for k, v in yields.items()}
    return {"n_bins": nb, "decade_pt": float(D), "lowest_decade": lowest, "yields": yields,
            "n_ticks": len(ticks), "n_majors": len(majors)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for fig, (run, ch, sr) in FIGS.items():
        r = extract(DATA / f"latest_fig{fig}.pdf")
        r.update(run=run, channel=ch, sr=sr, figure=fig)
        res[f"{run}_{ch}_{sr}"] = r
        tot = {k: round(sum(v), 2) for k, v in r["yields"].items()}
        print(fig, run, ch, sr, r["n_bins"], tot, flush=True)
    json.dump(res, open(OUT / "latest_bins.json", "w"), indent=1)


if __name__ == "__main__":
    main()
