"""Plots for the frozen same-event spin readout comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.summary).read_text())
    d = np.load(args.scores)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.grid":True, "grid.alpha":.18})
    reg = "mHH_gt350"
    names = list(s["arms"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for i, name in enumerate(names):
        for seed in range(3):
            r = s["arms"][name][seed]["regions"][reg]
            axes[0].scatter(r["HZ_AUC"], i + (seed-1)*.10, color=f"C{i%10}", s=27)
            axes[1].scatter(100*(r["R"]-1), i + (seed-1)*.10, color=f"C{i%10}", s=27)
    for ax in axes:
        ax.set_yticks(range(len(names)), names); ax.invert_yaxis()
    axes[0].set(xlabel="H/Z AUC, weighted hypothesis copies", xlim=(.50,.59))
    axes[1].set(xlabel="Proxy Z gain over one inclusive count [%]")
    fig.suptitle("Same HH kinematics, mHH > 350 GeV; three initializations per readout\n"
                 "Fixed illustrative S=40, B=160; H/Z/W/U mixture; normalization and contrast profiling")
    fig.tight_layout(); save(fig,out/"spin_readouts")

    pairs = ["Raw4/Angular4", "Mean6/Unit6", "Joint15/Mean6",
             "Joint15/Angular4", "K+Angular4/K", "K+Joint15/K", "K+Joint15/K+Mean6"]
    fig, ax = plt.subplots(figsize=(9.5,4.7))
    for i,pair in enumerate(pairs):
        p=s["paired_bootstrap"][reg+":"+pair]
        v=100*(p["median"]-1);lo,hi=100*(np.array(p["interval68"])-1)
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=f"C{i}")
    ax.set_yticks(range(len(pairs)),pairs); ax.invert_yaxis()
    ax.axvline(0,color="k",lw=.8)
    ax.set(xlabel="Paired proxy Z increase [%]",title="Seed 0, mHH > 350 GeV: 100 paired event bootstraps\n"
           "68% MC intervals; classifiers and validation bins fixed")
    fig.tight_layout(); save(fig,out/"spin_paired_contrasts")

    fig, axes=plt.subplots(1,2,figsize=(12.5,4))
    row=s["arms"]["K+Joint15"][0]["regions"][reg]
    t=np.array(row["templates"]);v=np.array(row["template_var"])
    for i,label in enumerate(("H (signal and single H)","Z","W polarization","U (spin-flat)")):
        axes[0].errorbar(np.arange(t.shape[1]),t[i],yerr=np.sqrt(v[i]),fmt="o-",label=label,lw=1)
    axes[0].set(xlabel="Validation-fixed score bin, increasing signal-like",ylabel="Fraction within each hypothesis")
    axes[0].legend(fontsize=8)
    for name in ("Angular4","Joint15","K","K+Angular4","K+Joint15"):
        axes[1].plot([r["top"] for r in s["composition_scan"]],
                     [100*(r[name]-1) for r in s["composition_scan"]],"o-",label=name,ms=3)
    axes[1].set(xlabel="Illustrative W background fraction (Z = 0.60 - W)",ylabel="Proxy Z gain over one count [%]")
    axes[1].legend(fontsize=8)
    fig.suptitle("Conditional spin experiment: no production-kinematic H/Z separation\n"
                 "Single H=0.28 and U=0.12 fixed; no ATLAS background-fraction inference")
    fig.tight_layout();save(fig,out/"spin_shapes_composition")

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ("Angular4","Raw4","Mean6","Joint15"):
        history=s["arms"][name][0]["history"]
        for ax,key in zip(axes,("train","validation")):
            ax.plot(history[key]["mlogloss"],label=name)
    for ax,label in zip(axes,("Train uid%3=0","Validation uid%3=1")):
        ax.set(xlabel="Boosting iteration",ylabel="Balanced four-class log loss",title=label);ax.legend()
    fig.suptitle("Spin readout optimization: validation stopping, seed 0")
    fig.tight_layout();save(fig,out/"spin_learning")


if __name__=="__main__":main()
