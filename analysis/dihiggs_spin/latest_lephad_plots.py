"""Inspectible process/learning/score views for the fixed latest lep-had proxy."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = dict(np.load(args.prepared, allow_pickle=True))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.grid":True, "grid.alpha":.15})
    prompt = ~D["lep_from_tau"].astype(bool)
    fig, ax = plt.subplots(1,3,figsize=(13,3.6))
    pull = D["d0_null"] / D["ip_null"][:,1]
    ax[0].hist(pull[prompt], np.linspace(-6,6,81), density=True, histtype="step", label="Corrected scalar, nominal mixture")
    x=np.linspace(-6,6,301); ax[0].plot(x,np.exp(-x*x/2)/np.sqrt(2*np.pi),"--",label="Unit Gaussian core")
    ax[0].set(xlabel="Prompt scalar d0 / core sigma",ylabel="Density (tails outside view)")
    ax[0].legend(fontsize=8)
    tau = ~prompt
    for key,label,style in (("d0_true_corrected","Track transverse closest approach","-"),
                            ("d0_true_legacy_radius","Legacy transverse vector length","--")):
        ax[1].hist(np.abs(D[key][tau]),np.linspace(0,500,101),density=True,histtype="step",label=label,linestyle=style)
    ax[1].set(xlabel="True transverse impact magnitude [um]",ylabel="Density (overflow excluded)",yscale="log")
    ax[1].legend(fontsize=8)
    for key,label,style in (("z0sin_true_corrected","Transverse closest point","-"),("z0sin_true_legacy","Legacy Iz sin(theta)","--")):
        ax[2].hist(D[key][tau],np.linspace(-500,500,101),density=True,histtype="step",label=label,linestyle=style)
    ax[2].set(xlabel="True z0 sin(theta) [um]",ylabel="Density (overflow excluded)",yscale="log")
    ax[2].legend(fontsize=8)
    fig.suptitle("Toy lep-had MC before downstream selection: impact geometry and null resolution")
    fig.tight_layout();save(fig,out/"geometry_resolution")
    run=Path(args.run)
    summary=run/"summary.json"
    if not summary.exists(): summary=run/"pilot_summary.json"
    R=json.loads(summary.read_text())
    nominal={k:v for k,v in R.items() if k.startswith("nominal/") and "GBDT" not in k}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    colors={"B":"C0","B+d0":"C1","B+Had":"C2","B+d0+Had":"C3"}
    for key,r in nominal.items():
        _,arm,seed=key.split("/")
        curve=json.loads((Path(r["score_path"]).parent/"curve.json").read_text())
        if seed!="0":continue
        epoch=[h["epoch"] for h in curve]
        for a,name in zip(axes,("train_loss","val_loss")):
            a.plot(epoch,[h[name] for h in curve],label=arm,color=colors[arm])
    for a in axes:a.set(xlabel="Epoch",ylabel="Class-balanced weighted BCE");a.legend()
    axes[0].set_title("Training fold uid%3=0");axes[1].set_title("Validation fold uid%3=1")
    fig.tight_layout();save(fig,out/"learning_seed0")
    for key,r in nominal.items():
        if "evaluation" not in r or key.split("/")[-1]!="0":continue
        arm=key.split("/")[1]
        score=np.load(r["score_path"])["score"]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for a,(name,category) in zip(axes,(("Hi",1),("Lo",0))):
            m=D["sel_nominal"]&(D["uid"]%3==2)&(D["mass_category"]==category)
            for p in np.unique(D["proc"]):
                mm=m&(D["proc"]==p)
                a.hist(score[mm],np.linspace(0,1,41),weights=3*D["w"][mm],histtype="step",label=str(p))
            edges=[float(x) for x in r["evaluation"]["categories"][name]["edges"]]
            for e in edges[1:-1]:a.axvline(e,color="k",lw=.7,ls=":")
            a.set(xlabel="Proxy score",ylabel="Expected weighted yield / 0.025 (3 ab^-1)",yscale="log",
                  title="SLT " + name + (": mHH > 350 GeV" if category else ": mHH <= 350 GeV"))
            a.legend(fontsize=7,ncol=2)
        fig.suptitle(f"{arm}, seed 0: exploratory reused-MC evaluation; validation-fixed bins")
        fig.tight_layout();save(fig,out/("score_"+arm.replace("+","_")))
        cats=r["evaluation"]["categories"]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for a,name in zip(axes,("Hi","Lo")):
            c=cats[name]["evaluation"]; S=np.array(c["S"]);B=np.array(c["total_background"])
            centers=np.arange(len(S));bottom=np.zeros(len(S))
            for p,v in c["processes"].items():
                if p=="hhC":continue
                values=np.array(v["yield"]);a.bar(centers,values,bottom=bottom,label=p);bottom+=values
            a.errorbar(centers,B,yerr=np.sqrt(c["mc_variance"]),fmt="k_",label="Background MC uncertainty")
            a.plot(centers,S,"ko--",label="HH (unscaled)")
            a.set(xlabel="Score bin, increasing signal-like",ylabel="Expected yield (3 ab^-1)",yscale="log",
                  title="SLT " + name + (": mHH > 350 GeV" if name == "Hi" else ": mHH <= 350 GeV"))
            a.legend(fontsize=7,ncol=2)
        fig.suptitle(f"{arm}, seed 0: process composition and finite MC statistics")
        fig.tight_layout();save(fig,out/("bins_"+arm.replace("+","_")))


if __name__=="__main__":main()
