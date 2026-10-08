"""Predeclared flavor diagnostics of frozen nominal proxy scores, no retraining."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from latest_lephad import evaluate, bootstrap, dump


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepared",required=True)
    ap.add_argument("--run",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--bootstrap",type=int,default=100)
    args=ap.parse_args()
    D=dict(np.load(args.prepared,allow_pickle=True))
    all_results=json.loads((Path(args.run)/"summary.json").read_text())
    results={k:v for k,v in all_results.items()
             if k.startswith("nominal/") and "GBDT" not in k and k.endswith("/0")}
    if len(results)!=4 or not all(v["status"]["optimization_complete"] for v in results.values()):
        raise ValueError("Need all four converged nominal seed-0 arms")
    out={"definition":"Fixed mixed-flavor classifiers, same validation-only bin rule; flavor subsets; constant toy ID efficiencies. Not an ID calibration.",
         "model_commit":next(iter(results.values()))["status"]["git_commit"],"flavors":{}}
    for flavor,is_e in (("electron",True),("muon",False)):
        subset=D.copy()
        subset["sel_nominal"]=D["sel_nominal"] & (D["lep_is_e"]==is_e)
        r={}
        for key,v in results.items():
            r[key]={"score_path":v["score_path"],
                    "evaluation":evaluate(np.load(v["score_path"])["score"],subset,"nominal")}
        boot=bootstrap(SimpleNamespace(bootstrap=args.bootstrap),subset,r)
        base=r["nominal/B/0"]["evaluation"]["profile"]
        ratios={}
        for key,v in r.items():
            p=v["evaluation"]["profile"]
            value=p["Z"]/base["Z"] if p["success"] and base["success"] and base["Z"] else None
            resolved=(v["evaluation"]["sensitivity_resolved"] and
                      r["nominal/B/0"]["evaluation"]["sensitivity_resolved"])
            ratios[key]={"R":value,"both_arms_sensitivity_resolved":resolved,
                         "interpretation":"Supported proxy estimate" if resolved else "Diagnostic only: MC-support or optimizer issue"}
        out["flavors"][flavor]={"selected_rows":int(subset["sel_nominal"].sum()),
                "results":r,"R_to_B":ratios,"bootstrap":boot}
    Path(args.out).mkdir(parents=True,exist_ok=True)
    dump(Path(args.out)/"flavor.json",out)


if __name__=="__main__":main()
