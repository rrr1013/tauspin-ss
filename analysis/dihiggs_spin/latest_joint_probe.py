"""Exploratory mechanism probe: learned joint targets vs explicit products of means.

Requested by independent skeptical review after the original readout results.
Fixed seed 0/config/split/bootstrap; does not select the primary method.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from latest_spin import weights, train, discriminant, edges, templates, zprofile, auc_hz


def main():
    ap=argparse.ArgumentParser()
    for name in ("data","pred","reference-summary","reference-scores","out"):
        ap.add_argument("--"+name,required=True)
    ap.add_argument("--bootstrap",type=int,default=100)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=np.load(args.data,allow_pickle=True);p=np.load(args.pred,allow_pickle=True)
    assert np.array_equal(d["uid"],p["uid"]) and np.array_equal(d["proc"],p["proc"])
    reference=json.loads(Path(args.reference_summary).read_text())
    rs=np.load(args.reference_scores)
    assert np.array_equal(d["uid"],rs["uid"])
    W,_=weights(d["h_exact"]);W*=d["w"][:,None]
    split=d["uid"]%3
    hp=p["h_pred"]
    mode=np.stack([d[f"spin_rmode{j}_{side}"] for side in (0,1) for j in range(5)],1)
    product=(hp[:,0,:,None]*hp[:,1,None,:]).reshape(-1,9)
    X=np.c_[hp.reshape(-1,6),product,mode].astype(np.float32)
    K=np.stack([d[k] for k in sorted(k for k in d.files if k.startswith("kin_"))],1).astype(np.float32)
    regions={"inclusive":np.ones(len(split),bool),"mHH_gt350":d["kin_m_hh"]>350}
    fraction=np.array(reference["fraction"])
    result={"definition":"Seed-0 fixed readout expression control; no claim of posterior calibration or optimal information.",
            "protocol":vars(args),"arms":{},"paired_bootstrap":{}}
    scores={}
    for name,x in (("Product15",X),("K+Product15",np.c_[K,X])):
        pp,model,hist=train(x,W,split,0)
        model.save_model(out/(name.replace("+","_")+"_model.json"))
        D=discriminant(pp,fraction);scores[name]=D
        hz=np.log(np.maximum(pp[:,0],1e-12)/np.maximum(pp[:,1],1e-12))
        row={"history":hist,"best_iteration":model.best_iteration,"regions":{}}
        for reg,m in regions.items():
            e=edges(D,W,(split==1)&m,fraction)
            T,V=templates(D,W,(split==2)&m,e)
            row["regions"][reg]={"edges":e.tolist(),"templates":T.tolist(),"template_var":V.tolist(),
                    "Z":zprofile(T,fraction),"HZ_AUC":auc_hz(hz[(split==2)&m],W[(split==2)&m])}
        result["arms"][name]=row
    rng=np.random.default_rng(20261008)
    for reg,m in regions.items():
        ev=(split==2)&m;counts=rng.poisson(1,(args.bootstrap,int(ev.sum())))
        zb={}
        for name in ("Product15","K+Product15","Joint15","K+Joint15"):
            if name in scores:
                D=scores[name];e=result["arms"][name]["regions"][reg]["edges"]
            else:
                D=rs[f"D_{name}_0"];e=reference["arms"][name][0]["regions"][reg]["edges"]
            zb[name]=np.array([zprofile(templates(D,W,ev,np.array(e),c)[0],fraction) for c in counts])
        for full,control in (("Joint15","Product15"),("K+Joint15","K+Product15")):
            r=zb[full]/zb[control]
            result["paired_bootstrap"][reg+":"+full+"/"+control]={"median":float(np.median(r)),
                    "interval68":np.quantile(r,[.16,.84]).tolist(),"interval95":np.quantile(r,[.025,.975]).tolist()}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(out/"scores.npz",uid=d["uid"],**scores)


if __name__=="__main__":main()
