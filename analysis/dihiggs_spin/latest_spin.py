"""Compare TauPolaris-style observables and existing TauSpin information on one
spin-flat HH cohort. This is a readout comparison, not execution of TauPolaris.

Protocol: uid%3 train=0, validation=1, evaluation=2; event-level split before
duplicating rows under H/Z/W/U spin hypotheses. XGBoost validation-loss stopping;
no evaluation-dependent training, thresholds, bins or background fractions.
The same reconstructed polarimeters and generated spin weights feed every arm.
This previously inspected cohort is exploratory. Yields/background fractions
are explicitly illustrative; there is no ATLAS likelihood or detector claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

HYPS = ("H", "Z", "W", "U")


def weights(h):
    valid = np.isfinite(h).all(-1)
    a = np.where(valid[..., None], h, 0.0)
    hm, hp = a[:, 0], a[:, 1]
    nn, rr, kk = hm[:, 0]*hp[:, 0], hm[:, 1]*hp[:, 1], hm[:, 2]*hp[:, 2]
    # Canonical h is minus physical h on both sides: B_k=-0.147.
    # Equal/opposite transverse coefficients ensure a physical Z density.
    w = np.stack((1+nn+rr-kk,
                  1-.147*(hm[:, 2]+hp[:, 2])+kk+.5*(nn-rr),
                  (1-hm[:, 2])*(1-hp[:, 2]), np.ones(len(h))), 1)
    if np.min(w) < -1e-6:
        raise ValueError(f"Negative physical spin density: {np.min(w)}")
    return np.maximum(w, 0), valid


def angular4(h, normalize):
    if normalize:
        h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-8)
    a, b = h[:, 0], h[:, 1]
    return np.stack((a[:, 0]*b[:, 0]-a[:, 1]*b[:, 1],
                     a[:, 2]*b[:, 2], a[:, 2], b[:, 2]), 1)


def train(X, W, split, seed):
    mats = []
    for f in (0, 1):
        m = split == f
        ww = W[m].T.copy()
        ww *= m.sum() / ww.sum(1, keepdims=True)
        mats.append(xgb.DMatrix(np.tile(X[m], (4, 1)),
                                label=np.repeat(np.arange(4), m.sum()),
                                weight=ww.reshape(-1)))
    hist = {}
    model = xgb.train(dict(objective="multi:softprob", num_class=4,
                           eval_metric="mlogloss", tree_method="hist", max_depth=4,
                           eta=.04, min_child_weight=100, reg_lambda=2, nthread=4,
                           subsample=.8, colsample_bytree=.9, seed=seed),
                      mats[0], num_boost_round=2000,
                      evals=[(mats[0], "train"), (mats[1], "validation")],
                      early_stopping_rounds=50, evals_result=hist, verbose_eval=False)
    if model.best_iteration >= 1949:
        raise RuntimeError("Training ceiling reached; technical recovery required")
    p = model.predict(xgb.DMatrix(X), iteration_range=(0, model.best_iteration+1))
    return p, model, hist


def discriminant(p, fraction):
    return np.log(np.maximum(p[:, 0], 1e-12))-np.log(np.maximum(p@fraction, 1e-12))


def edges(D, W, m, fraction, n=10):
    i = np.flatnonzero(m)
    o = np.argsort(D[i]); bw = W[i]@fraction
    cw = np.cumsum(bw[o]); cw /= cw[-1]
    return np.unique(np.interp(np.arange(1, n)/n, cw, D[i][o]))


def templates(D, W, m, e, count=None):
    b = np.searchsorted(e, D[m]); w = W[m]
    if count is not None:
        w = w*count[:, None]
    T = np.stack([np.bincount(b, weights=w[:, j], minlength=len(e)+1) for j in range(4)])
    V = np.stack([np.bincount(b, weights=w[:, j]**2, minlength=len(e)+1) for j in range(4)])
    total=T.sum(1,keepdims=True)
    prob=T/total
    normalized_var=(V*(1-2*prob)+prob**2*V.sum(1,keepdims=True))/total**2
    return prob, normalized_var


def zprofile(T, fraction, s=40., b=160., contrast=.25):
    B = b*fraction[:, None]*T
    S = s*T[0]; obs=S+B.sum(0)
    # Three positive-normalisation nuisances, shared over bins; four hyp with
    # H(single-H), Z, W, U; contrast nuisance weakens spin distinctions.
    prior = np.array([.15, .10, .10, .20])
    average = (S+B.sum(0))/(s+b)
    def loss(x):
        # Physical bounded morph range: alpha in [0,1], varied down by contrast.
        alpha = 1-contrast*np.tanh(x[4])
        morph = alpha*T+(1-alpha)*average
        rate = (b*fraction*np.exp(prior*x[:4]))@morph
        if np.any(rate <= 0):
            return 1e50
        return float(np.sum(rate-obs+obs*np.log(obs/rate))+.5*x@x)
    fit=minimize(loss, np.zeros(5), method="L-BFGS-B", bounds=[(-5,5)]*4+[(0,5)])
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(str(fit.message))
    return float(np.sqrt(2*max(fit.fun,0)))


def auc_hz(score, w):
    n=len(score)
    return float(roc_auc_score(np.r_[np.ones(n),np.zeros(n)],
                               np.r_[score,score],sample_weight=np.r_[w[:,0],w[:,1]]))


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--data",required=True); a.add_argument("--pred",required=True)
    a.add_argument("--out",required=True); a.add_argument("--seeds",type=int,default=3)
    a.add_argument("--bootstrap",type=int,default=100)
    args=a.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    d=np.load(args.data,allow_pickle=True); p=np.load(args.pred,allow_pickle=True)
    assert np.array_equal(d["uid"],p["uid"]) and np.array_equal(d["proc"],p["proc"])
    assert len(np.unique(d["uid"])) == len(d["uid"])
    split=d["uid"]%3
    W,valid=weights(d["h_exact"]); W*=d["w"][:,None]
    assert np.min(d["w"])>0
    hp=p["h_pred"]
    mode=np.stack([d[f"spin_rmode{j}_{s}"] for s in (0,1) for j in range(5)],1)
    angle=np.c_[angular4(hp,True),mode].astype(np.float32)
    raw=np.c_[angular4(hp,False),mode].astype(np.float32)
    full=np.c_[hp.reshape(-1,6),p["hh_pred"].reshape(-1,9),mode].astype(np.float32)
    cols=sorted(k for k in d.files if k.startswith("kin_"))
    K=np.stack([d[k] for k in cols],1).astype(np.float32)
    unit6=np.c_[(hp/np.maximum(np.linalg.norm(hp,axis=-1,keepdims=True),1e-8)).reshape(-1,6),mode].astype(np.float32)
    mean6=np.c_[hp.reshape(-1,6),mode].astype(np.float32)
    sets={"Angular4":angle,"Raw4":raw,"Unit6":unit6,"Mean6":mean6,"Joint15":full,
          "K":K,"K+Angular4":np.c_[K,angle],"K+Mean6":np.c_[K,mean6],"K+Joint15":np.c_[K,full]}
    fraction=np.array([.28,.44,.16,.12]) # illustrative historical-like mix
    ev=split==2; val=split==1
    regions={"inclusive":np.ones(len(split),bool),"mHH_gt350":d["kin_m_hh"]>350}
    res={"protocol":vars(args),"N":len(split),"split_counts":np.bincount(split).tolist(),
         "implemented_sides":valid.mean(0).tolist(),"fraction":fraction.tolist(),
         "note":"Readout on identical TauSpin h, not a TauPolaris network benchmark. Fixed toy yields S40/B160; central Z profiles illustrative normalization/contrast with fixed MC templates; paired Poisson bootstrap propagates MC shape uncertainty; no ATLAS transfer.","arms":{}}
    scores={"uid":d["uid"],"split":split,"W":W,"h_pred":hp,"h_exact":d["h_exact"]}
    for name,X in sets.items():
        assert np.isfinite(X).all(),name
        arm=[]
        for seed in range(args.seeds):
            pp,model,hist=train(X,W,split,seed)
            model.save_model(out/f"model_{name.replace('+','_')}_{seed}.json")
            D=discriminant(pp,fraction); hz=np.log(np.maximum(pp[:,0],1e-12)/np.maximum(pp[:,1],1e-12))
            row={"seed":seed,"best_iteration":model.best_iteration,"history":hist,"regions":{}}
            scores[f"D_{name}_{seed}"]=D; scores[f"HZ_{name}_{seed}"]=hz
            for reg,m in regions.items():
                e=edges(D,W,val&m,fraction)
                T,V=templates(D,W,ev&m,e)
                z=zprofile(T,fraction); z1=zprofile(np.ones((4,1)),fraction)
                row["regions"][reg]={"n_eval":int((ev&m).sum()),"HZ_AUC":auc_hz(hz[ev&m],W[ev&m]),
                                       "Z":z,"R":z/z1,"edges":e.tolist(),"templates":T.tolist(),"template_var":V.tolist()}
            arm.append(row); print(name,seed,{k:(round(v['HZ_AUC'],4),round(v['R'],4)) for k,v in row['regions'].items()},flush=True)
        res["arms"][name]=arm
        (out/"summary.json").write_text(json.dumps(res,indent=1))
    # Paired event bootstrap, frozen classifiers and edges; no method selection.
    rng=np.random.default_rng(20261008); pairs={}
    for reg,m in regions.items():
        v=ev&m; counts=rng.poisson(1,(args.bootstrap,int(v.sum())))
        rb={}
        for name in sets:
            D=scores[f"D_{name}_0"]; e=np.array(res['arms'][name][0]['regions'][reg]['edges'])
            rb[name]=np.array([zprofile(templates(D,W,v,e,c)[0],fraction) for c in counts])
        for n1,n0 in (("Joint15","Angular4"),("Raw4","Angular4"),("Mean6","Unit6"),("Joint15","Mean6"),("K+Angular4","K"),("K+Joint15","K"),("K+Joint15","K+Mean6")):
            r=rb[n1]/rb[n0]
            pairs[f"{reg}:{n1}/{n0}"]={"median":float(np.median(r)),"interval68":np.quantile(r,[.16,.84]).tolist(),"interval95":np.quantile(r,[.025,.975]).tolist()}
    res['paired_bootstrap']=pairs
    # Composition-response at fixed D/templates, not classifier re-optimisation.
    scan=[]
    for top in np.linspace(0,.60,13):
        f=np.array([.28,.60-top,top,.12]); rr={"top":float(top)}
        for name in ('Angular4','Joint15','K','K+Angular4','K+Joint15'):
            T=np.array(res['arms'][name][0]['regions']['mHH_gt350']['templates'])
            rr[name]=zprofile(T,f)/zprofile(np.ones((4,1)),f)
        scan.append(rr)
    res['composition_scan']=scan
    (out/"summary.json").write_text(json.dumps(res,indent=1)); np.savez_compressed(out/"scores.npz",**scores)


if __name__ == "__main__":
    main()
