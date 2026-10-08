"""Small numerical validation of unchanged likelihood on archived templates."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, least_squares


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    ap.add_argument("--summary",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    root=Path(args.source)
    ns={"np":np,"minimize":minimize}
    definitions=ast.parse((root/"classify_lephad.py").read_text())
    for node in definitions.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("GROUPS","PRIOR") for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),"constants","exec"),ns)
    code=ast.parse((root/"latest_lephad.py").read_text())
    node=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=="profile")
    exec(compile(ast.Module(body=[node],type_ignores=[]),"profile","exec"),ns)
    results={}
    for key,r in json.loads(Path(args.summary).read_text()).items():
        cats=[r["evaluation"]["categories"][k]["evaluation"] for k in ("Hi","Lo")]
        s=np.concatenate([c["S"] for c in cats])
        B=np.concatenate([np.array([c["background_groups"][g] for g in ns["GROUPS"]]) for c in cats],1)
        V=np.concatenate([c["mc_variance"] for c in cats])
        p=ns["profile"](s,B,V)
        noMC=ns["profile"](s,B,V,mc=False)
        stat=ns["profile"](s,B,V,norm=False,mc=False)
        # Independent full theta+log(beta) fit, no analytic beta elimination.
        n=np.array(s+B.sum(0),dtype=np.longdouble)
        tau=np.array(B.sum(0)**2/V,dtype=np.longdouble)
        prior=np.log1p(np.array(list(ns["PRIOR"].values()),dtype=np.longdouble))
        def residual(par):
            theta=np.array(par[:len(prior)],dtype=np.longdouble)
            eta=np.array(par[len(prior):],dtype=np.longdouble)
            bs=(B*np.exp(prior[:,None]*theta[:,None])).sum(0)
            logratio=np.log1p((bs-n)/n)+eta
            data=2*n*(np.expm1(logratio)-logratio)
            aux=2*tau*(np.expm1(eta)-eta)
            return np.asarray(np.r_[np.sign(logratio)*np.sqrt(np.maximum(data,0)),
                                     np.sign(eta)*np.sqrt(np.maximum(aux,0)),theta],dtype=float)
        other=least_squares(residual,np.zeros(len(prior)+len(n)),ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=2000)
        zother=float(np.linalg.norm(other.fun))
        rel=abs(p["Z"]-zother)/zother
        bg=np.asarray(B.sum(0),dtype=np.longdouble)
        sig=np.asarray(s,dtype=np.longdouble)
        zclosed=float(np.sqrt(2*np.sum((sig+bg)*np.log1p(sig/bg)-sig)))
        alternate=None
        if key in ("nominal/B/0","nominal/B+d0/0"):
            start=np.zeros(len(prior)+len(n));start[:len(prior)]=np.linspace(-.03,.03,len(prior))
            alt=least_squares(residual,start,ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=2000)
            alternate={"success":bool(alt.success),"relative_Z_discrepancy":abs(float(np.linalg.norm(alt.fun))-p["Z"])/p["Z"]}
        results[key]={"profile":p,"normalization_only":noMC,"stat_only":stat,
                      "independent_full_fit_Z":zother,"independent_fit_success":bool(other.success),
                      "relative_Z_discrepancy":rel,
                      "stat_closed_form_relative_discrepancy":abs(stat["Z"]-zclosed)/zclosed,
                      "alternate_initialization":alternate,
                      "nested_Z_order":p["Z"] <= noMC["Z"]+1e-8 <= stat["Z"]+2e-8}
    checks={"all_profile_converged":all(v["profile"]["success"] and v["normalization_only"]["success"] and v["stat_only"]["success"] for v in results.values()),
            "all_independent_fits_converged":all(v["independent_fit_success"] for v in results.values()),
            "max_relative_Z_discrepancy":max(v["relative_Z_discrepancy"] for v in results.values()),
            "all_nested_Z_order":all(v["nested_Z_order"] for v in results.values()),
            "max_stat_closed_form_discrepancy":max(v["stat_closed_form_relative_discrepancy"] for v in results.values()),
            "alternate_initializations_pass":all(v["alternate_initialization"]["success"] and v["alternate_initialization"]["relative_Z_discrepancy"]<1e-5 for v in results.values() if v["alternate_initialization"])}
    Path(args.out).write_text(json.dumps({"checks":checks,"results":results},indent=2)+"\n")
    print(json.dumps(checks))
    if not checks["all_profile_converged"] or not checks["all_independent_fits_converged"] or checks["max_relative_Z_discrepancy"]>1e-5 or not checks["all_nested_Z_order"] or checks["max_stat_closed_form_discrepancy"]>1e-9 or not checks["alternate_initializations_pass"]:
        raise RuntimeError("Numerical validation incomplete")


if __name__=="__main__":main()
