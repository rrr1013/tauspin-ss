import sys,json,time,socket,pathlib,importlib,numpy as np
from types import SimpleNamespace
root=pathlib.Path("/home/rbaba/dihiggs-latest-20261008");source=root/"source-inference-9f4b634";out=root/"lephad-inference-9f4b634";oldroot=root/"lephad-slt";prepared=root/"prepared_latest_slt_hilo/prepared.npz"
commit="9f4b6341eae975bb35ea8bbd316d40e41318554b";modelcommit="e34d272e69f1a5f4fe511bf93ce2a8bfa029abfe"
sys.path.insert(0,str(source/"analysis/dihiggs_spin"));import latest_lephad as L
L.verify_snapshot(commit)
D=dict(np.load(prepared,allow_pickle=True));old=json.loads((oldroot/"summary.json").read_text());R=json.loads((out/"summary.json").read_text())
assert len(R)==24
context=["initialization"];fits={};failures=[]
original_profile=L.profile;original_evaluate=L.evaluate;original_bootstrap=L.bootstrap

def traced_profile(*args,**kwargs):
 result=original_profile(*args,**kwargs)
 mode="norm%s_mc%s" % (kwargs.get("norm", True), kwargs.get("mc", True));key=context[0]+"/"+mode
 c=fits.setdefault(key,{"total":0,"success":0,"failure":0});c["total"]+=1;c["success"]+=int(result["success"]);c["failure"]+=int(not result["success"])
 if not result["success"]:failures.append({"context":key,"fit":c["total"],"result":result})
 return result

def flavor_tag(data):
 values=np.unique(data["lep_is_e"][data["sel_nominal"]]);return ("electron" if values[0] else "muon") if len(values)==1 else "mixed"

def traced_evaluate(score,data,variant):
 saved=context[0]
 if saved=="flavor":context[0]="flavor_"+flavor_tag(data)+"_evaluation"
 try:return original_evaluate(score,data,variant)
 finally:context[0]=saved

def traced_bootstrap(args,data,results):
 saved=context[0]
 if saved=="flavor":context[0]="flavor_"+flavor_tag(data)+"_bootstrap"
 try:return original_bootstrap(args,data,results)
 finally:context[0]=saved

L.profile=traced_profile;L.evaluate=traced_evaluate;L.bootstrap=traced_bootstrap
prov={"inference_commit":commit,"model_commit":modelcommit,"retraining_performed":False,"host":socket.gethostname(),"threads":4,"start_unix":time.time(),"source_manifest_sha256":L.sha256(source/"source_snapshot.json"),"source_archive_sha256":L.sha256(root/"lephad-inference-9f4b634.tgz"),"prepared":{"path":str(prepared),"sha256":L.sha256(prepared)},"input":{"path":"/home/rbaba/dihiggs-spin-20261006/runs/lh/dataset_v2.npz","sha256":L.sha256("/home/rbaba/dihiggs-spin-20261006/runs/lh/dataset_v2.npz")},"scores":{},"fit_trace_definition":"Unmodified frozen profile result is returned; external wrapper only counts successes/failures by phase"}
assert prov["prepared"]["sha256"]=="d61908a7794c4784e2280e32381164da4226913bf153ace5b53b10c05cda9813"
for key,r in R.items():
 assert r["status"]["optimization_complete"] and r["status"]["git_commit"]==modelcommit
 with np.load(r["score_path"]) as z:
  assert np.array_equal(z["uid"],D["uid"]) and np.array_equal(z["proc"],D["proc"])
  variant=key.split("/")[0];assert np.array_equal(z["selected"],D["sel_"+variant]);score=z["score"]
  assert np.isfinite(score[D["sel_"+variant]]).all()
 prov["scores"][key]={"path":r["score_path"],"sha256":L.sha256(r["score_path"]),"identity_and_selection_verified":True}
 context[0]="evaluation/"+key;r["evaluation"]=L.evaluate(score,D,variant);r["inference_commit"]=commit
 for category in ("Hi","Lo"):
  assert old[key]["evaluation"]["categories"][category]["edges"]==r["evaluation"]["categories"][category]["edges"]
  assert old[key]["evaluation"]["categories"][category]["evaluation"]["poor_bins"]==r["evaluation"]["categories"][category]["evaluation"]["poor_bins"]
prov["old_new_bins_exact"]=True;prov["old_mc_support_flags_unchanged"]=True
L.dump(out/"summary.json",R);L.dump(out/"aggregate.json",L.aggregate(R));L.dump(out/"inference_provenance.json",prov)
L.dump(out/"fit_diagnostics.json",{"counts":fits,"failures":failures})
print("EVALUATION_COMPLETE",len(R),"fit_failures",len(failures),flush=True)
if failures:raise SystemExit(3)
context[0]="nominal_bootstrap";nominal={k:v for k,v in R.items() if "GBDT" not in k};bt=L.bootstrap(SimpleNamespace(bootstrap=100),D,nominal)
for v in bt.values():v["failed_ratio_replicates"]=v["requested"]-v["n_valid"]
L.dump(out/"bootstrap.json",bt);print("NOMINAL_BOOTSTRAP_COMPLETE",{k:v["n_valid"] for k,v in bt.items()},flush=True)
context[0]="flavor";F=importlib.import_module("latest_flavor_probe");sys.argv=[str(source/"analysis/dihiggs_spin/latest_flavor_probe.py"),"--prepared",str(prepared),"--run",str(out),"--out",str(out/"flavor"),"--bootstrap","100"];F.main()
flavor=json.loads((out/"flavor/flavor.json").read_text())
for f in flavor["flavors"].values():
 for b in f["bootstrap"].values():b["failed_ratio_replicates"]=b["requested"]-b["n_valid"]
L.dump(out/"flavor/flavor.json",flavor);print("FLAVOR_COMPLETE",flush=True)
context[0]="gbdt_bootstrap";pair={"nominal/B/0":R["nominal/GBDT_B/0"],"nominal/B+d0/0":R["nominal/GBDT_B+d0/0"]};gb=L.bootstrap(SimpleNamespace(bootstrap=100),D,pair)
for v in gb.values():v["failed_ratio_replicates"]=v["requested"]-v["n_valid"]
L.dump(out/"gbdt_bootstrap.json",{"definition":"Diagnostic GBDT pair; baseline Lo bin7 remains MC-support flagged","key_mapping":{"nominal/B/0":"nominal/GBDT_B/0","nominal/B+d0/0":"nominal/GBDT_B+d0/0"},"baseline_sensitivity_resolved":R["nominal/GBDT_B/0"]["evaluation"]["sensitivity_resolved"],"bootstrap":gb})
L.dump(out/"fit_diagnostics.json",{"counts":fits,"failures":failures,"total_fits":sum(v["total"] for v in fits.values()),"total_failures":len(failures)})
L.verify_snapshot(commit);prov["end_unix"]=time.time();prov["all_bootstrap_ratios_valid"]=all(v["n_valid"]==100 for v in bt.values()) and all(b["n_valid"]==100 for f in flavor["flavors"].values() for b in f["bootstrap"].values()) and all(v["n_valid"]==100 for v in gb.values());L.dump(out/"inference_provenance.json",prov)
print("REEVALUATION_COMPLETE",prov["all_bootstrap_ratios_valid"],"total_fit_failures",len(failures),flush=True)
if failures or not prov["all_bootstrap_ratios_valid"]:raise SystemExit(4)
