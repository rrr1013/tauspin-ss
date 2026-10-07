"""Shower an LHE file with Pythia 8 and store the truth needed by the
bb tau tau spin study as fixed-size numpy arrays.

Tau leptons are decayed by Pythia (TauDecays:mode = 4, spin correlations from
the mother h, Z or W found in the event record).  pi0 are kept stable so that
they play the role of calorimeter pi0 clusters.  Stored per event:
  * taus (last copy): 4-momentum, charge, production / decay vertex [mm],
    mother id, decay products split into charged hadrons (<=3), pi0 (<=2),
    other neutral visibles (summed), neutrinos (summed), leptons (e/mu);
  * anti-kt R=0.4 jets (pT > 20 GeV, |eta| < 4.5) built from final-state
    visibles except tau descendants and prompt e/mu, with b / c hadron labels;
  * for tau-fake modelling (--fakes): the core (dR<0.2) charged hadrons
    (<=4, with production vertices), core pi0 (<=3), core neutral energy and
    isolation (0.2<dR<0.4) of each jet with |eta|<2.5;
  * truth missing momentum (all neutrinos), prompt e/mu.
"""
import argparse
import math

import numpy as np
import fastjet
import pythia8

NTAU, NCH, NPI0 = 2, 3, 2
NJET, NFCH, NFPI0 = 10, 4, 3


def four(p):
    return (p.px(), p.py(), p.pz(), p.e())


def is_b_hadron(aid):
    return (aid // 100) % 10 == 5 or (aid // 1000) % 10 == 5


def is_c_hadron(aid):
    return ((aid // 100) % 10 == 4 or (aid // 1000) % 10 == 4) and not is_b_hadron(aid)


def descendants(ev, i, out):
    for d in ev[i].daughterList():
        if ev[d].isFinal():
            out.append(d)
        else:
            descendants(ev, d, out)


def mother_id(ev, i):
    top = ev[i].iTopCopyId()
    m = ev[top].mother1()
    return ev[m].id() if m > 0 else 0


def delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(eta1 - eta2, dphi)


def setup(lhe, seed, proc, spinflat=False):
    py = pythia8.Pythia("", False)
    cmds = [
        "Beams:frameType = 4", f"Beams:LHEF = {lhe}",
        "Random:setSeed = on", f"Random:seed = {seed}",
        "111:mayDecay = off",              # pi0 kept as clusters
        "TauDecays:mode = 4",              # internal helicity MEs only: MG5 writes SPINUP = 0 for h,
        #                                    which mode 1 misroutes and decorrelates H -> tau tau
        "ParticleDecays:limitTau0 = off",
        "Next:numberCount = 0", "Init:showChangedSettings = off",
        "Init:showChangedParticleData = off",
    ]
    if spinflat:
        # TauSpinner-style reference: taus decayed unpolarised and uncorrelated (generator density 1)
        cmds = [c for c in cmds if not c.startswith("TauDecays:mode")]
        cmds += ["TauDecays:mode = 3", "TauDecays:tauPolarization = 0."]
    if proc in ("hh", "hhU"):
        # one H -> bb and one H -> tautau is selected later; equal BRs raise the yield
        cmds += ["25:oneChannel = 1 0.5 100 5 -5", "25:addChannel = 1 0.5 100 15 -15"]
    elif proc in ("zh", "tth"):
        cmds += ["25:oneChannel = 1 1.0 100 15 -15"]
    for c in cmds:
        py.readString(c)
    py.init()
    return py


def empty(n):
    return {
        "weight": np.zeros(n), "n_h_bb": np.zeros(n, np.int8), "n_h_tt": np.zeros(n, np.int8),
        "tau_p4": np.zeros((n, NTAU, 4)), "tau_q": np.zeros((n, NTAU), np.int8),
        "tau_vprod": np.zeros((n, NTAU, 3)), "tau_vdec": np.zeros((n, NTAU, 3)),
        "tau_mother": np.zeros((n, NTAU), np.int32), "tau_ok": np.zeros((n, NTAU), bool),
        "tau_nch": np.zeros((n, NTAU), np.int8), "tau_npi0": np.zeros((n, NTAU), np.int8),
        "tau_nlep": np.zeros((n, NTAU), np.int8), "tau_nother": np.zeros((n, NTAU), np.int8),
        "tau_ch_p4": np.zeros((n, NTAU, NCH, 4)), "tau_ch_q": np.zeros((n, NTAU, NCH), np.int8),
        "tau_pi0_p4": np.zeros((n, NTAU, NPI0, 4)), "tau_oth_p4": np.zeros((n, NTAU, 4)),
        "tau_nu_p4": np.zeros((n, NTAU, 4)), "tau_lep_p4": np.zeros((n, NTAU, 4)),
        "tau_lep_id": np.zeros((n, NTAU), np.int32),
        "jet_p4": np.zeros((n, NJET, 4)), "jet_flav": np.zeros((n, NJET), np.int8),
        "met_true": np.zeros((n, 2)), "nlep_prompt": np.zeros(n, np.int8),
        "lep_p4": np.zeros((n, 2, 4)), "lep_id": np.zeros((n, 2), np.int32),
        "fk_ch_p4": np.zeros((n, NJET, NFCH, 4)), "fk_ch_q": np.zeros((n, NJET, NFCH), np.int8),
        "fk_ch_v": np.zeros((n, NJET, NFCH, 3)), "fk_nch": np.zeros((n, NJET), np.int8),
        "fk_pi0_p4": np.zeros((n, NJET, NFPI0, 4)), "fk_npi0": np.zeros((n, NJET), np.int8),
        "fk_neu_e": np.zeros((n, NJET)), "fk_iso_pt": np.zeros((n, NJET)),
        "fk_iso_n": np.zeros((n, NJET), np.int8),
    }


def process_event(ev, out, k, fakes, jetdef):
    # Higgs decays (for the hh filter)
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 25 and not any(ev[d].id() == 25 for d in p.daughterList()):
            ds = {abs(ev[d].id()) for d in p.daughterList()}
            if 5 in ds:
                out["n_h_bb"][k] += 1
            if 15 in ds:
                out["n_h_tt"][k] += 1
    # taus
    tau_desc = set()
    taus = [i for i in range(ev.size()) if abs(ev[i].id()) == 15
            and not any(abs(ev[d].id()) == 15 for d in ev[i].daughterList())
            and len(ev[i].daughterList()) > 0]
    # taus from h / Z / W first (b- and c-hadron decays also produce taus)
    taus.sort(key=lambda i: (abs(mother_id(ev, i)) not in (23, 24, 25), -ev[i].pT()))
    for t, i in enumerate(taus[:NTAU]):
        p = ev[i]
        out["tau_ok"][k, t] = True
        out["tau_p4"][k, t] = four(p)
        out["tau_q"][k, t] = -1 if p.id() == 15 else 1
        out["tau_vprod"][k, t] = (p.xProd(), p.yProd(), p.zProd())
        out["tau_vdec"][k, t] = (p.xDec(), p.yDec(), p.zDec())
        out["tau_mother"][k, t] = mother_id(ev, i)
        fin = []
        descendants(ev, i, fin)
        tau_desc.update(fin)
        nch = npi0 = 0
        for d in fin:
            q, aid = ev[d], abs(ev[d].id())
            if aid in (12, 14, 16):
                out["tau_nu_p4"][k, t] += four(q)
            elif aid in (11, 13):
                out["tau_lep_p4"][k, t] += four(q)
                out["tau_lep_id"][k, t] = q.id()
                out["tau_nlep"][k, t] += 1
            elif aid == 111:
                if npi0 < NPI0:
                    out["tau_pi0_p4"][k, t, npi0] = four(q)
                npi0 += 1
            elif q.isCharged():
                if nch < NCH:
                    out["tau_ch_p4"][k, t, nch] = four(q)
                    out["tau_ch_q"][k, t, nch] = int(round(q.charge()))
                nch += 1
            else:
                out["tau_oth_p4"][k, t] += four(q)
                out["tau_nother"][k, t] += 1
        out["tau_nch"][k, t] = nch
        out["tau_npi0"][k, t] = npi0
    # prompt leptons, truth MET, jet inputs, hadron labels
    met = [0.0, 0.0]
    nlep = 0
    inputs, bhad, chad = [], [], []
    for i in range(ev.size()):
        p = ev[i]
        aid = p.idAbs()
        if is_b_hadron(aid) and p.pT() > 5 and not any(is_b_hadron(ev[d].idAbs()) for d in p.daughterList()):
            bhad.append((p.eta(), p.phi()))
        elif is_c_hadron(aid) and p.pT() > 5 and not any(is_c_hadron(ev[d].idAbs()) for d in p.daughterList()):
            chad.append((p.eta(), p.phi()))
        if not p.isFinal():
            continue
        if aid in (12, 14, 16) or (aid > 1000000):
            met[0] += p.px()
            met[1] += p.py()
            continue
        if i in tau_desc:
            continue
        if aid in (11, 13) and p.pT() > 7:
            m = ev[p.iTopCopyId()].mother1()
            if m > 0 and ev[m].idAbs() in (23, 24, 15):
                if nlep < 2:
                    out["lep_p4"][k, nlep] = four(p)
                    out["lep_id"][k, nlep] = p.id()
                nlep += 1
                continue
        if abs(p.eta()) < 4.9:
            pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_index(i)
            inputs.append(pj)
    out["met_true"][k] = met
    out["nlep_prompt"][k] = nlep
    cs = fastjet.ClusterSequence(inputs, jetdef)
    jets = [j for j in fastjet.sorted_by_pt(cs.inclusive_jets(20.0)) if abs(j.eta()) < 4.5][:NJET]
    for jj, j in enumerate(jets):
        out["jet_p4"][k, jj] = (j.px(), j.py(), j.pz(), j.e())
        eta, phi = j.eta(), j.phi_std()
        if any(delta_r(eta, phi, a, b) < 0.3 for a, b in bhad):
            out["jet_flav"][k, jj] = 5
        elif any(delta_r(eta, phi, a, b) < 0.3 for a, b in chad):
            out["jet_flav"][k, jj] = 4
        if not fakes or abs(eta) > 2.5:
            continue
        nch = npi0 = 0
        ch = []
        for c in j.constituents():
            q = ev[c.user_index()]
            dr = delta_r(eta, phi, q.eta(), q.phi())
            if dr < 0.2:
                if q.idAbs() == 111:
                    if npi0 < NFPI0:
                        out["fk_pi0_p4"][k, jj, npi0] = four(q)
                    npi0 += 1
                elif q.isCharged() and q.pT() > 1.0:
                    ch.append(c.user_index())
                elif not q.isCharged():
                    out["fk_neu_e"][k, jj] += q.e()
            elif dr < 0.4 and q.isCharged() and q.pT() > 1.0:
                out["fk_iso_pt"][k, jj] += q.pT()
                out["fk_iso_n"][k, jj] += 1
        ch.sort(key=lambda i: -ev[i].pT())
        for n, i in enumerate(ch[:NFCH]):
            out["fk_ch_p4"][k, jj, n] = four(ev[i])
            out["fk_ch_q"][k, jj, n] = int(round(ev[i].charge()))
            out["fk_ch_v"][k, jj, n] = (ev[i].xProd(), ev[i].yProd(), ev[i].zProd())
        out["fk_nch"][k, jj] = len(ch)
        out["fk_npi0"][k, jj] = npi0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lhe", required=True)
    ap.add_argument("--proc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=10 ** 9)
    ap.add_argument("--fakes", action="store_true")
    ap.add_argument("--spinflat", action="store_true", help="unpolarised, uncorrelated tau decays")
    args = ap.parse_args()
    py = setup(args.lhe, args.seed, args.proc, args.spinflat)
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    chunk = 20000
    out = empty(chunk)
    store, k, nfail = [], 0, 0
    for n in range(args.nmax):
        if not py.next():
            if py.infoPython().atEndOfFile():
                break
            nfail += 1
            continue
        out["weight"][k] = py.infoPython().weight()
        process_event(py.event, out, k, args.fakes, jetdef)
        k += 1
        if k == chunk:
            store.append(out)
            out, k = empty(chunk), 0
    store.append({key: v[:k] for key, v in out.items()})
    merged = {key: np.concatenate([s[key] for s in store]) for key in store[0]}
    if not args.fakes:
        merged = {key: v for key, v in merged.items() if not key.startswith("fk_")}
    merged["sigma_pb"] = np.array(py.infoPython().sigmaGen() * 1e9)
    merged["n_fail"] = np.array(nfail)
    np.savez_compressed(args.out, **merged)
    print(f"stored {len(merged['weight'])} events, sigmaGen {merged['sigma_pb']:.4g} pb, failures {nfail}")


if __name__ == "__main__":
    main()
