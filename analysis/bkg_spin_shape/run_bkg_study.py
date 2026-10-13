"""Comprehensive study of background spin shapes, instrumental correlations,
kinematic mass drift, and auxiliary transfer systematics in H -> tautau entanglement.

Covers:
1. Instrumental correlation anatomy across models and decay modes
2. Kinematic drift of (g_T, g_L) across m_tautau (9 mass bins) and q_T
3. Transfer systematic scans for auxiliary samples (embedding and fake-factor)
4. 2D tolerance envelope (|Z_ent - 3.02| and false significance Z_false)
5. Mitigated transfer / smooth mass drift model
"""
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.home() / 'Projects' / 'tauspin'
sys.path.insert(0, str(REPO / 'analysis' / 'ip_tau_reco'))
import ent_robust as er

CACHE_FILE = REPO / 'analysis/bkg_spin_shape/results/ent_robust_cache.npz'
DATASET_FILE = Path.home() / 'Projects/tauspin-wt-dihiggs/analysis/ip_tau_reco/outputs/ent/dataset_HU.npz'
OUT_DIR = REPO / 'analysis/bkg_spin_shape/results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def study_part1_instrumental_structure(P, dataset_npz):
    """Part 1: Instrumental correlation anatomy across models and decay modes."""
    print("=== Part 1: Instrumental Correlation Anatomy ===")
    p = P['p'] # (N, 3): h-_i h+_i
    w = P['w']
    
    d = np.load(dataset_npz, allow_pickle=True)
    h = d['h_exact']
    import entanglement_feasibility as ef
    W, s, ok = ef.weights(h, False)
    
    # Physical weights for the three states:
    w_H = w * W['H']
    w_Z = w * W['Z']
    w_U = w * W['U']
    m0 = d['spin_rmode0_0']
    m1 = d['spin_rmode1_0']
    m2 = d['spin_rmode2_0']
    m3 = d['spin_rmode3_0'] # 3-prong
    # Modes: 0=pi, 1=rho, 2=a1(1p2pi0), 3=a1(3p)
    mode0 = np.where(m0 > 0, 0, np.where(m1 > 0, 1, np.where(m2 > 0, 2, 3)))
    
    m0_1 = d['spin_rmode0_1']
    m1_1 = d['spin_rmode1_1']
    m2_1 = d['spin_rmode2_1']
    m3_1 = d['spin_rmode3_1']
    mode1 = np.where(m0_1 > 0, 0, np.where(m1_1 > 0, 1, np.where(m2_1 > 0, 2, 3)))
    
    pairs = {
        'all': np.ones(len(w), dtype=bool),
        'pi x pi': (mode0 == 0) & (mode1 == 0),
        'pi x rho': ((mode0 == 0) & (mode1 == 1)) | ((mode0 == 1) & (mode1 == 0)),
        'rho x rho': (mode0 == 1) & (mode1 == 1),
        '3pi x 3pi': (mode0 == 3) & (mode1 == 3),
    }
    
    results = {}
    for lev in ['exact', 'tauspin', 'tauspin_noIPSV', 'textbook']:
        G = P['levels'][lev]['G'] # (N, 2): (gT, gL)
        res_lev = {}
        for pname, sel in pairs.items():
            wh = w_H[sel]
            wz_s = w_Z[sel]
            wu_s = w_U[sel]
            
            gT = G[sel, 0]
            gL = G[sel, 1]
            
            mean_gT_H = float(np.average(gT, weights=wh))
            mean_gT_Z = float(np.average(gT, weights=wz_s))
            mean_gT_U = float(np.average(gT, weights=wu_s))
            
            mean_gL_H = float(np.average(gL, weights=wh))
            mean_gL_Z = float(np.average(gL, weights=wz_s))
            mean_gL_U = float(np.average(gL, weights=wu_s))
            
            # Instrumental fraction: ratio of background gT to signal gT
            f_inst_Z = mean_gT_Z / mean_gT_H if mean_gT_H != 0 else np.nan
            f_inst_U = mean_gT_U / mean_gT_H if mean_gT_H != 0 else np.nan
            
            res_lev[pname] = {
                'N': int(sel.sum()),
                'gT': {'H': mean_gT_H, 'Z': mean_gT_Z, 'U': mean_gT_U},
                'gL': {'H': mean_gL_H, 'Z': mean_gL_Z, 'U': mean_gL_U},
                'f_inst_Z': float(f_inst_Z),
                'f_inst_U': float(f_inst_U),
                'std_gT': {'H': float(np.sqrt(np.cov(gT, aweights=wh))),
                           'Z': float(np.sqrt(np.cov(gT, aweights=wz_s))),
                           'U': float(np.sqrt(np.cov(gT, aweights=wu_s)))},
                'std_gL': {'H': float(np.sqrt(np.cov(gL, aweights=wh))),
                           'Z': float(np.sqrt(np.cov(gL, aweights=wz_s))),
                           'U': float(np.sqrt(np.cov(gL, aweights=wu_s)))}
            }
        results[lev] = res_lev
        print(f"[{lev}] overall: H gT={res_lev['all']['gT']['H']:.4f}, Z gT={res_lev['all']['gT']['Z']:.4f}, U gT={res_lev['all']['gT']['U']:.4f}, f_inst={res_lev['all']['f_inst_Z']:.3f}")
    
    return results


def study_part2_kinematic_mass_drift(P, dataset_npz):
    """Part 2: Kinematic and mass drift across 9 m_tautau bins and q_T."""
    print("=== Part 2: Kinematic Mass Drift ===")
    d = np.load(dataset_npz, allow_pickle=True)
    m_tt = d['kin_m_tautau']
    pt_tt = d['kin_pt_tautau']
    
    edges = P['acc']['edges'] # [70, 90, 100, 110, 120, 130, 140, 150, 170, 200]
    n_bins = len(edges) - 1
    
    w_H = P['w']
    w_Z = P['wz'] * P['w']
    w_U = P['w']
    
    results = {}
    for lev in ['exact', 'tauspin', 'tauspin_noIPSV']:
        G = P['levels'][lev]['G']
        bn = er.Binner(G, er.NQ)
        b = bn(G)
        
        m_bins_data = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i+1]
            sel = (m_tt >= lo) & (m_tt < hi)
            n_ev = int(sel.sum())
            if n_ev == 0:
                continue
            
            wh = w_H[sel]
            wz_s = w_Z[sel]
            wu_s = w_U[sel]
            
            gT_H = float(np.average(G[sel, 0], weights=wh))
            gT_Z = float(np.average(G[sel, 0], weights=wz_s))
            gT_U = float(np.average(G[sel, 0], weights=wu_s))
            
            gL_H = float(np.average(G[sel, 1], weights=wh))
            gL_Z = float(np.average(G[sel, 1], weights=wz_s))
            gL_U = float(np.average(G[sel, 1], weights=wu_s))
            
            # Spin bin template in 36 bins
            hist_Z = np.bincount(b[sel], weights=wz_s, minlength=bn.nb)
            hist_Z /= hist_Z.sum()
            hist_U = np.bincount(b[sel], weights=wu_s, minlength=bn.nb)
            hist_U /= hist_U.sum()
            hist_H = np.bincount(b[sel], weights=wh, minlength=bn.nb)
            hist_H /= hist_H.sum()
            
            m_bins_data.append({
                'bin': i, 'm_range': [lo, hi], 'm_mid': float(0.5*(lo+hi)), 'N': n_ev,
                'gT': {'H': gT_H, 'Z': gT_Z, 'U': gT_U},
                'gL': {'H': gL_H, 'Z': gL_Z, 'U': gL_U},
                'hist_Z': hist_Z.tolist(),
                'hist_U': hist_U.tolist(),
                'hist_H': hist_H.tolist(),
                'pt_mean': float(np.mean(pt_tt[sel]))
            })
        
        # Linear drift slope for Z and U
        m_mids = np.array([x['m_mid'] for x in m_bins_data])
        gT_Z_arr = np.array([x['gT']['Z'] for x in m_bins_data])
        gT_U_arr = np.array([x['gT']['U'] for x in m_bins_data])
        slope_Z, intercept_Z = np.polyfit(m_mids, gT_Z_arr, 1)
        slope_U, intercept_U = np.polyfit(m_mids, gT_U_arr, 1)
        
        # Sideband vs Signal Region comparison
        # Signal region is bins 3, 4, 5 (110-140 GeV); Sidebands are 0, 1, 2 (70-110) and 6, 7, 8 (140-200)
        sel_SR = (m_tt >= 110) & (m_tt < 140)
        sel_SB = ((m_tt >= 70) & (m_tt < 110)) | ((m_tt >= 140) & (m_tt < 200))
        
        gT_Z_SR = float(np.average(G[sel_SR, 0], weights=w_Z[sel_SR]))
        gT_Z_SB = float(np.average(G[sel_SB, 0], weights=w_Z[sel_SB]))
        gT_U_SR = float(np.average(G[sel_SR, 0], weights=w_U[sel_SR]))
        gT_U_SB = float(np.average(G[sel_SB, 0], weights=w_U[sel_SB]))
        
        sh_Z_SR = np.bincount(b[sel_SR], weights=w_Z[sel_SR], minlength=bn.nb)
        sh_Z_SR /= sh_Z_SR.sum()
        sh_Z_SB = np.bincount(b[sel_SB], weights=w_Z[sel_SB], minlength=bn.nb)
        sh_Z_SB /= sh_Z_SB.sum()
        
        # Total variation distance between SB and SR: 0.5 * sum(|p - q|)
        tvd_Z = float(0.5 * np.sum(np.abs(sh_Z_SR - sh_Z_SB)))
        
        results[lev] = {
            'bins': m_bins_data,
            'slope_gT_Z': float(slope_Z),
            'slope_gT_U': float(slope_U),
            'SR_vs_SB': {
                'gT_Z_SR': gT_Z_SR, 'gT_Z_SB': gT_Z_SB, 'delta_gT_Z': gT_Z_SR - gT_Z_SB,
                'gT_U_SR': gT_U_SR, 'gT_U_SB': gT_U_SB, 'delta_gT_U': gT_U_SR - gT_U_SB,
                'tvd_Z': tvd_Z
            }
        }
        print(f"[{lev}] slope_gT_Z = {slope_Z:.6f} /GeV, TVD(SB, SR) = {tvd_Z:.4f}, gT_Z(SR) - gT_Z(SB) = {gT_Z_SR - gT_Z_SB:+.4f}")
    
    return results


def study_part3_transfer_systematics(P):
    """Part 3: Stress scans of auxiliary transfer systematics (embedding & fakes)."""
    print("=== Part 3: Transfer Systematic Scans ===")
    lev = 'tauspin'
    A0, D, Zsh, Ush, Xr = er.templates_for(lev, er.NQ)
    S, Bg, Bf = er.yields(P['acc'], 1.0)
    
    # Base nominal model
    M_nom = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 1.0, 1.0)
    M_nom.set_aux(Zsh, Ush)
    n_nom = M_nom.asimov(er.TRUE_C, Zsh, Ush)
    r_nom = er.test(M_nom, n_nom, (Zsh, Ush), 'oct', er.TET_STARTS[:1])
    Z_nom = r_nom['Z']
    c_null_nom = r_nom['c_null']
    print(f"Nominal auxiliary (tau=1.0, delta=0): Z_ent = {Z_nom:.3f} sigma, c_null = {np.round(c_null_nom, 3)}")
    
    # Also evaluate sidebands-only baseline
    M_sb = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 0.0, 0.0)
    n_sb = M_sb.asimov(er.TRUE_C, Zsh, Ush)
    r_sb = er.test(M_sb, n_sb, (Zsh, Ush), 'oct', er.TET_STARTS[:1])
    Z_sb = r_sb['Z']
    print(f"Sidebands-only baseline: Z_ent = {Z_sb:.3f} sigma")
    
    # Distortion models:
    # 1. Embedding distortion: mix Zsh with Ush (unpolarized tau embedding) or with shifted template
    # delta_g > 0: auxiliary template has more unpolarized contamination than true signal-region Z
    # delta_g < 0: auxiliary template has less
    deltas = [-0.10, -0.06, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.06, 0.10]
    
    scan_emb = []
    for d_g in deltas:
        # Distorted auxiliary template for genuine tau
        if d_g >= 0:
            Zsh_aux = (1.0 - d_g) * Zsh + d_g * Ush
        else:
            # Shift in opposite direction: sharpen Zsh against Ush
            Zsh_aux = np.clip((1.0 - d_g) * Zsh + d_g * Ush, 1e-9, None)
            Zsh_aux /= Zsh_aux.sum(axis=1, keepdims=True)
            
        M = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 1.0, 1.0)
        M.set_aux(Zsh_aux, Ush)
        # Asimov data is generated with the TRUE shapes (Zsh, Ush)
        n_true = M.asimov(er.TRUE_C, Zsh, Ush)
        r_ent = er.test(M, n_true, (Zsh_aux, Ush), 'oct', er.TET_STARTS[:1])
        
        # Test false positive significance: inject separable null state
        n_null = M.asimov(c_null_nom, Zsh, Ush)
        r_null = er.test(M, n_null, (Zsh_aux, Ush), 'oct', er.TET_STARTS)
        
        scan_emb.append({
            'delta_g': d_g,
            'Z_ent': float(r_ent['Z']),
            'c_null': [float(x) for x in r_ent['c_null']],
            'kT_null': float(r_ent['kT_null']),
            'Z_false': float(r_null['Z']),
            'c_null_false': [float(x) for x in r_null['c_null']]
        })
        print(f"  delta_g={d_g:+.2f}: Z_ent={r_ent['Z']:.3f} sigma, Z_false={r_null['Z']:.3f} sigma")
        
    scan_fake = []
    for d_f in deltas:
        # Distorted auxiliary template for fake tau: shift Ush towards Zsh or generic skew
        if d_f >= 0:
            Ush_aux = (1.0 - d_f) * Ush + d_f * Zsh
        else:
            Ush_aux = np.clip((1.0 - d_f) * Ush + d_f * Zsh, 1e-9, None)
            Ush_aux /= Ush_aux.sum(axis=1, keepdims=True)
            
        M = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 1.0, 1.0)
        M.set_aux(Zsh, Ush_aux)
        n_true = M.asimov(er.TRUE_C, Zsh, Ush)
        r_ent = er.test(M, n_true, (Zsh, Ush_aux), 'oct', er.TET_STARTS[:1])
        
        n_null = M.asimov(c_null_nom, Zsh, Ush)
        r_null = er.test(M, n_null, (Zsh, Ush_aux), 'oct', er.TET_STARTS)
        
        scan_fake.append({
            'delta_f': d_f,
            'Z_ent': float(r_ent['Z']),
            'c_null': [float(x) for x in r_ent['c_null']],
            'kT_null': float(r_ent['kT_null']),
            'Z_false': float(r_null['Z']),
            'c_null_false': [float(x) for x in r_null['c_null']]
        })
        print(f"  delta_f={d_f:+.2f}: Z_ent={r_ent['Z']:.3f} sigma, Z_false={r_null['Z']:.3f} sigma")
    
    # 2D Grid Scan: (delta_g, delta_f) on 7x7 grid
    grid_deltas = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
    grid_2d = []
    print("\n--- 2D Grid Scan (delta_g, delta_f) ---")
    for dg in grid_deltas:
        row = []
        for df in grid_deltas:
            Zsh_aux = np.clip((1.0 - dg) * Zsh + dg * Ush, 1e-9, None)
            Zsh_aux /= Zsh_aux.sum(axis=1, keepdims=True)
            Ush_aux = np.clip((1.0 - df) * Ush + df * Zsh, 1e-9, None)
            Ush_aux /= Ush_aux.sum(axis=1, keepdims=True)
            
            M = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 1.0, 1.0)
            M.set_aux(Zsh_aux, Ush_aux)
            n_true = M.asimov(er.TRUE_C, Zsh, Ush)
            r_ent = er.test(M, n_true, (Zsh_aux, Ush_aux), 'oct', er.TET_STARTS[:1])
            
            n_null = M.asimov(c_null_nom, Zsh, Ush)
            r_null = er.test(M, n_null, (Zsh_aux, Ush_aux), 'oct', er.TET_STARTS)
            
            item = {
                'dg': float(dg), 'df': float(df),
                'Z_ent': float(r_ent['Z']),
                'Z_false': float(r_null['Z']),
                'kT_null': float(r_ent['kT_null']),
                'kL_null': float(r_ent['kL_null'])
            }
            row.append(item)
            sys.stdout.write(f"[{dg:+.2f},{df:+.2f}: Z={r_ent['Z']:.2f}, Zf={r_null['Z']:.2f}] ")
            sys.stdout.flush()
        print()
        grid_2d.append(row)
        
    return {
        'nominal': {'Z_ent': float(Z_nom), 'Z_sb': float(Z_sb), 'c_null': [float(x) for x in c_null_nom]},
        'scan_emb': scan_emb,
        'scan_fake': scan_fake,
        'grid_2d': grid_2d,
        'grid_deltas': grid_deltas
    }


def study_part4_mitigated_framework(P):
    """Part 4: Mitigated transfer with sideband drift calibration."""
    print("=== Part 4: Mitigated Transfer Framework ===")
    lev = 'tauspin'
    A0, D, Zsh, Ush, Xr = er.templates_for(lev, er.NQ)
    S, Bg, Bf = er.yields(P['acc'], 1.0)
    
    # Under real mass drift:
    # Drift genuine tau background from Zsh to Ush linearly with mass
    ti = np.linspace(0, 1, len(P['acc']['S']))
    drift = (Ush, ti)
    
    # 1. Unmitigated sidebands only under mass drift:
    M_sb = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 0.0, 0.0)
    n_drift_true = M_sb.asimov(er.TRUE_C, Zsh, Ush, drift)
    r_sb_drift = er.test(M_sb, n_drift_true, (Zsh, Ush), 'oct', er.TET_STARTS[:1])
    
    # 2. Unmitigated auxiliary (tau=1) with flat auxiliary template:
    M_aux = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 1.0, 1.0)
    M_aux.set_aux(Zsh, Ush)
    r_aux_drift = er.test(M_aux, n_drift_true, (Zsh, Ush), 'oct', er.TET_STARTS[:1])
    
    # 3. Mitigated auxiliary: auxiliary template calibrated with measured mass drift
    # In the signal region (mass bins 3, 4, 5, average t ~ 0.5), the true effective template is
    # Zsh_eff = (1 - t_sr) * Zsh + t_sr * Ush where t_sr ~ 0.5
    t_sr = float(ti[3:6].mean()) # ~ 0.5
    Zsh_mitigated = (1.0 - t_sr) * Zsh + t_sr * Ush
    M_mit = er.Model(A0, D, S, Bg, Bf, 0.1, 0.1, None, 1.0, 1.0)
    M_mit.set_aux(Zsh_mitigated, Ush)
    r_mit = er.test(M_mit, n_drift_true, (Zsh_mitigated, Ush), 'oct', er.TET_STARTS[:1])
    
    # Null injected check for mitigation
    c_null = r_mit['c_null']
    n_drift_null = M_sb.asimov(c_null, Zsh, Ush, drift)
    r_mit_null = er.test(M_mit, n_drift_null, (Zsh_mitigated, Ush), 'oct', er.TET_STARTS)
    
    res = {
        'sb_under_drift': float(r_sb_drift['Z']),
        'aux_unmitigated_under_drift': float(r_aux_drift['Z']),
        'aux_mitigated': float(r_mit['Z']),
        'aux_mitigated_false': float(r_mit_null['Z']),
        'c_null_mitigated': [float(x) for x in r_mit['c_null']]
    }
    print(f"Drift Results: SB={res['sb_under_drift']:.3f}s, UnmitAux={res['aux_unmitigated_under_drift']:.3f}s, Mitigated={res['aux_mitigated']:.3f}s, MitFalse={res['aux_mitigated_false']:.3f}s")
    return res


def main():
    t0 = time.time()
    torch.set_num_threads(3)
    print("Loading cache from", CACHE_FILE)
    er.load_prep(CACHE_FILE)
    P = er.PREP['P']
    
    part1 = study_part1_instrumental_structure(P, DATASET_FILE)
    part2 = study_part2_kinematic_mass_drift(P, DATASET_FILE)
    part3 = study_part3_transfer_systematics(P)
    part4 = study_part4_mitigated_framework(P)
    
    summary = {
        'part1_instrumental': part1,
        'part2_kinematic_drift': part2,
        'part3_transfer_systematics': part3,
        'part4_mitigation': part4,
        'elapsed_sec': time.time() - t0
    }
    
    out_file = OUT_DIR / 'bkg_study_results.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"All studies completed in {time.time() - t0:.1f}s. Saved to {out_file}")


if __name__ == '__main__':
    main()
