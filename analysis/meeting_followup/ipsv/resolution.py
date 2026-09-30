"""IP / SV resolution of the simple-smearing and ATLAS full-reco samples with one definition.

Validation h-valid rows of both samples (build_simple_inputs.py / build_atlas_geometry.py).
For the leading charged track of each tau (p = measured track direction, t = true tau direction):
  * the true IP direction is u_true = unit(t - (t.p) p); the measured one u = unit(IP);
    dphi = signed angle from u_true to u around p  (IP azimuth error);
  * tau-direction error carried by the IP: |dphi| * angle(t, p)  [mrad];
  * |IP| [um].
For reco 3-prong taus with an SV: angle between the PV->SV direction and t [mrad], and L [mm].
Also the reference: angle between the reco visible axis and t (what IP/SV have to beat).

usage: python resolution.py SIMPLE_INPUTS ATLAS_GEOMETRY OUTDIR
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def quantities(ip, p, t, sv, sv_ok, prong3, vis):
    u_true = unit(t - np.sum(t * p, -1, keepdims=True) * p)
    u = unit(ip)
    c = np.clip(np.sum(u * u_true, -1), -1, 1)
    s = np.sum(np.cross(u_true, u) * p, -1)
    dphi = np.arctan2(s, c)
    cone = np.arccos(np.clip(np.sum(t * p, -1), -1, 1))
    ok = np.isfinite(dphi) & (np.linalg.norm(ip, axis=-1) > 0)
    sv_ang = np.arccos(np.clip(np.sum(unit(sv) * t, -1), -1, 1))
    vis_ang = np.arccos(np.clip(np.sum(unit(vis) * t, -1), -1, 1))
    return dict(dphi=dphi[ok], prong3=prong3[ok], dtau=(np.abs(dphi) * cone * 1e3)[ok],
                ip_um=(np.linalg.norm(ip, axis=-1) * 1e3)[ok], cone_mrad=(cone * 1e3)[ok],
                sv_mrad=(sv_ang * 1e3)[sv_ok], sv_len=np.linalg.norm(sv, axis=-1)[sv_ok],
                vis_mrad=(vis_ang * 1e3), vis_prong3=prong3)


def main(simple_path, atlas_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    s = np.load(simple_path)
    a = np.load(atlas_path)
    Q = {}
    # simple: one IP per tau (leading track); basis row k = reco visible direction
    Q['simple'] = quantities(s['res_ip'], s['res_track_dir'], s['res_tau_true'], s['res_sv'], s['res_sv_ok'],
                             s['res_reco_digit'] == 3, s['res_basis'][:, :, 2])
    # ATLAS: leading core track (index 0)
    ip0 = a['res_ip'][:, :, 0]
    ok0 = a['res_ip_ok'][:, :, 0]
    ip0 = np.where(ok0[..., None], ip0, 0.0)
    Q['atlas'] = quantities(np.nan_to_num(ip0), a['res_track_dir'][:, :, 0], a['res_tau_true'], a['res_sv'], a['res_sv_ok'],
                            a['res_n_core'] == 3, a['res_basis'][:, :, 2])
    summ = {}
    for k, q in Q.items():
        r = {}
        for name, m in (('1-prong', ~q['prong3']), ('3-prong', q['prong3'])):
            r[name] = dict(n=int(m.sum()), dphi_median_abs=float(np.median(np.abs(q['dphi'][m]))),
                           dphi_mean_cos=float(np.mean(np.cos(q['dphi'][m]))),
                           dtau_median_mrad=float(np.median(q['dtau'][m])),
                           ip_median_um=float(np.median(q['ip_um'][m])),
                           cone_median_mrad=float(np.median(q['cone_mrad'][m])),
                           vis_axis_median_mrad=float(np.median(q['vis_mrad'][q['vis_prong3'] == (name == '3-prong')])))
        r['sv'] = dict(n=int(q['sv_mrad'].size), angle_median_mrad=float(np.median(q['sv_mrad'])),
                       angle_q16_q84=np.quantile(q['sv_mrad'], [.16, .84]).tolist(), length_median_mm=float(np.median(q['sv_len'])))
        summ[k] = r
    (outdir / 'resolution.json').write_text(json.dumps(summ, indent=1))

    lab = {'simple': 'simple smearing (Chen)', 'atlas': 'ATLAS full reco'}
    sty = {'simple': dict(color='C1', ls='-'), 'atlas': dict(color='C0', ls='--')}
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    b = np.linspace(-np.pi, np.pi, 61)
    for k, q in Q.items():
        for pr, lw in ((False, 1.8), (True, 1.0)):
            m = q['prong3'] == pr
            ax[0].hist(q['dphi'][m], b, density=True, histtype='step', lw=lw, **sty[k],
                       label=f"{lab[k]}, {'3' if pr else '1'}-prong (median |Δφ| {summ[k]['3-prong' if pr else '1-prong']['dphi_median_abs']:.2f})")
    ax[0].set_xlabel('IP azimuth error Δφ around the leading track [rad]')
    ax[0].set_ylabel('density')
    ax[0].set_title('(a) direction of the IP (vs. true τ direction)')
    ax[0].legend(fontsize=7)
    b = np.logspace(-3.5, 2, 67)
    for k, q in Q.items():
        m = ~q['prong3']
        ax[1].hist(q['dtau'][m], b, density=True, histtype='step', lw=1.8, **sty[k],
                   label=f"{lab[k]}: IP, median {summ[k]['1-prong']['dtau_median_mrad']:.2f}")
        ax[1].hist(q['vis_mrad'][~q['vis_prong3']], b, density=True, histtype='step', lw=1.0, color=sty[k]['color'], ls=':',
                   label=f"{lab[k]}: visible axis, median {summ[k]['1-prong']['vis_axis_median_mrad']:.2f}")
    ax[1].set_xscale('log')
    ax[1].set_xlabel('transverse τ-direction error [mrad]')
    ax[1].set_title('(b) 1-prong: τ direction from the IP\n|Δφ| × angle(τ, track) vs visible-axis error')
    ax[1].legend(fontsize=7)
    b = np.logspace(0, 3.3, 61)
    for k, q in Q.items():
        for pr, lw in ((False, 1.8), (True, 1.0)):
            m = q['prong3'] == pr
            ax[2].hist(q['ip_um'][m], b, density=True, histtype='step', lw=lw, **sty[k],
                       label=f"{lab[k]}, {'3' if pr else '1'}-prong")
    ax[2].set_xscale('log')
    ax[2].set_xlabel('|IP| of the leading track [μm]')
    ax[2].set_title('(c) IP magnitude')
    ax[2].legend(fontsize=7)
    b = np.logspace(-2, 3.2, 63)
    for k, q in Q.items():
        ax[3].hist(q['sv_mrad'], b, density=True, histtype='step', lw=1.8, **sty[k],
                   label=f"{lab[k]}: PV→SV, median {summ[k]['sv']['angle_median_mrad']:.1f}")
        ax[3].hist(q['vis_mrad'][q['vis_prong3']], b, density=True, histtype='step', lw=1.0, color=sty[k]['color'], ls=':',
                   label=f"{lab[k]}: visible axis, median {summ[k]['3-prong']['vis_axis_median_mrad']:.2f}")
    ax[3].set_xscale('log')
    ax[3].set_xlabel('angle to the true τ direction [mrad]')
    ax[3].set_title('(d) reco 3-prong: SV direction')
    ax[3].legend(fontsize=7)
    fig.text(0.5, -0.03, f"Validation h-valid events (π/ρ/3π truth on both sides), unit weight: simple {len(s['res_reco_digit']):,}, "
             f"ATLAS {len(a['res_n_core']):,} events. ATLAS IP: track perigee + beam spot − PV (no track error matrix, PV includes τ tracks); "
             "simple IP: 25 μm/component Gaussian, SV: 0.5 mm/component.", ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_ipsv_resolution.png', dpi=150, bbox_inches='tight')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
