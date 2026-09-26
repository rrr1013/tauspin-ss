"""q5: P_tau as a physics parameter, and the three measurements side by side.

sin^2 theta_eff:   sigma(sin^2) = sigma(P_tau) / |dP_tau/dsin^2|,  dP/ds2 = 7.87.
LEP reference:     A_tau = 0.1439 +- 0.0043 combined, of which about 0.0035 is
                   statistical and 0.0026 systematic.  0.0043 is the TOTAL, so
                   "matched at N events" below compares an idealised signal-only
                   shape uncertainty against LEP's total; matching only LEP's
                   statistical component needs (0.0043/0.0035)^2 = 1.5x more.
Signal events only: no background, no trigger or identification efficiency, no
energy scale, no cross section, no luminosity.  These event counts are NOT an
LHC reach projection; they are the number of *selected signal* events at which
the idealised shape uncertainty reaches a given value.

The second half of the file places this run next to the 2026-09-25 CP-mixing run
and the 2026-09-26 entanglement run, converting each to the same currency: the
ratio of the reconstructed to the exact-h uncertainty, and what the geometry
blocks buy.  The CP and entanglement numbers are quoted from those run notes.
"""
import json

import numpy as np

import pol_data as PD
from pol_density import dp_dsin2w, p_tau_of_sin2w

LEP_A_TAU_ERR = 0.0043       # total
LEP_A_TAU_STAT = 0.0035      # statistical component of the LEP combination
DPDS = dp_dsin2w()

q1 = json.loads((PD.RESULTS / 'q1_sensitivity.json').read_text())
q2 = json.loads((PD.RESULTS / 'q2_classical.json').read_text())
q4 = json.loads((PD.RESULTS / 'q4_transport.json').read_text())
N0 = q1['n_ref']

out = {'dP_dsin2w': DPDS, 'P_tau_sample': p_tau_of_sin2w(0.23152),
       'lep_sigma_P': LEP_A_TAU_ERR, 'lep_sigma_P_stat': LEP_A_TAU_STAT,
       'lep_sigma_sin2': LEP_A_TAU_ERR / abs(DPDS), 'n_ref': N0}
print(f'dP_tau/dsin2theta = {DPDS:.3f};  LEP sigma(A_tau) = {LEP_A_TAU_ERR}'
      f' -> sigma(sin2) = {LEP_A_TAU_ERR/abs(DPDS):.2e}')

print('\nsigma(P_tau) and sigma(sin^2 theta_eff) per %d Z events, and the number of'
      ' Z events needed to match LEP:' % N0)
ladder = {}
for key, label in (('exact_score', 'exact h, Cramer-Rao'),
                   ('exact_hk', 'exact h, h_k sum'),
                   ('idealip22_s42_hk', 'reco + ideal-IP oracle'),
                   ('full22_s42_hk', 'reco + 3 IP + SV'),
                   ('base_s43_hk', 'reco, no geometry'),
                   ('full22_shuffle_s42_hk', 'reco + shuffled geometry')):
    s = q1['rows'][key]['sigma']
    ladder[key] = {'label': label, 'sigma_P': s, 'sigma_sin2': s / abs(DPDS),
                   'n_to_match_lep': N0 * (s / LEP_A_TAU_ERR) ** 2}
    print(f'  {label:30s} sigma(P) {s:.5f}  sigma(sin2) {s/abs(DPDS):.2e}'
          f'  N(LEP) {ladder[key]["n_to_match_lep"]:.3g}')
for key, label in (('x_truth', 'E_vis/E_tau, truth E_tau: single variable'),
                   ('vismet_best', 'best visible + MET readout'),
                   ('vis_best', 'best visible-only readout')):
    s = q2['rows'][key]['sigma']
    ladder[key] = {'label': label, 'sigma_P': s, 'sigma_sin2': s / abs(DPDS),
                   'n_to_match_lep': N0 * (s / LEP_A_TAU_ERR) ** 2}
    print(f'  {label:30s} sigma(P) {s:.5f}  sigma(sin2) {s/abs(DPDS):.2e}'
          f'  N(LEP) {ladder[key]["n_to_match_lep"]:.3g}')
out['ladder'] = ladder

n_lep = ladder['full22_s42_hk']['n_to_match_lep']
n_lep_stat = N0 * (ladder['full22_s42_hk']['sigma_P'] / LEP_A_TAU_STAT) ** 2
out['lep_equivalence'] = {'n_for_lep_total': n_lep, 'n_for_lep_statistical': n_lep_stat}
print(f'\nThe idealised signal-only shape uncertainty of reco + IP/SV equals the'
      f' LEP total uncertainty at {n_lep:.3g} selected Z events and the LEP'
      f' statistical component at {n_lep_stat:.3g}.  Not a reach projection.')

# The tolerance on the acceptance first moment is the transferable statement;
# the transport bias measured in q4 is only a ~2 sigma indication on this sample
# and is consistent with zero once the control is reweighted in data-available
# variables (q8), so no "systematics limited by X" factor is quoted.
req = q4.get('requirement', {})
if req:
    out['acceptance_tolerance'] = {
        'dP_dE0_partial_per_side': q4['dP_hat_dE0_hk'],
        'dP_dE0_total_attribution_per_side': q4.get('dP_hat_dE0_hk_total_attribution'),
        'needed_E0_accuracy_partial': req['needed_E0_accuracy'],
        'needed_E0_accuracy_total': req.get('needed_E0_accuracy_total')}
    print('  Transferable statement: E_0[h_k] must be known to'
          f' {req.get("needed_E0_accuracy_total", req["needed_E0_accuracy"])["1e+05"]:.2e}'
          f' at 1e5 events and'
          f' {req.get("needed_E0_accuracy_total", req["needed_E0_accuracy"])["1e+07"]:.2e}'
          ' at 1e7, against its own size of +0.049.')

# ---- the three measurements in the same currency -----------------------------
print('\n--- the same reconstructed h, three different measurements ---')
cp = {'quantity': 'CP mixing angle phi_tau (transverse direction of C)',
      'unit': 'deg per 1e4 H events', 'exact': 0.591, 'base': 1.604,
      'ip_sv': 1.281, 'ideal_ip': 0.934, 'source': 'run 2026-09-25'}
ent = {'quantity': 'entanglement witness <W> (magnitude of C)',
       'unit': 'N(3 sigma) events', 'exact': 46, 'base': 344, 'ip_sv': 240,
       'ideal_ip': 143, 'source': 'run 2026-09-26', 'is_event_count': True}
pol = {'quantity': 'tau polarisation P_tau (longitudinal first moment B)',
       'unit': 'sigma(P) per 1e5 Z events',
       'exact': q1['rows']['exact_hk']['sigma'],
       'base': q1['rows']['base_s43_hk']['sigma'],
       'ip_sv': q1['rows']['full22_s42_hk']['sigma'],
       'ideal_ip': q1['rows']['idealip22_s42_hk']['sigma'],
       'source': 'this run'}
tab = {}
for name, d in (('cp_angle', cp), ('entanglement', ent), ('polarisation', pol)):
    f = (lambda a, b: np.sqrt(b / a)) if d.get('is_event_count') else (lambda a, b: b / a)
    row = dict(d)
    row['reco_over_exact'] = float(f(d['exact'], d['base']))
    row['ip_sv_gain'] = float(f(d['ip_sv'], d['base']))
    row['ideal_ip_gain'] = float(f(d['ideal_ip'], d['base']))
    row['residual_after_ideal_ip'] = float(f(d['exact'], d['ideal_ip']))
    tab[name] = row
    print(f'  {d["quantity"]:52s} reco/exact {row["reco_over_exact"]:.2f}'
          f'   IP+SV gain {row["ip_sv_gain"]:.2f}'
          f'   ideal-IP gain {row["ideal_ip_gain"]:.2f}')
out['three_measurements'] = tab

(PD.RESULTS / 'q5_translate.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q5_translate.json')
