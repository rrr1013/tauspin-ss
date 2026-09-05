#!/usr/bin/env python3
"""Validation-only descriptive spin-1 mass-reference diagnostic (no fit).

Kuehn, Acta Phys. Pol. B29 (1998) 1371, section 2.  Applying the
direction-only coefficient to reconstructed masses is a reference transform,
not a measurement of spin response or classifier information.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

MTAU = 1.77693  # PDG 2025, GeV
MPI = 0.1396  # frozen source track mass convention, GeV


def alpha(q, mtau=MTAU):
    return (mtau**2 - 2*q*q)/(mtau**2 + 2*q*q)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def raw_masses(payload, stats):
    """Independent pairwise invariant calculation, not E_total^2-p_total^2."""
    f = payload['track_features'].numpy().astype(float)
    for j, enabled in enumerate(stats['standardize']):
        if enabled:
            f[:, j] = f[:, j]*stats['std'][j]+stats['mean'][j]
    names = stats['names']
    get = lambda name: f[:, names.index(name)]
    pt = np.expm1(get('log1p_track_pt'))
    eta = get('track_eta')
    # Source sin/cos are float32; preserve their radius in the independent
    # momentum construction to match the actual stored representation.
    px = pt*get('cos_track_phi')
    py = pt*get('sin_track_phi')
    pz = pt*np.sinh(eta)
    p = np.column_stack((px, py, pz))
    e = np.sqrt(np.sum(p*p, axis=1)+MPI**2)
    offsets = payload['track_offsets'].numpy()
    sides = payload['track_sides'].numpy()
    out = np.empty((len(offsets)-1, 2))
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        for side in (0, 1):
            ix = np.arange(start, stop)[sides[start:stop] == side]
            assert len(ix) == 3
            charges = np.rint(get('track_charge')[ix]).astype(int)
            assert sorted(charges.tolist()) == ([-1, -1, 1] if side == 0 else [-1, 1, 1])
            mass2 = 3*MPI**2
            for a, b in ((0, 1), (0, 2), (1, 2)):
                u, v = ix[a], ix[b]
                mass2 += 2*(e[u]*e[v]-np.dot(p[u], p[v]))
            assert mass2 > 0
            out[i, side] = np.sqrt(mass2)
    return out


def moments(b):
    pos = np.maximum(b, 0).mean()
    neg = np.minimum(b, 0).mean()
    mean = b.mean()
    absolute = np.abs(b).mean()
    return dict(mean_b=float(mean), mean_abs_b=float(absolute),
                rms_b=float(np.sqrt(np.mean(b*b))),
                positive_contribution=float(pos), negative_contribution=float(neg),
                signed_survival=float(abs(mean)/absolute),
                negative_fraction=float(np.mean(b < 0)))


def bootstrap(b):
    # Resample events, preserving the paired legs; conditional finite-sample
    # intervals, not detector/model/systematic uncertainty.
    rng = np.random.default_rng(20260905)
    reps = {k: [] for k in moments(b)}
    for _ in range(2000):
        m = moments(b[rng.integers(len(b), size=len(b))])
        for k, v in m.items():
            reps[k].append(v)
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in reps.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--stats', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.dataset/'metadata.json').read_text())
    stats = json.loads(args.stats.read_text())['track']
    assert set(meta['shards']) == {'train', 'validation'}
    assert meta['test_split_loaded'] is False
    assert meta['feature_names']['track'] == stats['names']
    assert meta['derived_features']['source_stats_sha256'] == digest(args.stats)
    qi = meta['feature_names']['tau'].index('bg_Q')
    st = meta['derived_features']['standardization']['bg_Q']
    data = {}
    audit = {'metadata_sha256': digest(args.dataset/'metadata.json'),
             'stats_sha256': digest(args.stats), 'script_sha256': digest(Path(__file__)),
             'input_files': [], 'test_loaded': False, 'tau_mass_gev': MTAU,
             'pion_mass_gev': MPI, 'zero_gev': MTAU/np.sqrt(2),
             'weighting': 'unit event weight, H and Z separately',
             'bootstrap_caveat': 'iid stored-row resampling approximation; generator-event independence and interval coverage unvalidated',
             'numpy': np.__version__, 'torch': torch.__version__}
    for sample in ('H', 'Z'):
        qs, raws, ids, row_ids, fingerprints = [], [], [], [], []
        for rec in meta['shards']['validation'][sample]:
            path = args.dataset/rec['path']
            assert Path(rec['path']).parts[0] == 'validation'
            x = torch.load(path, map_location='cpu', weights_only=True)
            base_path = Path(meta['derived_features']['source_dataset'])/rec['path']
            base = torch.load(base_path, map_location='cpu', weights_only=True)
            for key, value in base.items():
                actual = x[key][..., :value.shape[-1]] if key == 'tau_features' else x[key]
                assert torch.equal(actual, value), f'source row mismatch: {key}'
            assert len(x['labels']) == rec['events']
            assert np.all(x['labels'].numpy() == (1 if sample == 'H' else 0))
            assert np.all(x['tau_decay_mode'].numpy() == 4)
            qs.append(x['tau_features'].numpy()[:, :, qi].astype(float)*st['train_std']+st['train_mean'])
            raws.append(raw_masses(x, stats))
            ids.append(x['event_numbers'].numpy())
            offsets = x['track_offsets'].numpy()
            for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
                row_ids.append(f'{sample}:{rec["path"]}:{row}')
                fingerprint = hashlib.sha256()
                for arr in (x['event_features'][row], base['tau_features'][row], x['track_features'][start:stop]):
                    fingerprint.update(arr.numpy().tobytes())
                fingerprints.append(fingerprint.hexdigest())
            audit['input_files'].append({'path': str(path), 'sha256': digest(path)})
            audit['input_files'].append({'path': str(base_path), 'sha256': digest(base_path)})
        q, raw, eid = map(np.concatenate, (qs, raws, ids))
        assert np.isfinite(q).all() and np.isfinite(raw).all()
        assert len(set(row_ids)) == len(row_ids)
        assert len(set(fingerprints)) == len(fingerprints), 'repeated reconstructed content'
        residual = q-raw
        # Stored derived features are float32; tolerance is 2e-6 GeV,
        # generous relative to their quantization, not a physics threshold.
        np.testing.assert_allclose(q, raw, rtol=0, atol=2e-6)
        good_leg = (q >= 3*MPI) & (q <= MTAU)
        good = good_leg.all(axis=1)
        a = alpha(q[good])
        b = a.prod(axis=1)
        data[sample] = (q, good, a, b, residual)
        audit[sample] = dict(events=len(q), valid_pair_events=int(good.sum()),
            repeated_event_number_rows=int(len(eid)-len(np.unique(eid))),
            unique_reconstructed_content=len(set(fingerprints)), source_payload_exact_match=True,
            below_domain_legs=int((q < 3*MPI).sum()), above_domain_legs=int((q > MTAU).sum()),
            invalid_pair_events=int((~good).sum()), q_range=[float(q.min()), float(q.max())],
            q_quantiles=np.quantile(q, [0, .01, .25, .5, .75, .99, 1]).tolist(),
            max_mass_residual_gev=float(np.max(np.abs(residual))),
            negative_alpha_fraction_valid_legs=float((alpha(q[good_leg]) < 0).mean()),
            pair_moments=moments(b), bootstrap_95=bootstrap(b),
            mtau_shift_survival={str(shift): moments(alpha(q[good], MTAU+shift).prod(axis=1))['signed_survival'] for shift in (-.00009, .00009)})
        np.savez_compressed(args.output/f'{sample}_validation.npz', q=q, raw_q=raw,
                            event_numbers=eid, row_ids=np.asarray(row_ids), valid_pair=good)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'H': '#0072B2', 'Z': '#D55E00'}
    styles = {'H': '-', 'Z': '--'}
    zero = MTAU/np.sqrt(2)
    # Complete range is data-driven solely for display; no event selection.
    upper = max(q.max() for q, *_ in data.values())
    edges = np.arange(0, np.ceil(upper/.05)*.05+.025, .05)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for sample, (q, good, a, b, r) in data.items():
        ax[0].hist(q.ravel(), bins=edges, weights=np.ones(q.size)/q.size/.05,
                   histtype='step', color=colors[sample], linestyle=styles[sample], label=f'{sample}: {len(q)} events / {q.size} legs')
    ax[0].axvline(zero, color='black', ls=':', label=r'$m_\tau/\sqrt{2}$')
    ax[0].axvline(MTAU, color='grey', ls='-.', label=r'$m_\tau$')
    ax[0].set(xlabel=r'Reconstructed $Q_{3\pi}$ [GeV]', ylabel='Fraction of all selected legs / GeV (log)', xlim=(0, edges[-1]), yscale='log', ylim=(.004, 8))
    ax[0].legend(fontsize=9)
    grid = np.linspace(3*MPI, MTAU, 500)
    ax[1].plot(grid, alpha(grid), color='black')
    ax[1].axhline(0, color='grey', ls=':')
    ax[1].axvline(zero, color='grey', ls=':')
    ax[1].set(xlabel=r'$Q$ [GeV]', ylabel=r'Spin-1 reference $\alpha(Q)$', ylim=(-.4, 1))
    ax[1].text(.05, .98, r'$\alpha=(m_\tau^2-2Q^2)/(m_\tau^2+2Q^2)$'+'\n'+f'Zero: {zero:.5f} GeV', transform=ax[1].transAxes, va='top')
    fig.suptitle('Reco 3p0n × 3p0n validation | unit weights | direction-only reference')
    fig.tight_layout(); fig.savefig(args.output/'mass_reference.png', dpi=170); fig.savefig(args.output/'mass_reference.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    joint_edges = np.linspace(3*MPI, MTAU, 26)
    maxbin = max(np.histogram2d(q[g, 0], q[g, 1], bins=joint_edges)[0].max() for q, g, *_ in data.values())
    for ax, (sample, (q, g, *_)) in zip(axes, data.items()):
        counts, _, _ = np.histogram2d(q[g, 0], q[g, 1], bins=joint_edges)
        im = ax.pcolormesh(joint_edges, joint_edges, np.ma.masked_equal(counts.T, 0), norm=LogNorm(vmin=1, vmax=maxbin), cmap='cividis')
        ax.axvline(zero, color='red', ls='--'); ax.axhline(zero, color='red', ls='--')
        ax.set(xlabel=r'$Q_{3\pi}^{-}$ [GeV]', ylabel=r'$Q_{3\pi}^{+}$ [GeV]', title=f'{sample}: {g.sum()} pairs in reference domain')
        fig.colorbar(im, ax=ax, label='Events / 2D bin (log scale)')
    fig.suptitle('Mass pairing | dashed lines: coefficient sign changes | 25 × 25 bins')
    fig.tight_layout(); fig.savefig(args.output/'joint_mass.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bmax = np.ceil(max(np.abs(d[3]).max() for d in data.values())/.005)*.005
    bedges = np.linspace(-bmax, bmax, int(round(2*bmax/.005))+1)
    for sample, (q, g, a, b, r) in data.items():
        axes[0].hist(b, bins=bedges, weights=np.ones(len(b))/len(b)/.005, histtype='step', color=colors[sample], linestyle=styles[sample], label=sample)
        order = np.argsort(b)
        axes[1].plot(b[order], np.cumsum(b[order])/len(b), color=colors[sample], ls=styles[sample], label=sample)
    for ax in axes:
        ax.axvline(0, color='grey', ls=':'); ax.legend()
        ax.set_xlabel(r'$b=\alpha(Q^-)\alpha(Q^+)$ (reference)')
    axes[0].set(ylabel='Fraction of valid pairs / unit b', xlim=(-bmax, bmax))
    axes[1].axhline(0, color='grey', ls=':')
    axes[1].set(ylabel=r'Cumulative signed contribution $\sum_{b_i\leq b} b_i/N$', xlim=(-bmax, bmax))
    fig.suptitle('Signed arithmetic cancellation | both reco masses in reference domain')
    fig.tight_layout(); fig.savefig(args.output/'pair_cancellation.png', dpi=170); fig.savefig(args.output/'pair_cancellation.pdf'); plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k: audit[k] for k in ('H', 'Z')}, indent=2))


if __name__ == '__main__':
    main()
