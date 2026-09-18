"""Kinematic closure of the MET-constrained null space.

These tests need neither ROOT nor the cohort: they build exact two-body tau
decays from random numbers and check that the constructed solution set contains
the configuration it was built from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metspace as ms
import nullspace as ns

TAU = ms.UPSTREAM_TAU_MASS


def make_events(rows: int, seed: int = 7) -> dict[str, np.ndarray]:
    """Random tau pairs, each tau decaying to one massive visible system."""
    rng = np.random.default_rng(seed)
    m_vis = rng.uniform(0.14, 1.4, size=(rows, 2))
    momentum = rng.uniform(10.0, 400.0, size=(rows, 2))
    cos_theta = rng.uniform(-0.9, 0.9, size=(rows, 2))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(rows, 2))
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    direction = np.stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), axis=-1)
    tau = np.concatenate((momentum[..., None] * direction,
                          np.sqrt(momentum ** 2 + TAU ** 2)[..., None]), axis=-1)

    # Isotropic neutrino in the tau rest frame.
    e_star = (TAU ** 2 - m_vis ** 2) / (2.0 * TAU)
    cos_star = rng.uniform(-1.0, 1.0, size=(rows, 2))
    phi_star = rng.uniform(0.0, 2.0 * np.pi, size=(rows, 2))
    sin_star = np.sqrt(1.0 - cos_star ** 2)
    nu_star = e_star[..., None] * np.stack(
        (sin_star * np.cos(phi_star), sin_star * np.sin(phi_star), cos_star), axis=-1)
    nu_star4 = np.concatenate((nu_star, e_star[..., None]), axis=-1)
    beta = -tau[..., :3] / tau[..., 3, None]          # boost from tau rest frame to lab
    nu = ns.boost(nu_star4, beta)
    visible = tau - nu
    met = nu[:, 0, :2] + nu[:, 1, :2]
    return {'tau': tau, 'nu': nu, 'visible': visible, 'met': met, 'm_vis': m_vis}


def test_truth_is_inside_the_region():
    events = make_events(2000)
    for side in (0, 1):
        assert ms.inside_region(events['visible'][:, side], events['nu'][:, side, :2], TAU).all()


def test_sheets_recover_the_truth_neutrinos():
    events = make_events(2000)
    nu_t = events['nu'][:, 0, :2][None]                # (1,R,2)
    sheets = ms.solution_sheets(events['visible'], events['met'], nu_t, TAU)
    residual = np.abs(sheets['nu'][0] - events['nu'][None, :, :, :3]).max(axis=(-1, -2))
    best = residual.min(axis=0)
    assert np.isfinite(best).all()
    assert best.max() < 1e-6, best.max()
    assert sheets['valid'].any(axis=1).all()


def test_sheets_satisfy_the_constraints():
    events = make_events(500)
    rng = np.random.default_rng(3)
    unit = ms.lattice(8, len(events['met']), rng).transpose(1, 0, 2)
    drawn = ms.sample_region(events['visible'], events['met'], unit, TAU)
    nu_t = drawn['nu_t']
    sheets = ms.solution_sheets(events['visible'], events['met'], nu_t, TAU)
    nu = sheets['nu']
    ok = sheets['valid']
    nu4 = np.concatenate((nu, np.linalg.norm(nu, axis=-1)[..., None]), axis=-1)
    tau = events['visible'][None, None] + nu4
    mass = ns.invariant_mass(tau)
    assert np.abs(mass[ok] - TAU).max() < 1e-6, np.abs(mass[ok] - TAU).max()
    met = nu[..., 0, :2] + nu[..., 1, :2]
    target = np.broadcast_to(events['met'][None, None], met.shape)
    assert np.abs(met[ok] - target[ok]).max() < 1e-9


def test_solid_angle_weight_matches_the_jacobian():
    """E_vis nu_z - vz E_nu must equal +- sqrt(discriminant) on every sheet."""
    events = make_events(400)
    rng = np.random.default_rng(11)
    unit = ms.lattice(6, len(events['met']), rng).transpose(1, 0, 2)
    nu_t = ms.sample_region(events['visible'], events['met'], unit, TAU)['nu_t']
    sheets = ms.solution_sheets(events['visible'], events['met'], nu_t, TAU)
    nu = sheets['nu']
    ok = sheets['valid']
    energy = np.linalg.norm(nu, axis=-1)
    vis = events['visible'][None, None]
    jacobian = vis[..., 3] * nu[..., 2] - vis[..., 2] * energy
    root = np.broadcast_to(np.sqrt(np.maximum(sheets['discriminant'], 0.0))[:, None],
                           jacobian.shape)
    keep = np.broadcast_to(ok[..., None], jacobian.shape)
    diff = np.abs(np.abs(jacobian) - root)[keep]
    scale = np.maximum(root[keep], 1.0)
    assert (diff / scale).max() < 1e-7, (diff / scale).max()


def test_proposal_contains_the_truth_and_accepts_often():
    """The proposal must contain every point the region contains, and be tight
    enough that a few hundred draws per event leave a usable sample."""
    events = make_events(3000)
    rng = np.random.default_rng(5)
    spec = ms.proposal(events['visible'], events['met'], TAU)
    truth = events['nu'][:, 0, :2]
    slab0 = np.abs(np.sum((truth - spec['ellipse0']['centre'])
                          * spec['ellipse0']['axis_minor'], axis=-1)
                   / spec['ellipse0']['semi_minor'])
    slab1 = np.abs(np.sum((truth - (events['met'] - spec['ellipse1']['centre']))
                          * spec['ellipse1']['axis_minor'], axis=-1)
                   / spec['ellipse1']['semi_minor'])
    assert slab0.max() <= spec['ellipse0']['semi_minor'].max() * (1.0 + 1e-9)
    assert (slab0 <= spec['ellipse0']['semi_minor'] * (1.0 + 1e-9)).all()
    assert (slab1 <= spec['ellipse1']['semi_minor'] * (1.0 + 1e-9)).all()

    unit = ms.lattice(16, len(events['met']), rng).transpose(1, 0, 2)
    drawn = ms.sample_region(events['visible'], events['met'], unit, TAU)
    acceptance = drawn['accepted'].mean(axis=0)
    assert acceptance.mean() > 0.25, acceptance.mean()
    print(f'    acceptance mean {acceptance.mean():.3f}, median {np.median(acceptance):.3f}, '
          f'5% {np.quantile(acceptance, 0.05):.3f}')


def test_draws_are_uniform_on_the_region():
    """Acceptance times proposal area must reproduce the region area measured on
    a fine independent grid."""
    events = make_events(40, seed=19)
    rng = np.random.default_rng(23)
    spec = ms.proposal(events['visible'], events['met'], TAU)
    unit = ms.lattice(90, len(events['met']), rng).transpose(1, 0, 2)
    drawn = ms.sample_region(events['visible'], events['met'], unit, TAU)
    estimate = drawn['accepted'].mean(axis=0) * spec['area']

    # Independent estimate: brute-force grid over the ellipse-0 disk frame.
    grid = ms.lattice(220, len(events['met']), np.random.default_rng(29)).transpose(1, 0, 2)
    disk = 2.0 * grid - 1.0
    nu_t = ms.disk_to_plane(spec['ellipse0'], disk)
    ok = (np.sum(disk * disk, axis=-1) <= 1.0)
    ok &= ms.inside_region(events['visible'][None, :, 0], nu_t, TAU)
    ok &= ms.inside_region(events['visible'][None, :, 1], events['met'][None] - nu_t, TAU)
    reference = ok.mean(axis=0) * 4.0 * spec['ellipse0']['semi_major'] * spec['ellipse0']['semi_minor']
    ratio = estimate / np.maximum(reference, 1e-300)
    assert np.abs(np.median(ratio) - 1.0) < 0.05, np.median(ratio)
    print(f'    area ratio median {np.median(ratio):.4f}, '
          f'5-95% {np.quantile(ratio, 0.05):.3f}-{np.quantile(ratio, 0.95):.3f}')


def test_compaction_keeps_only_accepted_draws():
    accepted = np.zeros((500, 40), bool)
    rng = np.random.default_rng(31)
    for column in range(40):
        picks = rng.choice(500, size=rng.integers(0, 400), replace=False)
        accepted[picks, column] = True
    chosen = ms.compact(accepted, 64, rng)
    available = accepted.sum(axis=0)
    for column in range(40):
        taken = chosen[:, column]
        real = taken[taken >= 0]
        assert len(real) == min(available[column], 64)
        assert len(np.unique(real)) == len(real)
        assert accepted[real, column].all()


def test_region_radius_matches_the_polar_form():
    events = make_events(300)
    visible = events['visible'][:, 0]
    ellipse = ms.region_ellipse(visible, TAU)
    angles = np.linspace(0.0, 2.0 * np.pi, 37)[:, None]
    unit = np.stack((np.cos(angles) * np.ones_like(ellipse['c']),
                     np.sin(angles) * np.ones_like(ellipse['c'])), axis=-1)
    projection = np.einsum('ari,ri->ar', unit, visible[..., :2])
    radius = ellipse['c'] / (ellipse['sqrt_a'] - projection / np.linalg.norm(unit, axis=-1))
    assert (radius > 0).all()
    inside = ms.inside_region(visible[None], unit * (radius * 0.999)[..., None], TAU)
    outside = ms.inside_region(visible[None], unit * (radius * 1.001)[..., None], TAU)
    assert inside.all()
    assert not outside.any()
    offset = unit * radius[..., None] - ellipse['centre'][None]
    assert (np.abs(offset) <= ellipse['half'][None] + 1e-6).all()


if __name__ == '__main__':
    for name, function in sorted(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'{name}: ok')
