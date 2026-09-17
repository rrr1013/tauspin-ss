"""Kinematic null space of tau-pair reconstruction, and the exact polarimeter
evaluated on it.

Everything here is truth level.  The visible daughters and the neutrinos come
from `build_truth_surface.py`; the polarimeter functional is the existing
vendored CLEO current wrapped by `HybridPolarimeter`, reused unmodified.

Why the exact on-shell polarimeter stays valid on the null space
---------------------------------------------------------------
A hypothesis here is a neutrino momentum that satisfies the same constraints the
truth neutrino satisfies: it is massless and it puts the tau exactly on shell.
So `HybridPolarimeter`, which is an off-shell extrapolation in general, is
evaluated only at physical tau masses in this study.  `fixed_contraction_mass`
is set to the upstream 1.7769 GeV so the mode-3 current matches the canonical
exact-h target bit for bit.

Conventions (identical to reco_h_supervision_20260906/build_targets.py)
----------------------------------------------------------------------
Four-vectors are (px,py,pz,E) in GeV.  Side 0 is tau-, side 1 is tau+.  The
polarimeter components are taken along n, r, k where k is the tau+ direction in
the tau-pair centre-of-mass frame, beam = (0,0,-1), n = unit(beam x k) and
r = unit(beam - (beam.k) k).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

UPSTREAM_TAU_MASS = 1.7769
DEFAULT_NN_ROOT = Path('/home/rbaba/tauspin-truth-neutrino-20260906/NN')

SURFACE_KEYS = ('pions', 'pion_charges', 'pion_counts', 'pi0', 'pi0_counts',
                'nu4', 'modes', 'labels', 'weights', 'h_ref',
                'global_indices', 'event_numbers', 'file_indices', 'entry_indices')


def load_polarimeter(nn_root: Path = DEFAULT_NN_ROOT):
    """Import the existing HybridPolarimeter without copying or editing it."""
    root = Path(nn_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.find_spec('neutrino_statistics_20260907.hybrid_polarimeter')
    if spec is None or spec.origin is None:
        raise RuntimeError(f'hybrid_polarimeter not importable from {root}')
    module = importlib.import_module('neutrino_statistics_20260907.hybrid_polarimeter')
    return module.HybridPolarimeter(), Path(spec.origin)


def load_surface(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        surface = {key: handle[key] for key in SURFACE_KEYS}
    surface['visible'] = surface['pions'].sum(axis=2) + surface['pi0']
    surface['tau'] = surface['visible'] + surface['nu4']
    return surface


def minkowski_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 3] * b[..., 3] - np.sum(a[..., :3] * b[..., :3], axis=-1)


def invariant_mass(p4: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(minkowski_dot(p4, p4), 0.0))


def boost(v: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Passive boost, same algebra as the vendored implementation."""
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bv = np.sum(v[..., :3] * beta, axis=-1)
    factor = gamma * gamma / (gamma + 1.0) * bv - gamma * v[..., 3]
    return np.concatenate((v[..., :3] + factor[..., None] * beta,
                           (gamma * (v[..., 3] - bv))[..., None]), axis=-1)


def pair_basis(tau: np.ndarray) -> np.ndarray:
    """(...,2,4) tau four-vectors -> (...,3,3) rows n, r, k in the pair CM."""
    pair = tau.sum(axis=-2)
    beta = pair[..., :3] / pair[..., 3, None]
    in_cm = boost(tau, beta[..., None, :])
    k = in_cm[..., 1, :3]
    k = k / np.linalg.norm(k, axis=-1)[..., None]
    beam = np.array([0.0, 0.0, -1.0])
    n = np.cross(np.broadcast_to(beam, k.shape), k)
    n = n / np.linalg.norm(n, axis=-1)[..., None]
    r = beam - np.sum(beam * k, axis=-1)[..., None] * k
    r = r / np.linalg.norm(r, axis=-1)[..., None]
    return np.stack((n, r, k), axis=-2)


def evaluate_h(polarimeter, surface: dict[str, np.ndarray], nu_xyz: np.ndarray,
               rows: np.ndarray | slice = slice(None)) -> dict[str, np.ndarray]:
    """Exact polarimeter for arbitrary neutrino momenta.

    `nu_xyz` has shape (...,2,3) and broadcasts against the selected rows on its
    leading axes; leading axes are hypothesis axes.  Returns h in the basis of
    the hypothesis itself (`h`), and the same polarimeter vectors re-expressed in
    the truth basis of the same event (`h_truth_basis`).
    """
    nu_xyz = np.asarray(nu_xyz, dtype=np.float64)
    if nu_xyz.shape[-2:] != (2, 3):
        raise ValueError('nu_xyz must end in (2,3)')
    lead = nu_xyz.shape[:-2]

    def spread(name: str, tail: tuple[int, ...]) -> np.ndarray:
        value = surface[name][rows]
        return np.broadcast_to(value.reshape((1,) * (len(lead) - 1) + value.shape), lead + tail)

    n_rows = len(surface['modes'][rows])
    if lead[-1] != n_rows:
        raise ValueError('last hypothesis axis must be the row axis')

    result = polarimeter(spread('pions', (2, 3, 4)), spread('pion_charges', (2, 3)),
                        spread('pion_counts', (2,)), spread('pi0', (2, 4)),
                        spread('pi0_counts', (2,)), spread('modes', (2,)), nu_xyz,
                        fixed_contraction_mass=UPSTREAM_TAU_MASS)

    visible = spread('visible', (2, 4))
    nu4 = np.concatenate((nu_xyz, np.linalg.norm(nu_xyz, axis=-1)[..., None]), axis=-1)
    hypothesis_basis = pair_basis(visible + nu4)
    truth_basis = pair_basis(surface['tau'][rows])
    # basis rows are orthonormal, so the transpose is the inverse.
    lab = np.einsum('...ij,...si->...sj', hypothesis_basis, result['h'])
    result['h_truth_basis'] = np.einsum('...ij,...sj->...si', truth_basis, lab)
    result['hypothesis_basis'] = hypothesis_basis
    return result


def azimuth_family(visible: np.ndarray, nu_xyz: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Rotate each neutrino about its own visible direction by `delta`.

    This is the exact one-parameter degeneracy at fixed tau energy: rotating the
    tau three-momentum about the visible direction preserves |p_tau|, E_tau and
    p_tau.p_vis, hence it preserves both m_tau and m_nu = 0 exactly.

    `visible` (...,2,4), `nu_xyz` (...,2,3), `delta` broadcastable to (...,2).
    Returns (...,2,3).
    """
    axis = visible[..., :3]
    axis = axis / np.linalg.norm(axis, axis=-1)[..., None]
    along = np.sum(nu_xyz * axis, axis=-1)[..., None] * axis
    perp = nu_xyz - along
    # Right-handed companion of perp in the plane orthogonal to axis.
    companion = np.cross(axis, perp)
    c = np.cos(delta)[..., None]
    s = np.sin(delta)[..., None]
    return along + c * perp + s * companion


def azimuth_of(visible: np.ndarray, nu_xyz: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Azimuth of the neutrino about the visible direction, measured from the
    plane spanned by the visible direction and `reference`.  (...,2) radians."""
    axis = visible[..., :3]
    axis = axis / np.linalg.norm(axis, axis=-1)[..., None]
    ref = reference - np.sum(reference * axis, axis=-1)[..., None] * axis
    ref = ref / np.linalg.norm(ref, axis=-1)[..., None]
    other = np.cross(axis, ref)
    perp = nu_xyz - np.sum(nu_xyz * axis, axis=-1)[..., None] * axis
    return np.arctan2(np.sum(perp * other, axis=-1), np.sum(perp * ref, axis=-1))


def solve_nu_z(visible: np.ndarray, nu_t: np.ndarray, tau_mass: np.ndarray) -> dict[str, np.ndarray]:
    """Solve the tau mass shell for the neutrino longitudinal momentum.

    With A = (m_tau^2 - m_vis^2)/2 + vx*nx + vy*ny and a = E_vis^2 - vz^2,
    the constraint E_vis*E_nu - vz*nz = A gives
        nz = (A*vz +- E_vis*sqrt(A^2 - a*|nT|^2)) / a
    and the branch is physical only when A + vz*nz >= 0 (E_nu >= 0).

    `visible` (...,4), `nu_t` (...,2), `tau_mass` (...).  Returns nz (...,2) for
    the two roots and a per-root validity mask.
    """
    vis_mass2 = minkowski_dot(visible, visible)
    energy = visible[..., 3]
    vz = visible[..., 2]
    a = energy * energy - vz * vz
    A = 0.5 * (tau_mass ** 2 - vis_mass2) + visible[..., 0] * nu_t[..., 0] + visible[..., 1] * nu_t[..., 1]
    nt2 = np.sum(nu_t * nu_t, axis=-1)
    discriminant = A * A - a * nt2
    root = np.sqrt(np.maximum(discriminant, 0.0))
    nz = np.stack(((A * vz + energy * root) / a, (A * vz - energy * root) / a), axis=-1)
    nu_energy = A[..., None] + vz[..., None] * nz
    valid = (discriminant[..., None] >= 0.0) & (nu_energy >= 0.0)
    return {'nz': nz, 'valid': valid, 'discriminant': discriminant, 'nu_energy': nu_energy}


def close_side_from_direction(visible: np.ndarray, direction: np.ndarray,
                              tau_mass: np.ndarray) -> dict[str, np.ndarray]:
    """Given an exact tau flight direction, solve for the tau momentum magnitude.

    With d the unit tau direction, P the unknown |p_tau| and the neutrino
    p_nu = P*d - p_vis, masslessness of the neutrino requires
        (E_tau - E_vis)^2 = |P*d - p_vis|^2 ,  E_tau = sqrt(P^2 + m_tau^2).
    Expanding, m_tau^2 - 2 E_tau E_vis + m_vis^2 + 2 P (d.p_vis) = 0, i.e.
        E_tau = (m_tau^2 + m_vis^2 + 2 P (d.p_vis)) / (2 E_vis) .
    Substituting E_tau^2 = P^2 + m_tau^2 gives a quadratic in P.
    """
    vis_mass2 = minkowski_dot(visible, visible)
    energy = visible[..., 3]
    c0 = tau_mass ** 2 + vis_mass2
    dv = np.sum(direction * visible[..., :3], axis=-1)
    # (c0 + 2 P dv)^2 = 4 E_vis^2 (P^2 + m_tau^2)
    a = 4.0 * (dv * dv - energy * energy)
    b = 4.0 * c0 * dv
    c = c0 * c0 - 4.0 * energy * energy * tau_mass ** 2
    discriminant = b * b - 4.0 * a * c
    root = np.sqrt(np.maximum(discriminant, 0.0))
    momentum = np.stack(((-b + root) / (2.0 * a), (-b - root) / (2.0 * a)), axis=-1)
    tau_energy = (c0[..., None] + 2.0 * momentum * dv[..., None]) / (2.0 * energy[..., None])
    nu = momentum[..., None] * direction[..., None, :] - visible[..., None, :3]
    nu_energy = tau_energy - energy[..., None]
    valid = ((discriminant[..., None] >= 0.0) & (momentum > 0.0) & (nu_energy >= 0.0)
             & np.isfinite(momentum))
    return {'momentum': momentum, 'nu': nu, 'nu_energy': nu_energy,
            'tau_energy': tau_energy, 'valid': valid, 'discriminant': discriminant}
