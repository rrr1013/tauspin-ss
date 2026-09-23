"""The MET-constrained kinematic null space of tau-pair reconstruction.

Given both visible four-momenta, both tau mass shells and the transverse
momentum sum of the two neutrinos, the neutrino system still has two free
parameters.  This module constructs that two-dimensional solution set exactly
and provides the two measures used on it.

Parametrisation
---------------
Take the transverse neutrino momentum of side 0 as the free coordinate,
`nu_t` in R^2.  Side 1 is then fixed transversally by `met - nu_t`, and each
side's longitudinal component follows from its own tau mass shell, which is a
quadratic with two roots.  The solution set is therefore four sheets over a
bounded region of the `nu_t` plane, joined where the two roots merge.

The allowed region
------------------
Writing `c = (m_tau^2 - m_vis^2)/2`, `a = E_vis^2 - vz^2` and `A = c + vT.nT`,
the mass shell has a real root iff `A^2 >= a |nT|^2`, and the neutrino energy is
non-negative iff `A >= 0`.  Together these are the single condition

    c + vT.nT >= sqrt(a) |nT| ,

which in polar form around the origin reads `t <= c / (sqrt(a) - vT.u)`.  Since
`sqrt(a) = sqrt(E_vis^2 - vz^2) >= |vT|` the denominator is positive, so the
region is the interior of an ellipse with one focus at the origin, eccentricity
`e = |vT| / sqrt(a)` and semi-latus rectum `l = c / sqrt(a)`.  Side 1 gives a
second such ellipse with its focus at `met`, and the allowed region is the
intersection of the two.

The two measures
----------------
`flat`
    Uniform in `nu_t` over the allowed region, sheets equally weighted.  A
    stated choice, not a derived posterior, but bounded and singularity free.

`solid`
    Uniform in each tau's rest-frame neutrino solid angle, the unpolarised
    two-body decay measure.  The invariant measure on one side's cone is

        d^3nu / (2 E_nu) * delta(2 p_vis.nu - (m_tau^2 - m_vis^2))
            = d^2nT / (4 |E_vis nu_z - vz E_nu|) ,

    and in the tau rest frame the same object is proportional to `dOmega*`
    with an event-dependent but hypothesis-independent constant.  Using the
    mass-shell solution, `E_vis nu_z - vz E_nu = +- sqrt(discriminant)`, so

        w_solid  =  1 / ( sqrt(disc_0) sqrt(disc_1) )

    up to a per-event constant that cancels in every normalised average.  The
    weight diverges integrably at the sheet-merging boundary, so the estimator
    reports its effective sample size and a clipped variant.

Conventions follow the rest of the project: four-vectors are (px,py,pz,E) in
GeV, side 0 is tau- and side 1 is tau+, and the polarimeter and the n,r,k basis
come from `analysis/azimuth_nullspace/nullspace.py` unmodified.
"""
from __future__ import annotations

import numpy as np

import nullspace as ns

UPSTREAM_TAU_MASS = ns.UPSTREAM_TAU_MASS


def region_ellipse(visible: np.ndarray, tau_mass: float = UPSTREAM_TAU_MASS) -> dict[str, np.ndarray]:
    """Ellipse of allowed transverse neutrino momenta for one side.

    `visible` (...,4).  Returns the polar description around the focus at the
    origin plus the axis-aligned bounding box, all with leading shape (...).
    """
    vis_mass2 = ns.minkowski_dot(visible, visible)
    energy = visible[..., 3]
    vz = visible[..., 2]
    vt = visible[..., :2]
    c = 0.5 * (tau_mass ** 2 - vis_mass2)
    a = energy * energy - vz * vz
    sqrt_a = np.sqrt(np.maximum(a, 0.0))
    vt_norm = np.linalg.norm(vt, axis=-1)
    eccentricity = np.where(sqrt_a > 0.0, vt_norm / np.maximum(sqrt_a, 1e-300), 0.0)
    latus = np.where(sqrt_a > 0.0, c / np.maximum(sqrt_a, 1e-300), 0.0)
    one_minus_e2 = np.maximum(1.0 - eccentricity ** 2, 1e-300)
    semi_major = latus / one_minus_e2
    semi_minor = latus / np.sqrt(one_minus_e2)
    unit = vt / np.maximum(vt_norm, 1e-300)[..., None]
    degenerate = vt_norm <= 1e-12
    unit = np.where(degenerate[..., None], np.array([1.0, 0.0]), unit)
    centre = (semi_major * eccentricity)[..., None] * unit
    ux, uy = unit[..., 0], unit[..., 1]
    half_x = np.sqrt((semi_major * ux) ** 2 + (semi_minor * uy) ** 2)
    half_y = np.sqrt((semi_major * uy) ** 2 + (semi_minor * ux) ** 2)
    return {
        'c': c, 'a': a, 'sqrt_a': sqrt_a, 'eccentricity': eccentricity,
        'semi_major': semi_major, 'semi_minor': semi_minor, 'centre': centre,
        'half': np.stack((half_x, half_y), axis=-1),
        'radius_max': np.where(eccentricity < 1.0, latus / np.maximum(1.0 - eccentricity, 1e-300),
                               np.inf),
        'area': np.pi * semi_major * semi_minor,
    }


def inside_region(visible: np.ndarray, nu_t: np.ndarray,
                  tau_mass: float = UPSTREAM_TAU_MASS) -> np.ndarray:
    """`c + vT.nT >= sqrt(a) |nT|`, the exact allowed-region test for one side."""
    vis_mass2 = ns.minkowski_dot(visible, visible)
    energy = visible[..., 3]
    vz = visible[..., 2]
    c = 0.5 * (tau_mass ** 2 - vis_mass2)
    sqrt_a = np.sqrt(np.maximum(energy * energy - vz * vz, 0.0))
    projection = visible[..., 0] * nu_t[..., 0] + visible[..., 1] * nu_t[..., 1]
    return (c + projection) >= sqrt_a * np.linalg.norm(nu_t, axis=-1)


def ellipse_frame(visible: np.ndarray, tau_mass: float = UPSTREAM_TAU_MASS) -> dict[str, np.ndarray]:
    """Affine frame that maps one side's allowed ellipse onto the unit disk.

    The region is a needle: `semi_minor = c / m_vis` is of order a GeV while
    `semi_major = c sqrt(a) / m_vis^2` grows with the visible transverse energy,
    so an axis-aligned box around it is almost empty and rejection sampling from
    a box is hopeless.  Working in this frame makes side 0 exactly uniform.
    """
    ellipse = region_ellipse(visible, tau_mass)
    vt = visible[..., :2]
    vt_norm = np.linalg.norm(vt, axis=-1)
    unit = vt / np.maximum(vt_norm, 1e-300)[..., None]
    unit = np.where((vt_norm <= 1e-12)[..., None], np.array([1.0, 0.0]), unit)
    perpendicular = np.stack((-unit[..., 1], unit[..., 0]), axis=-1)
    ellipse['axis_major'] = ellipse['semi_major'][..., None] * unit
    ellipse['axis_minor'] = ellipse['semi_minor'][..., None] * perpendicular
    return ellipse


def disk_to_plane(ellipse: dict[str, np.ndarray], disk: np.ndarray) -> np.ndarray:
    """Unit-disk coordinates -> transverse momenta inside that side's ellipse."""
    return (ellipse['centre'] + disk[..., 0, None] * ellipse['axis_major']
            + disk[..., 1, None] * ellipse['axis_minor'])


def plane_to_disk(ellipse: dict[str, np.ndarray], point: np.ndarray) -> np.ndarray:
    """Inverse of `disk_to_plane`; the axes are orthogonal so this is a scaling."""
    offset = point - ellipse['centre']
    major = ellipse['axis_major']
    minor = ellipse['axis_minor']
    x = np.sum(offset * major, axis=-1) / np.maximum(np.sum(major * major, axis=-1), 1e-300)
    y = np.sum(offset * minor, axis=-1) / np.maximum(np.sum(minor * minor, axis=-1), 1e-300)
    return np.stack((x, y), axis=-1)


def lattice(n_side: int, rows: int, rng: np.random.Generator) -> np.ndarray:
    """Jittered `n_side x n_side` lattice on the unit square, one per row.

    Stratification keeps the per-event Monte Carlo error well behaved at the few
    hundred points per event that the polarimeter budget allows.
    """
    grid = (np.arange(n_side) + 0.5) / n_side
    base = np.stack(np.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(-1, 2)
    jitter = (rng.random((rows, n_side * n_side, 2)) - 0.5) / n_side
    return np.clip(base[None] + jitter, 0.0, 1.0)


def unit_disk(unit: np.ndarray) -> np.ndarray:
    """Equal-area map of the unit square onto the unit disk."""
    radius = np.sqrt(unit[..., 0])
    angle = 2.0 * np.pi * unit[..., 1]
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def proposal(visible: np.ndarray, met: np.ndarray,
             tau_mass: float = UPSTREAM_TAU_MASS) -> dict[str, np.ndarray]:
    """A uniform proposal that contains the region both sides allow.

    Each side's region is a needle whose short direction is its minor axis, so
    the two minor-axis slabs already bound the intersection, and their common
    part is a parallelogram of area `4 w0 w1 / |sin theta|`.  That is tight when
    the needles cross at a large angle and useless when they are nearly
    parallel, so the wider ellipse-0 disk is kept as the alternative and the
    smaller of the two is used per event.  Both contain the intersection, so the
    choice changes only the efficiency, never the sampled distribution.
    """
    ellipse0 = ellipse_frame(visible[..., 0, :], tau_mass)
    ellipse1 = ellipse_frame(visible[..., 1, :], tau_mass)
    centre0 = ellipse0['centre']
    centre1 = met - ellipse1['centre']
    normal0 = ellipse0['axis_minor'] / np.maximum(
        ellipse0['semi_minor'], 1e-300)[..., None]
    normal1 = ellipse1['axis_minor'] / np.maximum(
        ellipse1['semi_minor'], 1e-300)[..., None]
    cross = normal0[..., 0] * normal1[..., 1] - normal0[..., 1] * normal1[..., 0]
    safe = np.abs(cross) > 1e-6
    # Solve [n0; n1] x = [s0 + n0.c0; s1 + n1.c1] for the two basis directions.
    inverse = np.stack((np.stack((normal1[..., 1], -normal0[..., 1]), axis=-1),
                        np.stack((-normal1[..., 0], normal0[..., 0]), axis=-1)), axis=-2)
    inverse = inverse / np.where(safe, cross, 1.0)[..., None, None]
    offset = np.stack((np.sum(normal0 * centre0, axis=-1),
                       np.sum(normal1 * centre1, axis=-1)), axis=-1)
    half = np.stack((ellipse0['semi_minor'], ellipse1['semi_minor']), axis=-1)
    parallelogram_area = 4.0 * half[..., 0] * half[..., 1] / np.maximum(np.abs(cross), 1e-300)
    use_parallelogram = safe & (parallelogram_area < ellipse0['area'])
    return {
        'ellipse0': ellipse0, 'ellipse1': ellipse1,
        'inverse': inverse, 'offset': offset, 'half': half,
        'use_parallelogram': use_parallelogram,
        'parallelogram_area': parallelogram_area,
        'area': np.where(use_parallelogram, parallelogram_area, ellipse0['area']),
    }


def draw_proposal(spec: dict[str, np.ndarray], unit: np.ndarray) -> np.ndarray:
    """Unit-square draws -> transverse momenta, uniform on the proposal region."""
    slab = spec['offset'][None] + (2.0 * unit - 1.0) * spec['half'][None]
    parallelogram = np.einsum('rij,srj->sri', spec['inverse'], slab)
    disk = disk_to_plane(spec['ellipse0'], unit_disk(unit))
    return np.where(spec['use_parallelogram'][None, :, None], parallelogram, disk)


def sample_region(visible: np.ndarray, met: np.ndarray, unit: np.ndarray,
                  tau_mass: float = UPSTREAM_TAU_MASS) -> dict[str, np.ndarray]:
    """Draw from the proposal and test both sides exactly.

    `unit` (S,R,2) in [0,1]^2.  Accepted points are uniform on the region both
    sides allow, because the proposal is uniform on a superset of it.
    """
    spec = proposal(visible, met, tau_mass)
    nu_t = draw_proposal(spec, unit)
    accepted = inside_region(visible[None, ..., 0, :], nu_t, tau_mass)
    accepted &= inside_region(visible[None, ..., 1, :], met[None] - nu_t, tau_mass)
    return {'nu_t': nu_t, 'accepted': accepted, 'spec': spec}


def compact(accepted: np.ndarray, keep: int, rng: np.random.Generator) -> np.ndarray:
    """Pick up to `keep` accepted draws per event, as evenly spread as possible.

    Accepted draws are already uniform on the region, so any subset of them is
    too; systematic selection over the accepted list keeps the retained sample
    spread across the proposal instead of clustered at its start.  Returns
    indices into the draw axis with `-1` padding.
    """
    n_draws, rows = accepted.shape
    order = np.argsort(~accepted, axis=0, kind='stable')     # accepted first
    available = accepted.sum(axis=0)
    take = np.minimum(available, keep)
    positions = np.arange(keep)[:, None]
    stride = np.where(take > 0, available / np.maximum(take, 1), 1.0)
    offset = rng.random(rows)
    picked = np.floor((positions + offset[None]) * stride[None]).astype(np.int64)
    picked = np.minimum(picked, np.maximum(available - 1, 0)[None])
    chosen = np.take_along_axis(order, picked, axis=0)
    return np.where(positions < take[None], chosen, -1)


def solution_sheets(visible: np.ndarray, met: np.ndarray, nu_t: np.ndarray,
                    tau_mass: float = UPSTREAM_TAU_MASS) -> dict[str, np.ndarray]:
    """All four sheets above a set of side-0 transverse momenta.

    `visible` (R,2,4), `met` (R,2), `nu_t` (S,R,2).  Returns neutrino momenta
    with shape (S,4,R,2,3): the second axis is the sheet, ordered as
    (root0 of side 0, root0 of side 1), (root0, root1), (root1, root0),
    (root1, root1).
    """
    nu_t_pair = np.stack((nu_t, met[None] - nu_t), axis=-2)          # (S,R,2,2)
    solved = ns.solve_nu_z(visible[None], nu_t_pair, np.asarray(tau_mass, dtype=np.float64))
    nz = solved['nz']                                                # (S,R,2,2roots)
    valid = solved['valid']                                          # (S,R,2,2roots)
    disc = solved['discriminant']                                    # (S,R,2)

    pick = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    sheets = []
    sheet_valid = []
    for root0, root1 in pick:
        chosen = np.stack((nz[..., 0, root0], nz[..., 1, root1]), axis=-1)   # (S,R,2)
        sheets.append(np.concatenate((nu_t_pair, chosen[..., None]), axis=-1))
        sheet_valid.append(valid[..., 0, root0] & valid[..., 1, root1])
    return {
        'nu': np.stack(sheets, axis=1),                              # (S,4,R,2,3)
        'valid': np.stack(sheet_valid, axis=1),                      # (S,4,R)
        'discriminant': disc,                                        # (S,R,2)
        'nu_t_pair': nu_t_pair,
    }


def solid_angle_weight(discriminant: np.ndarray) -> np.ndarray:
    """`1 / (sqrt(disc_0) sqrt(disc_1))`, the unpolarised two-body decay measure.

    The per-event constant that relates this to `dOmega*` is hypothesis
    independent and cancels in every normalised average, so it is dropped.
    """
    root = np.sqrt(np.maximum(discriminant, 0.0))
    return 1.0 / np.maximum(root[..., 0] * root[..., 1], 1e-300)


def to_reference_basis(h: np.ndarray, hypothesis_basis: np.ndarray,
                       reference_basis: np.ndarray) -> np.ndarray:
    """Re-express polarimeter vectors from each hypothesis' own n,r,k frame in a
    single per-event reference frame.  Basis rows are orthonormal, so the
    transpose is the inverse."""
    lab = np.einsum('...ij,...si->...sj', hypothesis_basis, h)
    return np.einsum('...ij,...sj->...si', reference_basis, lab)


def transverse_statistic(h: np.ndarray) -> np.ndarray:
    """T = h-_n h+_n + h-_r h+_r - h-_k h+_k, identical to the existing definition."""
    return (h[..., 0, 0] * h[..., 1, 0] + h[..., 0, 1] * h[..., 1, 1]
            - h[..., 0, 2] * h[..., 1, 2])


SPIN_SIGNATURE = np.array([1.0, 1.0, -1.0])

# Theory values, fixed before looking at any result.  The unpolarised joint decay
# density carries the factor 1 + sum_ij C_ij h-_i h+_j, so a likelihood ratio on
# the null space needs only the second moment matrix.
C_HIGGS = np.diag([1.0, 1.0, -1.0])
C_Z = np.diag([0.0, 0.0, 1.0])


def moment_estimators(m_minus: np.ndarray, m_plus: np.ndarray,
                      second: np.ndarray) -> dict[str, np.ndarray]:
    """Every estimator of this run from the per-event first and second moments.

    `m_minus`,`m_plus` are (R,3) measure averages of `h` per side, `second` is
    (R,3,3) with `second[r,i,j] = E[h-_i h+_j]`.
    """
    joint = np.einsum('i,rii->r', SPIN_SIGNATURE, second)
    marginal = np.einsum('i,ri,ri->r', SPIN_SIGNATURE, m_minus, m_plus)
    weight_h = 1.0 + np.einsum('ij,rij->r', C_HIGGS, second)
    weight_z = 1.0 + np.einsum('ij,rij->r', C_Z, second)
    return {
        'joint': joint,
        'marginal': marginal,
        'covariance_diagonal': np.stack(
            [second[:, i, i] - m_minus[:, i] * m_plus[:, i] for i in range(3)], axis=-1),
        'likelihood_ratio': np.log(np.maximum(weight_h, 1e-12))
                            - np.log(np.maximum(weight_z, 1e-12)),
        'weight_higgs': weight_h,
        'weight_z': weight_z,
    }
