"""Analytical theory and formulas for correlated systematic errors in Z -> tau tau calibration.

Decomposition under transverse basis rotations psi around the ditau axis k:
  z- = h-_n + i h-_r,  z+ = h+_n + i h+_r
  m = 0 (Higgs):
    T + i A = conj(z-) z+ = |z-||z+| exp(i (phi+ - phi-))
    T = h-_n h+_n + h-_r h+_r
    A = h-_n h+_r - h-_r h+_n
  m = 2 (Z):
    Q1 + i Q2 = z- z+ = |z-||z+| exp(i (phi- + phi+))
    Q1 = h-_n h+_n - h-_r h+_r
    Q2 = h-_n h+_r + h-_r h+_n

Properties:
1. Global frame rotation delta_psi (same for both taus):
   phi- -> phi- - delta_psi,  phi+ -> phi+ - delta_psi
   -> phi+ - phi- is strictly invariant (m = 0 immune to all orders)
   -> phi- + phi+ -> phi- + phi+ - 2 delta_psi (m = 2 rotates at 2x frequency)
   Under Gaussian frame jitter sigma_psi:
     <T> = T_0
     <Q1> = Q1_0 * exp(-2 sigma_psi^2)
     <Q2> = Q2_0 * exp(-2 sigma_psi^2)
     rho_T = 1.0
     rho_Q = exp(-2 sigma_psi^2)
     R_{T/Q} = rho_T / rho_Q = exp(+2 sigma_psi^2) >= 1

2. Shared Primary Vertex displacement delta_x_PV (transverse):
   For back-to-back taus (t2 ~= -t1 in transverse plane):
   A shared PV shift delta_x_PV in the transverse plane shifts both d0 vectors:
     delta_d0,1 = -delta_x_PV,perp1
     delta_d0,2 = -delta_x_PV,perp2
   In the local transverse coordinates of tau 1 and tau 2, this origin shift tilts
   the polarimeter angles in OPPOSITE directions:
     delta_phi1 ~= -delta_phi,  delta_phi2 ~= +delta_phi
   -> phi+ - phi- -> (phi+ - phi-) + 2 delta_phi  (m = 0 shifts by 2 delta_phi)
   -> phi- + phi+ -> (phi- + phi+) + 0            (m = 2 cancels at first order!)
   Under Gaussian PV jitter sigma_PV:
     rho_T ~= exp(-2 sigma_{delta_phi}^2) <= 1
     rho_Q ~= 1.0 (first-order protected)
     R_{T/Q} = rho_T / rho_Q ~= exp(-2 sigma_{delta_phi}^2) <= 1

These two effects are exact mathematical duals:
  - Frame rotation dampens m = 2 and leaves m = 0 invariant (R_{T/Q} > 1).
  - Shared PV jitter dampens m = 0 and leaves m = 2 invariant (R_{T/Q} < 1).
"""
import numpy as np


def rotate_transverse(h, psi):
    """Rotate polarimeter 3-vectors (..., 2, 3) by angle psi around k axis.
    
    Convention: local basis is (n, r, k). Rotation in (n, r) plane by psi.
    """
    c, s = np.cos(psi), np.sin(psi)
    while c.ndim > 0 and c.ndim < h.ndim - 1:
        c = np.expand_dims(c, -1)
        s = np.expand_dims(s, -1)
    out = np.empty_like(h)
    out[..., 0] = c * h[..., 0] + s * h[..., 1]
    out[..., 1] = -s * h[..., 0] + c * h[..., 1]
    out[..., 2] = h[..., 2]
    return out


def comps(h):
    """Compute (T, A, Q1, Q2, kk) from (..., 2, 3) polarimeters."""
    a, b = h[..., 0, :], h[..., 1, :]
    T = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    A = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    Q1 = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    Q2 = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    kk = a[..., 2] * b[..., 2]
    return np.stack([T, A, Q1, Q2, kk], axis=-1)


def theoretical_frame_transfer_ratio(sigma_psi):
    """Analytic double ratio R_{T/Q} as a function of Gaussian frame jitter sigma_psi (rad)."""
    return np.exp(2.0 * sigma_psi**2)


def theoretical_pv_transfer_ratio(sigma_pv, mean_d0):
    """Analytic double ratio R_{T/Q} for back-to-back 1-prong taus as a function of sigma_pv and d0."""
    sigma_phi = sigma_pv / np.maximum(mean_d0, 1e-12)
    return np.exp(-2.0 * sigma_phi**2)


def test_analytical_dualism():
    """Verify analytical invariance of T under global rotation and Q under anti-symmetric rotation."""
    rng = np.random.default_rng(42)
    h = rng.normal(size=(1000, 2, 3))
    # normalise transverse magnitude
    h /= np.linalg.norm(h, axis=-1, keepdims=True)
    
    c0 = comps(h)
    
    # Global frame rotation psi = 0.3 rad
    psi = 0.3
    h_rot = rotate_transverse(h, psi)
    c_rot = comps(h_rot)
    
    # Test T, A exact invariance
    np.testing.assert_allclose(c_rot[..., 0], c0[..., 0], atol=1e-14)
    np.testing.assert_allclose(c_rot[..., 1], c0[..., 1], atol=1e-14)
    
    # Test Q1, Q2 exact transformation: Q1' = cos(2psi) Q1 + sin(2psi) Q2
    q1_exp = np.cos(2*psi) * c0[..., 2] + np.sin(2*psi) * c0[..., 3]
    q2_exp = -np.sin(2*psi) * c0[..., 2] + np.cos(2*psi) * c0[..., 3]
    np.testing.assert_allclose(c_rot[..., 2], q1_exp, atol=1e-14)
    np.testing.assert_allclose(c_rot[..., 3], q2_exp, atol=1e-14)
    
    # Anti-symmetric rotation (e.g. shared PV shift): psi1 = +phi, psi2 = -phi
    phi = 0.2
    c_p, s_p = np.cos(phi), np.sin(phi)
    h_anti = np.empty_like(h)
    # side 0 rotated by +phi
    h_anti[..., 0, 0] = c_p * h[..., 0, 0] + s_p * h[..., 0, 1]
    h_anti[..., 0, 1] = -s_p * h[..., 0, 0] + c_p * h[..., 0, 1]
    h_anti[..., 0, 2] = h[..., 0, 2]
    # side 1 rotated by -phi
    h_anti[..., 1, 0] = c_p * h[..., 1, 0] - s_p * h[..., 1, 1]
    h_anti[..., 1, 1] = s_p * h[..., 1, 0] + c_p * h[..., 1, 1]
    h_anti[..., 1, 2] = h[..., 1, 2]
    
    c_anti = comps(h_anti)
    # Under anti-symmetric rotation, phi1+phi2 is invariant -> Q1 is invariant!
    np.testing.assert_allclose(c_anti[..., 2], c0[..., 2], atol=1e-14)
    np.testing.assert_allclose(c_anti[..., 3], c0[..., 3], atol=1e-14)
    # while T transforms as cos(2phi) T - sin(2phi) A
    t_exp = np.cos(2*phi) * c0[..., 0] - np.sin(2*phi) * c0[..., 1]
    np.testing.assert_allclose(c_anti[..., 0], t_exp, atol=1e-14)


if __name__ == '__main__':
    test_analytical_dualism()
    print("Analytical dualism test passed successfully!")
