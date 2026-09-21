import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_TAU = 1.77686
M_Z = 91.1876

def generate_events(n):
    # Z rest frame
    phi = np.random.uniform(0, 2*np.pi, n)
    costh = np.random.uniform(-1, 1, n)
    sinth = np.sqrt(1 - costh**2)
    p_tau = np.sqrt((M_Z/2)**2 - M_TAU**2)
    
    # tau1
    p1x = p_tau * sinth * np.cos(phi)
    p1y = p_tau * sinth * np.sin(phi)
    p1z = p_tau * costh
    E1 = np.full(n, M_Z/2)
    tau1 = np.column_stack([E1, p1x, p1y, p1z])
    
    # tau2
    tau2 = np.column_stack([E1, -p1x, -p1y, -p1z])
    
    # boost Z slightly (e.g. pT = 30 GeV along x)
    beta = 30.0 / np.sqrt(M_Z**2 + 30.0**2)
    gamma = 1.0 / np.sqrt(1 - beta**2)
    def boost_x(p):
        E = gamma * p[:, 0] + gamma * beta * p[:, 1]
        px = gamma * beta * p[:, 0] + gamma * p[:, 1]
        return np.column_stack([E, px, p[:, 2], p[:, 3]])
        
    tau1 = boost_x(tau1)
    tau2 = boost_x(tau2)
    
    return tau1, tau2

def decay_tau_pi_nu(tau_p4):
    M_PI = 0.13957
    n = len(tau_p4)
    # decay in tau rest frame
    phi = np.random.uniform(0, 2*np.pi, n)
    costh = np.random.uniform(-1, 1, n)
    sinth = np.sqrt(1 - costh**2)
    p_pi = (M_TAU**2 - M_PI**2) / (2 * M_TAU)
    
    pi_rest = np.column_stack([
        np.full(n, np.sqrt(M_PI**2 + p_pi**2)),
        p_pi * sinth * np.cos(phi),
        p_pi * sinth * np.sin(phi),
        p_pi * costh
    ])
    nu_rest = np.column_stack([
        p_pi * np.ones(n),
        -pi_rest[:, 1],
        -pi_rest[:, 2],
        -pi_rest[:, 3]
    ])
    
    # boost to lab frame
    tau_p = np.linalg.norm(tau_p4[:, 1:], axis=1)
    beta_vec = tau_p4[:, 1:] / tau_p4[:, 0:1]
    beta2 = np.sum(beta_vec**2, axis=1)
    gamma = 1.0 / np.sqrt(1 - beta2)
    
    def boost(p4_rest):
        bp = np.sum(p4_rest[:, 1:] * beta_vec, axis=1)
        gamma2 = (gamma - 1.0) / beta2
        E_lab = gamma * (p4_rest[:, 0] + bp)
        p_lab = p4_rest[:, 1:] + (gamma2 * bp + gamma * p4_rest[:, 0])[:, None] * beta_vec
        return np.column_stack([E_lab, p_lab])
        
    pi_lab = boost(pi_rest)
    nu_lab = boost(nu_rest)
    return pi_lab, nu_lab

def smear_direction(vec, sigma_theta):
    n = len(vec)
    v_norm = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    # create arbitrary orthogonal vector
    tmp = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n)])
    mask = np.abs(v_norm[:, 0]) > 0.9
    tmp[mask] = [0, 1, 0]
    
    u1 = np.cross(v_norm, tmp)
    u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
    u2 = np.cross(v_norm, u1)
    
    dtheta = np.random.normal(0, sigma_theta, n)
    dphi = np.random.uniform(0, 2*np.pi, n)
    
    smeared = (v_norm * np.cos(dtheta)[:, None] + 
               u1 * (np.sin(dtheta) * np.cos(dphi))[:, None] + 
               u2 * (np.sin(dtheta) * np.sin(dphi))[:, None])
    return smeared

def solve_nu_E(pi_p4, nu_dir):
    # E_nu = (M_TAU**2 - M_PI**2) / 2(E_vis - p_vis * cos_theta)
    M_PI = 0.13957
    p_vis_mag = np.linalg.norm(pi_p4[:, 1:], axis=1)
    p_vis_dir = pi_p4[:, 1:] / p_vis_mag[:, None]
    cos_theta = np.sum(p_vis_dir * nu_dir, axis=1)
    
    # filter out non-physical
    denom = pi_p4[:, 0] - p_vis_mag * cos_theta
    mask = denom > 1e-5
    
    E_nu = np.zeros(len(pi_p4))
    E_nu[mask] = (M_TAU**2 - M_PI**2) / (2 * denom[mask])
    return E_nu, mask

np.random.seed(42)
n_events = 50000
tau1, tau2 = generate_events(n_events)
pi1, nu1 = decay_tau_pi_nu(tau1)
pi2, nu2 = decay_tau_pi_nu(tau2)

angles = np.logspace(-4, -0.5, 20) # from 0.1 mrad to ~300 mrad
mass_res = []
spin_cos = []

# Polarimeter for pi mode is just p_pi in tau rest frame (ignoring some constant factors)
# Wait, polarimeter h is derived from pi direction. Let's just use exact tau for now,
# or simply measure the correlation of reconstructed tau direction with true tau direction.
# Actually, H/Z spin is sensitive to tau direction. Let's measure tau angle resolution.
# And mass resolution.

for ang in angles:
    nu1_dir_true = nu1[:, 1:] / np.linalg.norm(nu1[:, 1:], axis=1, keepdims=True)
    nu2_dir_true = nu2[:, 1:] / np.linalg.norm(nu2[:, 1:], axis=1, keepdims=True)
    
    nu1_dir_smeared = smear_direction(nu1_dir_true, ang)
    nu2_dir_smeared = smear_direction(nu2_dir_true, ang)
    
    E1_reco, m1 = solve_nu_E(pi1, nu1_dir_smeared)
    E2_reco, m2 = solve_nu_E(pi2, nu2_dir_smeared)
    
    mask = m1 & m2
    
    nu1_reco = np.column_stack([E1_reco[mask], nu1_dir_smeared[mask] * E1_reco[mask, None]])
    nu2_reco = np.column_stack([E2_reco[mask], nu2_dir_smeared[mask] * E2_reco[mask, None]])
    
    tau1_reco = pi1[mask] + nu1_reco
    tau2_reco = pi2[mask] + nu2_reco
    
    Z_reco = tau1_reco + tau2_reco
    Z_mass_reco = np.sqrt(np.maximum(0, Z_reco[:, 0]**2 - np.sum(Z_reco[:, 1:]**2, axis=1)))
    
    # RMS mass resolution
    mass_diff = Z_mass_reco - M_Z
    # use IQR to robustly estimate RMS due to tails
    q75, q25 = np.percentile(mass_diff, [75, 25])
    rms = (q75 - q25) / 1.349
    mass_res.append(rms)
    
    # Tau direction resolution
    tau1_dir_true = tau1[mask, 1:] / np.linalg.norm(tau1[mask, 1:], axis=1, keepdims=True)
    tau1_dir_reco = tau1_reco[:, 1:] / np.linalg.norm(tau1_reco[:, 1:], axis=1, keepdims=True)
    
    cos_tau = np.sum(tau1_dir_true * tau1_dir_reco, axis=1)
    spin_cos.append(np.mean(cos_tau))

fig, ax1 = plt.subplots(figsize=(8, 6))

color = 'tab:red'
ax1.set_xlabel('Neutrino Direction Smearing $\sigma_{\\theta}$ [rad]')
ax1.set_ylabel('$m_{\\tau\\tau}$ Resolution (IQR/1.349) [GeV]', color=color)
ax1.plot(angles, mass_res, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Mean $\\cos(\\hat{p}_{\\tau}^{true}, \\hat{p}_{\\tau}^{reco})$ (proxy for polarimeter axis)', color=color)
ax2.plot(angles, spin_cos, color=color, marker='s')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Requirement Curve: Neutrino Direction Precision vs Physics Observables')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig('/Users/ryunosuke/Projects/tauspin/analysis/neutrino_direction_requirement.png', dpi=150)
print("Saved plot to neutrino_direction_requirement.png")

