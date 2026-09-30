"""Decompose the static two-tau azimuthal harmonic into marginal and connected parts.

The primary population and statistic were fixed in the run note before this
script was executed.  This script reads validation predictions only; optional
train predictions are used for a supportive train/validation comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARMS = {
    "base": "gen3pi_base_s43.npz",
    "ip_sv": "gen3pi_full22_s42.npz",
    "geometry_shuffle": "cleo_full22_shuffle_s42.npz",
    "ideal_ip": "gen3pi_idealip22_s42.npz",
}
MODE_NAMES = {0: "pi", 1: "rho", 3: "3pi"}
MODE_PAIRS = ((0, 0), (0, 1), (0, 3), (1, 1), (1, 3), (3, 3))
SEED = 20261001
BOOTSTRAPS = 2000


def azimuth(h: np.ndarray) -> np.ndarray:
    """Azimuth in the transverse (n,r) plane; undefined zero-vectors are rejected upstream."""
    return np.arctan2(h[..., 1], h[..., 0])


def complex_moments(h: np.ndarray) -> dict[str, complex]:
    phi = azimuth(h)
    u = np.exp(1j * phi[:, 0])
    v = np.exp(-1j * phi[:, 1])
    obs = np.mean(u * v)
    fact = np.mean(u) * np.mean(v)
    return {"obs": obs, "fact": fact, "conn": obs - fact}


def serialise_complex(z: complex) -> dict[str, float]:
    return {
        "real": float(np.real(z)),
        "imag": float(np.imag(z)),
        "amplitude": float(2 * abs(z)),
        "phase_deg": float(np.degrees(np.angle(z))),
    }


def derived(m: dict[str, complex]) -> dict[str, float]:
    obs = m["obs"]
    den = max(abs(obs), 1e-15)
    ratio = m["conn"] / obs if abs(obs) > 1e-15 else complex(np.nan, np.nan)
    return {
        "abs_fact_over_obs": float(abs(m["fact"]) / den),
        "abs_conn_over_obs": float(abs(m["conn"]) / den),
        "aligned_fact_fraction": float(np.real(m["fact"] * np.conj(obs)) / den**2),
        "aligned_conn_fraction": float(np.real(m["conn"] * np.conj(obs)) / den**2),
        "conn_over_obs_real": float(np.real(ratio)),
        "conn_over_obs_imag": float(np.imag(ratio)),
    }


def summarise(h: np.ndarray, bootstraps: int, seed: int) -> dict:
    if h.ndim != 3 or h.shape[1:] != (2, 3):
        raise ValueError(f"expected [N,2,3], got {h.shape}")
    transverse = np.linalg.norm(h[..., :2], axis=-1)
    finite = np.isfinite(h).all(axis=(1, 2)) & (transverse > 1e-12).all(axis=1)
    if not finite.all():
        raise ValueError(f"non-finite or zero-transverse rows: {(~finite).sum()} / {len(h)}")
    m = complex_moments(h)
    point = {k: serialise_complex(v) for k, v in m.items()}
    point.update(derived(m))

    rng = np.random.default_rng(seed)
    keys = ("obs_amplitude", "fact_amplitude", "conn_amplitude",
            "abs_fact_over_obs", "abs_conn_over_obs",
            "aligned_fact_fraction", "aligned_conn_fraction")
    reps = {k: np.empty(bootstraps, dtype=np.float64) for k in keys}
    n = len(h)
    for b in range(bootstraps):
        idx = rng.integers(0, n, n)
        mb = complex_moments(h[idx])
        db = derived(mb)
        reps["obs_amplitude"][b] = 2 * abs(mb["obs"])
        reps["fact_amplitude"][b] = 2 * abs(mb["fact"])
        reps["conn_amplitude"][b] = 2 * abs(mb["conn"])
        for k in keys[3:]:
            reps[k][b] = db[k]
    point["bootstrap"] = {
        k: {
            "mean": float(np.mean(v)),
            "se": float(np.std(v, ddof=1)),
            "ci95": [float(x) for x in np.percentile(v, (2.5, 97.5))],
        }
        for k, v in reps.items()
    }
    point["n"] = int(n)
    return point


def permutation_null(h: np.ndarray, permutations: int, seed: int) -> dict:
    """Conditional independence null obtained by permuting the plus side."""
    phi = azimuth(h)
    u = np.exp(1j * phi[:, 0])
    v = np.exp(-1j * phi[:, 1])
    observed = abs(np.mean(u * v))
    expected = np.mean(u) * np.mean(v)
    rng = np.random.default_rng(seed)
    moments = np.empty(permutations, dtype=np.complex128)
    for i in range(permutations):
        moments[i] = np.mean(u * v[rng.permutation(len(v))])
    amplitudes = 2 * np.abs(moments)
    return {
        "permutations": int(permutations),
        "observed_amplitude": float(2 * observed),
        "null_amplitude_median": float(np.median(amplitudes)),
        "null_amplitude_ci95": [float(x) for x in np.percentile(amplitudes, (2.5, 97.5))],
        "p_ge_observed": float((1 + np.sum(np.abs(moments) >= observed)) / (permutations + 1)),
        "mean_permuted_real_minus_fact": float(np.real(np.mean(moments) - expected)),
        "mean_permuted_imag_minus_fact": float(np.imag(np.mean(moments) - expected)),
    }


def identity(reference: dict[str, np.ndarray], other: dict[str, np.ndarray], name: str) -> None:
    for key in ("labels", "global_indices", "truth_modes"):
        if key not in other or not np.array_equal(reference[key], other[key]):
            raise RuntimeError(f"identity mismatch for {name}: {key}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def pair_mask(modes: np.ndarray, a: int, b: int) -> np.ndarray:
    return ((modes[:, 0] == a) & (modes[:, 1] == b)) | ((modes[:, 0] == b) & (modes[:, 1] == a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=HERE / "results/moment_decomposition.json")
    ap.add_argument("--arrays", type=Path, default=HERE / "results/primary_arrays.npz")
    ap.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = ap.parse_args()

    data = {name: load_npz(args.input_dir / filename) for name, filename in ARMS.items()}
    ref = data["base"]
    for name, d in data.items():
        identity(ref, d, name)
        if not np.allclose(d["h"], ref["h"], rtol=0, atol=2e-6):
            # The geometry-shuffle file uses the CLEO 3pi teacher.  In rho-rho,
            # the exact target must still be identical; other pairs are omitted
            # from cross-arm comparisons if this whole-array check fails.
            rr = pair_mask(ref["truth_modes"], 1, 1)
            if not np.allclose(d["h"][rr], ref["h"][rr], rtol=0, atol=2e-6):
                raise RuntimeError(f"rho-rho exact-h mismatch for {name}")

    labels_h = ref["labels"].astype(bool)
    modes = ref["truth_modes"].astype(int)
    rr = pair_mask(modes, 1, 1)
    masks = {"H_rhorho": labels_h & rr, "Z_rhorho": (~labels_h) & rr}
    result = {
        "schema": 1,
        "seed": SEED,
        "bootstraps": args.bootstraps,
        "primary_population": "validation Z rho-rho",
        "identity": {
            "rows": int(len(labels_h)),
            "h_rows": int(labels_h.sum()),
            "z_rows": int((~labels_h).sum()),
            "global_index_unique": int(len(np.unique(ref["global_indices"]))),
        },
        "populations": {},
        "mode_pairs": {},
        "train_validation": {},
    }

    primary_arrays = {}
    for pop, mask in masks.items():
        result["populations"][pop] = {}
        exact = ref["h"][mask].astype(np.float64)
        result["populations"][pop]["exact_h"] = summarise(exact, args.bootstraps, SEED + 10)
        result["populations"][pop]["exact_h"]["permutation_null"] = permutation_null(
            exact, args.bootstraps, SEED + 20
        )
        if pop == "Z_rhorho":
            primary_arrays["exact_h"] = exact
        for offset, (name, d) in enumerate(data.items(), start=1):
            pred = d["h_pred"][mask].astype(np.float64)
            result["populations"][pop][name] = summarise(pred, args.bootstraps, SEED + 100 * offset + (0 if pop.startswith("Z") else 1))
            result["populations"][pop][name]["permutation_null"] = permutation_null(
                pred, args.bootstraps, SEED + 1000 * offset + (0 if pop.startswith("Z") else 1)
            )
            if pop == "Z_rhorho":
                primary_arrays[name] = pred

    # Supportive mode-pair table for base and exact h, kept separate from the primary comparison.
    for a, b in MODE_PAIRS:
        name = f"{MODE_NAMES[a]}x{MODE_NAMES[b]}"
        result["mode_pairs"][name] = {}
        pm = pair_mask(modes, a, b)
        for process, labmask in (("H", labels_h), ("Z", ~labels_h)):
            mask = pm & labmask
            result["mode_pairs"][name][process] = {
                "exact_h": summarise(ref["h"][mask].astype(np.float64), 400, SEED + a * 100 + b * 10),
                "base": summarise(ref["h_pred"][mask].astype(np.float64), 400, SEED + 500 + a * 100 + b * 10),
            }

    # Supportive train/validation stability for arms with exported train predictions.
    for name, filename in (("base", "gen3pi_base_s43_train.npz"), ("ip_sv", "gen3pi_full22_s42_train.npz")):
        path = args.input_dir / filename
        if not path.exists():
            continue
        train = load_npz(path)
        tm = train["truth_modes"].astype(int)
        ty = train["labels"].astype(bool)
        mask = (~ty) & pair_mask(tm, 1, 1)
        result["train_validation"][name] = {
            "train_Z_rhorho": summarise(train["h_pred"][mask].astype(np.float64), 400, SEED + 900),
            "validation_Z_rhorho": result["populations"]["Z_rhorho"][name],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **primary_arrays)
    print(json.dumps({
        "output": str(args.output),
        "arrays": str(args.arrays),
        "primary": {
            name: {
                "A_obs": result["populations"]["Z_rhorho"][name]["obs"]["amplitude"],
                "A_fact": result["populations"]["Z_rhorho"][name]["fact"]["amplitude"],
                "A_conn": result["populations"]["Z_rhorho"][name]["conn"]["amplitude"],
                "aligned_conn_fraction": result["populations"]["Z_rhorho"][name]["aligned_conn_fraction"],
            }
            for name in result["populations"]["Z_rhorho"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
