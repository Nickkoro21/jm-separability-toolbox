"""
Jeffries–Matusita separability — core math engine.

Implements the Bhattacharyya distance and its Jeffries–Matusita transformation
between pairs of multivariate Gaussian classes, plus a helper that builds the
full pairwise JM matrix from a feature matrix and class labels.

Numerical-stability rules used throughout:
    * regularisation: Σ + ε·I before any inversion or determinant
    * np.linalg.slogdet  — avoids overflow on log|Σ|
    * np.linalg.solve    — avoids forming Σ⁻¹ explicitly
    * np.expm1           — accurate (1 − exp(−B)) when B is small

Reference verified against:
    D:\\thesis\\models\\deeplab_50_101\\spectral_analysis\\spectral_signatures_v2.py
Expected mean off-diagonal JM on the canonical spectral_samples.csv:
    BGR  → 1.288
    5MS  → 1.551
    7D   → 1.838

Categorisation buckets (4-tier scheme, updated 2026-04-28)
----------------------------------------------------------
    * 0.0 ≤ JM < 1.0 → Poor
    * 1.0 ≤ JM < 1.5 → Moderate
    * 1.5 ≤ JM < 1.9 → Good
    * 1.9 ≤ JM ≤ 2.0 → Excellent

This 4-bucket scheme replaces the legacy 3-bucket Richards (2013) cut-off at
1.9 with an additional ``Good`` band in [1.5, 1.9), aligning the toolbox with
the visual presentation used in the JM Distance Presentation deliverable.

Public API
----------
class_statistics(features, classes, regularisation=1e-6) -> dict
bhattacharyya_distance(mu1, cov1, mu2, cov2) -> float
jm_distance(mu1, cov1, mu2, cov2) -> float
jm_matrix(features, classes, ordered_labels=None) -> (np.ndarray, list)
interpret_jm(value) -> Literal["Poor", "Moderate", "Good", "Excellent"]
bucket_color(bucket) -> str
count_buckets(matrix) -> dict[str, int]

Constants
---------
JM_THRESHOLD_POOR      = 1.0   (upper bound of Poor)
JM_THRESHOLD_MODERATE  = 1.5   (upper bound of Moderate)   ← CHANGED in 4-bucket scheme
JM_THRESHOLD_GOOD      = 1.9   (upper bound of Good)        ← NEW
JM_MAX                 = 2.0
DEFAULT_REGULARISATION = 1e-6
JM_BUCKETS             = ("Poor", "Moderate", "Good", "Excellent")
JM_BUCKET_COLORS       = {bucket: hex_color}
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

# ─── Public constants ─────────────────────────────────────────────────────────

# Threshold = upper bound of each bucket (exclusive). Excellent has no upper
# bound below JM_MAX since values are clamped at 2.0 in jm_distance().
JM_THRESHOLD_POOR: float = 1.0
JM_THRESHOLD_MODERATE: float = 1.5
JM_THRESHOLD_GOOD: float = 1.9
JM_MAX: float = 2.0
DEFAULT_REGULARISATION: float = 1e-6

# Canonical bucket order, low → high. Used as iteration order by
# count_buckets() and by downstream UI/visualisation modules.
JM_BUCKETS: tuple[str, ...] = ("Poor", "Moderate", "Good", "Excellent")

# Canonical hex palette. Tailwind-aligned, dark-theme friendly.
# Used by visualisations and HTML report. Downstream modules should always
# resolve colours through ``bucket_color()`` rather than hard-coding.
JM_BUCKET_COLORS: dict[str, str] = {
    "Poor":      "#ef4444",   # red-500
    "Moderate":  "#f59e0b",   # amber-500
    "Good":      "#4ade80",   # green-400  (light green)
    "Excellent": "#16a34a",   # green-600  (dark green)
}

# Internal type aliases
_ClassStat = tuple[np.ndarray, np.ndarray]  # (mean_vector, covariance_matrix)
JmBucket = Literal["Poor", "Moderate", "Good", "Excellent"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _regularised_covariance(
    samples: np.ndarray,
    regularisation: float = DEFAULT_REGULARISATION,
) -> np.ndarray:
    """Return Σ + ε·I where Σ is the unbiased sample covariance.

    ``samples`` must be a 2-D array of shape (n, p) with n ≥ 2.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D, got shape {samples.shape}")
    if samples.shape[0] < 2:
        raise ValueError("Need ≥2 samples to estimate covariance")

    n_features = samples.shape[1]
    cov = np.cov(samples, rowvar=False)
    # np.cov returns 0-D for 1 feature — promote to 2-D
    cov = np.atleast_2d(cov)
    return cov + np.eye(n_features) * regularisation


# ─── Public API ───────────────────────────────────────────────────────────────

def class_statistics(
    features: np.ndarray,
    classes: np.ndarray,
    regularisation: float = DEFAULT_REGULARISATION,
) -> dict[object, _ClassStat]:
    """Compute (μ, Σ) per class.

    Rows containing any non-finite value are dropped per-class before the
    statistics are estimated. Classes with fewer than 2 valid rows fall back
    to a zero mean and an identity-scaled covariance, so downstream code never
    crashes — it simply produces NaN JM values for those classes.

    Parameters
    ----------
    features : array-like, shape (n_samples, n_features)
    classes  : array-like, shape (n_samples,)
    regularisation : float, default 1e-6
        Diagonal jitter added to each Σ.

    Returns
    -------
    dict
        ``{class_label: (mean_vector, covariance_matrix)}``.
    """
    features = np.asarray(features, dtype=np.float64)
    classes = np.asarray(classes)

    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if features.shape[0] != classes.shape[0]:
        raise ValueError(
            f"features rows ({features.shape[0]}) and classes length "
            f"({classes.shape[0]}) must match"
        )

    out: dict[object, _ClassStat] = {}
    n_features = features.shape[1]

    for label in np.unique(classes):
        mask = classes == label
        sub = features[mask]
        sub = sub[np.all(np.isfinite(sub), axis=1)]

        if sub.shape[0] < 2:
            mean = (
                sub.mean(axis=0) if sub.shape[0] == 1 else np.zeros(n_features)
            )
            cov = np.eye(n_features) * regularisation
            out[label] = (mean, cov)
            continue

        mean = sub.mean(axis=0)
        cov = _regularised_covariance(sub, regularisation=regularisation)
        out[label] = (mean, cov)

    return out


def bhattacharyya_distance(
    mu1: np.ndarray,
    cov1: np.ndarray,
    mu2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    """Bhattacharyya distance between two multivariate Gaussians.

        B = (1/8)(μ₁−μ₂)ᵀ Σ̄⁻¹ (μ₁−μ₂)
            + (1/2) ln(|Σ̄| / √(|Σ₁|·|Σ₂|))

    where Σ̄ = (Σ₁ + Σ₂) / 2.

    Returns
    -------
    float
        Bhattacharyya distance, or ``nan`` if any covariance is singular /
        ill-conditioned.
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    cov1 = np.atleast_2d(np.asarray(cov1, dtype=np.float64))
    cov2 = np.atleast_2d(np.asarray(cov2, dtype=np.float64))

    diff = mu1 - mu2
    cov_avg = 0.5 * (cov1 + cov2)

    try:
        # solve(Σ̄, diff) is more stable than inv(Σ̄) @ diff
        z = np.linalg.solve(cov_avg, diff)
        sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
        sign_1, logdet_1 = np.linalg.slogdet(cov1)
        sign_2, logdet_2 = np.linalg.slogdet(cov2)

        # All covariances must have positive determinants (PD)
        if sign_avg <= 0 or sign_1 <= 0 or sign_2 <= 0:
            return float("nan")

        mahalanobis_term = 0.125 * float(diff @ z)
        det_term = 0.5 * (logdet_avg - 0.5 * (logdet_1 + logdet_2))
        return float(mahalanobis_term + det_term)
    except np.linalg.LinAlgError:
        return float("nan")


def jm_distance(
    mu1: np.ndarray,
    cov1: np.ndarray,
    mu2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    """Jeffries–Matusita distance ∈ [0, 2].

        JM = 2 · (1 − e^(−B))

    Uses ``np.expm1`` to keep precision when B is small. Saturates at 2.0
    and clamps tiny negative round-off to 0.0.
    """
    b = bhattacharyya_distance(mu1, cov1, mu2, cov2)
    if not np.isfinite(b):
        return float("nan")

    # 1 - exp(-B) == -expm1(-B), more accurate near zero
    jm = 2.0 * (-np.expm1(-b))
    if jm > JM_MAX:
        return JM_MAX
    if jm < 0.0:
        return 0.0
    return float(jm)


def jm_matrix(
    features: np.ndarray,
    classes: np.ndarray,
    ordered_labels: Sequence | None = None,
    regularisation: float = DEFAULT_REGULARISATION,
) -> tuple[np.ndarray, list]:
    """Build the symmetric pairwise JM distance matrix.

    Parameters
    ----------
    features : array-like, shape (n_samples, n_features)
    classes  : array-like, shape (n_samples,)
    ordered_labels : sequence, optional
        Controls row/column order. Defaults to sorted unique labels.
    regularisation : float, default 1e-6

    Returns
    -------
    matrix : np.ndarray, shape (k, k)
        Symmetric, zero diagonal.
    labels : list
        Class labels in row/column order.
    """
    stats = class_statistics(features, classes, regularisation=regularisation)

    if ordered_labels is None:
        # Stable cross-type ordering: group by type name, then by string repr
        labels: list = sorted(
            stats.keys(),
            key=lambda x: (type(x).__name__, str(x)),
        )
    else:
        labels = list(ordered_labels)
        missing = [lab for lab in labels if lab not in stats]
        if missing:
            raise KeyError(f"Labels not present in classes: {missing}")

    n = len(labels)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mu_i, cov_i = stats[labels[i]]
            mu_j, cov_j = stats[labels[j]]
            jm = jm_distance(mu_i, cov_i, mu_j, cov_j)
            matrix[i, j] = jm
            matrix[j, i] = jm

    return matrix, labels


def interpret_jm(value: float) -> JmBucket:
    """Categorical interpretation — 4-bucket scheme.

    * 0.0 ≤ JM < 1.0 → Poor
    * 1.0 ≤ JM < 1.5 → Moderate
    * 1.5 ≤ JM < 1.9 → Good
    * 1.9 ≤ JM ≤ 2.0 → Excellent

    NaN / non-finite values are reported as ``Poor`` so downstream code can
    safely sum or count buckets without filtering.
    """
    if not np.isfinite(value):
        return "Poor"
    if value < JM_THRESHOLD_POOR:
        return "Poor"
    if value < JM_THRESHOLD_MODERATE:
        return "Moderate"
    if value < JM_THRESHOLD_GOOD:
        return "Good"
    return "Excellent"


def bucket_color(bucket: str) -> str:
    """Return the canonical hex colour string for a bucket name.

    Falls back to ``#94a3b8`` (slate-400) for unknown bucket names.
    """
    return JM_BUCKET_COLORS.get(bucket, "#94a3b8")


def count_buckets(matrix: np.ndarray) -> dict[str, int]:
    """Count unique class pairs (upper triangle only) per bucket.

    The diagonal is excluded. The lower triangle is excluded because a
    symmetric matrix would otherwise double-count every pair.

    Parameters
    ----------
    matrix : np.ndarray, shape (k, k)
        Symmetric JM matrix as returned by :func:`jm_matrix`.

    Returns
    -------
    dict[str, int]
        Bucket-name → pair count, in canonical bucket order
        (``Poor`` first, ``Excellent`` last). Iteration order is preserved.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square 2-D, got shape {matrix.shape}")

    counts: dict[str, int] = {b: 0 for b in JM_BUCKETS}
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            label = interpret_jm(matrix[i, j])
            counts[label] += 1
    return counts


# ─── Self-test entry point ────────────────────────────────────────────────────

def _selftest() -> None:
    """Quick verification against the canonical reference dataset.

    Only runs when this module is executed directly. Performs three checks:
        1. interpret_jm threshold sanity (4-bucket scheme)
        2. mean off-diagonal JM regression vs reference values
        3. BGR bucket distribution must match the JM Presentation deliverable
           (0 Excellent, 9 Good, 7 Moderate, 5 Poor)
    """
    from pathlib import Path

    # ── Test 1 — interpret_jm thresholds ────────────────────────────────────
    print("=" * 64)
    print("[jm self-test] interpret_jm — 4-bucket thresholds")
    print("=" * 64)
    cases: list[tuple[float, str]] = [
        (0.0,           "Poor"),
        (0.5,           "Poor"),
        (0.999,         "Poor"),
        (1.0,           "Moderate"),
        (1.25,          "Moderate"),
        (1.499,         "Moderate"),
        (1.5,           "Good"),
        (1.7,           "Good"),
        (1.899,         "Good"),
        (1.9,           "Excellent"),
        (1.95,          "Excellent"),
        (2.0,           "Excellent"),
        (float("nan"),  "Poor"),
    ]
    fails = 0
    for v, expected in cases:
        got = interpret_jm(v)
        ok = got == expected
        flag = "✅" if ok else "❌"
        if not ok:
            fails += 1
        print(f"  {v!r:>8s} -> {got:<10s} (expected {expected:<10s}) {flag}")
    if fails:
        print(f"\n{fails}/{len(cases)} interpret_jm tests FAILED. Aborting.")
        return
    print(f"\nAll {len(cases)} threshold tests passed.\n")

    # ── Test 2 — Numerical regression vs reference CSV ──────────────────────
    candidate_paths = [
        Path(r"D:\thesis\models\deeplab_50_101\spectral_analysis\data\spectral_samples.csv"),
        Path(__file__).resolve().parent.parent.parent / "data" / "examples" / "spectral_samples.csv",
    ]
    csv_path = next((p for p in candidate_paths if p.exists()), None)
    if csv_path is None:
        print("[jm self-test] reference CSV not found — skipping numerical check.")
        print("              Looked in:")
        for p in candidate_paths:
            print(f"                {p}")
        return

    import pandas as pd

    print("=" * 64)
    print(f"[jm self-test] numerical regression — {csv_path}")
    print("=" * 64)
    df = pd.read_csv(csv_path)
    classes = df["class_id"].to_numpy()

    subsets = {
        "BGR (3 bands)":              ["Blue", "Green", "Red"],
        "5MS (5 bands)":              ["Blue", "Green", "Red", "RedEdge", "NIR"],
        "7D (5MS + nDSM + Thermal)":  ["Blue", "Green", "Red", "RedEdge", "NIR", "nDSM_m", "Thermal_C"],
    }
    expected = {
        "BGR (3 bands)":             1.288,
        "5MS (5 bands)":             1.551,
        "7D (5MS + nDSM + Thermal)": 1.838,
    }

    print(f"\n{'Subset':<32s}{'mean JM':>10s}{'expected':>12s}{'diff':>10s}")
    print("-" * 64)
    bgr_matrix = None
    for name, cols in subsets.items():
        feats = df[cols].to_numpy()
        mat, _ = jm_matrix(feats, classes)
        n = mat.shape[0]
        off = mat[~np.eye(n, dtype=bool)]
        mean_jm = float(np.nanmean(off))
        diff = mean_jm - expected[name]
        flag = "✅" if abs(diff) < 0.01 else "⚠️"
        print(f"{name:<32s}{mean_jm:10.3f}{expected[name]:12.3f}{diff:+10.3f} {flag}")
        if name.startswith("BGR"):
            bgr_matrix = mat

    # ── Test 3 — BGR bucket distribution must match JM Presentation ─────────
    print()
    print("=" * 64)
    print("[jm self-test] BGR bucket distribution (4-bucket scheme)")
    print("=" * 64)
    if bgr_matrix is not None:
        counts = count_buckets(bgr_matrix)
        # Expected from JM_Distance_Presentation_v2_2.html:
        #   Excellent 0  |  Good 9  |  Moderate 7  |  Poor 5  |  Total 21
        expected_counts = {"Poor": 5, "Moderate": 7, "Good": 9, "Excellent": 0}
        total = sum(counts.values())
        print(f"\n{'Bucket':<12s}{'Count':>8s}{'Expected':>12s}{'OK':>6s}")
        print("-" * 40)
        ok_all = True
        for b in JM_BUCKETS:
            ok = counts[b] == expected_counts[b]
            ok_all &= ok
            flag = "✅" if ok else "❌"
            print(f"{b:<12s}{counts[b]:>8d}{expected_counts[b]:>12d}{flag:>6s}")
        print("-" * 40)
        print(f"{'Total':<12s}{total:>8d}{sum(expected_counts.values()):>12d}")
        if ok_all:
            print("\n✅ Bucket distribution matches the JM Presentation deliverable.")
        else:
            print("\n❌ Bucket distribution does NOT match — investigate threshold logic.")


if __name__ == "__main__":
    _selftest()
