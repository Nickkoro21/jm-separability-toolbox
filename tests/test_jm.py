"""Tests for src.core.jm — Bhattacharyya / Jeffries–Matusita core math.

Run with ``pytest`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core import (
    JM_BUCKETS,
    JM_MAX,
    bhattacharyya_distance,
    class_statistics,
    count_buckets,
    interpret_jm,
    jm_distance,
    jm_matrix,
)

DEMO_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "spectral_samples.csv"


@pytest.mark.parametrize("value, bucket", [
    (0.0, "Poor"), (0.999, "Poor"),
    (1.0, "Moderate"), (1.499, "Moderate"),
    (1.5, "Good"), (1.899, "Good"),
    (1.9, "Excellent"), (2.0, "Excellent"),
    (float("nan"), "Poor"),
])
def test_interpret_jm_buckets(value, bucket):
    assert interpret_jm(value) == bucket


def test_identical_gaussians_have_zero_distance():
    mu, cov = np.zeros(3), np.eye(3)
    assert bhattacharyya_distance(mu, cov, mu, cov) == pytest.approx(0.0)
    assert jm_distance(mu, cov, mu, cov) == pytest.approx(0.0)


def test_far_apart_gaussians_saturate_at_two():
    mu1, mu2, cov = np.zeros(2), np.full(2, 50.0), np.eye(2)
    assert jm_distance(mu1, cov, mu2, cov) == pytest.approx(JM_MAX)


def test_shape_term_detects_different_spread():
    """Same mean, different covariance → strictly positive distance."""
    mu = np.zeros(2)
    assert jm_distance(mu, np.eye(2), mu, 9 * np.eye(2)) > 0.3


def test_singular_covariance_returns_nan():
    mu = np.zeros(2)
    sing = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.isnan(bhattacharyya_distance(mu, sing, mu + 1, sing))


def test_jm_matrix_symmetry_and_buckets():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(c * 3, 1, size=(200, 2)) for c in range(3)])
    y = np.repeat(["a", "b", "c"], 200)
    mat, labels = jm_matrix(X, y)
    assert labels == ["a", "b", "c"]
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 0.0)
    counts = count_buckets(mat)
    assert list(counts) == list(JM_BUCKETS)
    assert sum(counts.values()) == 3


def test_class_statistics_drops_non_finite_rows():
    X = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, 3.0], [3.0, 5.0]])
    y = np.array([0, 0, 0, 0])
    mu, cov = class_statistics(X, y)[0]
    assert mu == pytest.approx([2.0, 10 / 3])
    assert cov.shape == (2, 2)


@pytest.mark.skipif(not DEMO_CSV.exists(), reason="bundled demo CSV not found")
def test_demo_canonical_mean_jm():
    df = pd.read_csv(DEMO_CSV)
    y = df["class_id"].to_numpy()
    expected = {
        ("Blue", "Green", "Red"): 1.288,
        ("Blue", "Green", "Red", "RedEdge", "NIR"): 1.551,
        ("Blue", "Green", "Red", "RedEdge", "NIR", "nDSM_m", "Thermal_C"): 1.838,
    }
    for cols, exp in expected.items():
        mat, _ = jm_matrix(df[list(cols)].to_numpy(), y)
        off = mat[~np.eye(mat.shape[0], dtype=bool)]
        assert float(np.nanmean(off)) == pytest.approx(exp, abs=0.001)
    mat, _ = jm_matrix(df[["Blue", "Green", "Red"]].to_numpy(), y)
    assert count_buckets(mat) == {"Poor": 5, "Moderate": 7, "Good": 9, "Excellent": 0}


def test_resolve_class_label_lookup_order():
    from src.core import resolve_class_label
    assert resolve_class_label(1, None) == "1"
    assert resolve_class_label(1, {1: "Tree"}) == "Tree"
    assert resolve_class_label(1, {"1": "Tree"}) == "Tree"      # JSON-flattened keys
    assert resolve_class_label(2, {1: "Tree"}) == "2"
    assert resolve_class_label("Road", {"Road": "Road"}) == "Road"
