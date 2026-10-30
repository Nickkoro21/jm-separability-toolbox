"""Tests for src.core.exhaustive — exhaustive band-combination JM search.

Run with ``pytest`` from the repository root.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core import jm_matrix
from src.core.exhaustive import (
    BAND_SEP,
    DEFAULT_MIN_K,
    MAX_BANDS,
    RANKING_COLUMNS,
    count_combinations,
    exhaustive_search,
)

DEMO_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "spectral_samples.csv"
DEMO_BANDS = ["Blue", "Green", "Red", "RedEdge", "NIR", "nDSM_m", "Thermal_C"]


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    """4 Gaussian classes × 6 correlated bands, 250 samples each."""
    rng = np.random.default_rng(42)
    n_bands, per_class = 6, 250
    frames = []
    for c in range(4):
        A = rng.normal(size=(n_bands, n_bands))
        cov = A @ A.T + np.eye(n_bands)
        mu = rng.normal(loc=c * 1.5, scale=1.0, size=n_bands)
        X = rng.multivariate_normal(mu, cov, size=per_class)
        f = pd.DataFrame(X, columns=[f"b{i}" for i in range(n_bands)])
        f["cls"] = c
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def demo() -> pd.DataFrame:
    if not DEMO_CSV.exists():
        pytest.skip("bundled demo CSV not found")
    return pd.read_csv(DEMO_CSV)


# ─── count_combinations ───────────────────────────────────────────────────────

def test_count_combinations_basic():
    assert count_combinations(7, 3) == 35 + 35 + 21 + 7 + 1 == 99
    assert count_combinations(5, 3, 4) == 10 + 5
    assert count_combinations(4, 5) == 0
    assert count_combinations(20, 3) == 2**20 - 1 - 20 - 190


# ─── Numerical equivalence with jm_matrix ─────────────────────────────────────

def test_matches_jm_matrix_per_combination(synthetic):
    bands = [c for c in synthetic.columns if c != "cls"]
    res = exhaustive_search(synthetic, "cls", bands, min_k=2)
    assert res.n_combinations == count_combinations(len(bands), 2)
    assert len(res.ranking) == res.n_combinations

    by_bands = res.ranking.set_index("bands")
    for k in range(2, len(bands) + 1):
        for combo in combinations(bands, k):
            mat, _ = jm_matrix(synthetic[list(combo)].to_numpy(), synthetic["cls"].to_numpy())
            iu, ju = np.triu_indices(mat.shape[0], k=1)
            vals = mat[iu, ju]
            row = by_bands.loc[BAND_SEP.join(combo)]
            assert row["k"] == k
            assert row["mean_jm"] == pytest.approx(float(np.nanmean(vals)), abs=1e-9)
            assert row["min_jm"] == pytest.approx(float(np.nanmin(vals)), abs=1e-9)
            assert row["max_jm"] == pytest.approx(float(np.nanmax(vals)), abs=1e-9)


def test_ranking_schema_and_order(synthetic):
    bands = [c for c in synthetic.columns if c != "cls"]
    res = exhaustive_search(synthetic, "cls", bands)
    r = res.ranking
    assert list(r.columns) == list(RANKING_COLUMNS)
    # Sorted by mean desc, then min desc; ranks are 1..N permutations.
    assert (np.diff(r["mean_jm"].to_numpy()) <= 1e-12).all()
    assert sorted(r["rank_mean"]) == list(range(1, len(r) + 1))
    assert sorted(r["rank_min"]) == list(range(1, len(r) + 1))
    assert r.loc[r["rank_min"] == 1, "min_jm"].iloc[0] == pytest.approx(r["min_jm"].max())
    # Bucket counts sum to the number of class pairs (4 classes → 6 pairs).
    counts = r[["count_poor", "count_moderate", "count_good", "count_excellent"]].sum(axis=1)
    assert (counts == 6).all()
    assert set(r["k"].unique()) == set(range(DEFAULT_MIN_K, len(bands) + 1))


def test_best_per_k_and_subsets(synthetic):
    bands = [c for c in synthetic.columns if c != "cls"]
    res = exhaustive_search(synthetic, "cls", bands)
    best = res.best_per_k("mean")
    assert list(best["k"]) == list(range(3, 7))
    # Each best-per-k row must be the max mean_jm among rows with that k.
    for _, row in best.iterrows():
        assert row["mean_jm"] == pytest.approx(res.ranking.loc[res.ranking["k"] == row["k"], "mean_jm"].max())
    subsets = res.subsets_for_tab5()
    assert list(subsets) == ["Best-3", "Best-4", "Best-5", "Best-6"]
    assert sorted(subsets["Best-6"]) == sorted(bands)
    # Capping keeps the smallest k's and always the full set last.
    capped = res.subsets_for_tab5(max_subsets=2)
    assert list(capped) == ["Best-3", "Best-6"]
    # top() under the min criterion is sorted by min_jm desc.
    t = res.top(5, by="min")
    assert (np.diff(t["min_jm"].to_numpy()) <= 1e-12).all()


def test_progress_callback_and_class_labels(synthetic):
    bands = [c for c in synthetic.columns if c != "cls"]
    calls: list[tuple[float, str]] = []
    labels = {0: "Water", 1: "Urban", 2: "Forest", 3: "Soil"}
    res = exhaustive_search(
        synthetic, "cls", bands, min_k=5,
        class_labels=labels, progress=lambda f, m: calls.append((f, m)),
    )
    assert calls and calls[-1][0] == pytest.approx(1.0)
    assert res.class_names == labels
    worst = res.ranking["worst_pair"].astype(str).iloc[0]
    assert " – " in worst and all(p in labels.values() for p in worst.split(" – "))


# ─── Guards ───────────────────────────────────────────────────────────────────

def test_rejects_bad_inputs(synthetic):
    bands = [c for c in synthetic.columns if c != "cls"]
    with pytest.raises(ValueError, match="at least 2"):
        exhaustive_search(synthetic, "cls", bands[:1])
    with pytest.raises(ValueError, match="capped"):
        many = pd.concat([synthetic] + [synthetic[bands].add_suffix(f"_{i}") for i in range(3)], axis=1)
        exhaustive_search(many, "cls", [c for c in many.columns if c != "cls"][: MAX_BANDS + 1])
    with pytest.raises(ValueError, match="not found"):
        exhaustive_search(synthetic, "cls", bands + ["nope"])
    with pytest.raises(ValueError, match="min_k"):
        exhaustive_search(synthetic, "cls", bands, min_k=7)
    with pytest.raises(ValueError, match="duplicates"):
        exhaustive_search(synthetic, "cls", bands + [bands[0]])
    with pytest.raises(ValueError, match="2 classes"):
        exhaustive_search(synthetic[synthetic["cls"] == 0], "cls", bands)


def test_handles_singular_class_without_crashing(synthetic):
    """A class with a constant band must yield NaN/degenerate JM, not raise."""
    df = synthetic.copy()
    bands = [c for c in df.columns if c != "cls"]
    df.loc[df["cls"] == 0, "b0"] = 1.0          # zero variance in one class
    res = exhaustive_search(df, "cls", bands, min_k=5)
    assert len(res.ranking) == count_combinations(len(bands), 5)
    assert np.isfinite(res.ranking["mean_jm"]).any()


# ─── Regression on the bundled demo dataset ───────────────────────────────────

def test_demo_dataset_canonical_values(demo):
    names = demo.groupby("class_id")["class_name"].first().to_dict()
    res = exhaustive_search(demo, "class_id", DEMO_BANDS, class_labels=names)
    assert res.n_combinations == 99
    r = res.ranking.set_index("bands")
    assert r.loc["Blue + Green + Red", "mean_jm"] == pytest.approx(1.288, abs=0.001)
    assert r.loc["Blue + Green + Red + RedEdge + NIR", "mean_jm"] == pytest.approx(1.551, abs=0.001)
    assert r.loc[BAND_SEP.join(DEMO_BANDS), "mean_jm"] == pytest.approx(1.838, abs=0.001)
    # The full 7-band set is the global optimum under both criteria.
    assert res.ranking.iloc[0]["k"] == 7
    assert res.top(1, by="min").iloc[0]["k"] == 7
    # Best 3-band subset by mean JM is Red + NIR + nDSM (thesis finding).
    best3 = res.best_per_k("mean").set_index("k").loc[3]
    assert sorted(res.bands_of(best3["bands"])) == ["NIR", "Red", "nDSM_m"]
    assert best3["mean_jm"] == pytest.approx(1.607, abs=0.001)
