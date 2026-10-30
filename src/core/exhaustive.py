"""
Exhaustive band-combination search — ranks every subset of ≥ k bands by JM.

Given a feature table with ``n`` candidate bands, this module evaluates the
Jeffries–Matusita separability of **every** band combination of size
``min_k … max_k`` and returns a ranked table. It answers the question
"which bands do I actually need?" without the user having to guess subsets
by hand in Step 4.

Why it is fast
--------------
Per-class statistics (mean vector μ and covariance matrix Σ) are estimated
**once** over all ``n`` candidate bands. For any band subset *S*, the
Gaussian statistics restricted to *S* are simply ``μ[S]`` and ``Σ[S, S]``
(a principal sub-matrix), so no re-fitting is needed per combination.
The regularisation ``ε·I`` commutes with sub-matrix extraction, so the
result is numerically identical to calling :func:`src.core.jm_matrix`
on each subset individually (verified by ``tests/test_exhaustive.py``).

The Bhattacharyya distance is then evaluated in **batches**: for a chunk
of *M* combinations and one class pair, the ``(M, k, k)`` stacks of
covariances go through ``np.linalg.solve`` and ``np.linalg.slogdet`` in a
single vectorised call. On the bundled 7-band / 7-class demo this is
≈ 50 µs per combination on a laptop core; the hard cap of :data:`MAX_BANDS` = 20 bands
(≈ 1.05 M combinations for ``min_k = 3``) completes in about a minute.

Ranking criteria
----------------
Two complementary criteria are reported for every combination:

* ``mean_jm`` — mean JM over all unique class pairs. Rewards subsets that
  separate *most* pairs well. Primary sort key.
* ``min_jm``  — JM of the **worst** class pair. Rewards subsets with no
  weak link. Secondary sort key (tie-break) and basis of ``rank_min``.

"Best per *k*" tables give the hierarchical view: the single best
combination at each subset size, so the marginal gain of each additional
band is visible at a glance.

Public API
----------
count_combinations(n_bands, min_k=3, max_k=None) -> int
exhaustive_search(df, class_col, bands, *, min_k=3, max_k=None,
                  class_labels=None, progress=None) -> ExhaustiveResult
ExhaustiveResult.ranking            -> pd.DataFrame (one row per combination)
ExhaustiveResult.best_per_k(by)     -> pd.DataFrame (one row per k)
ExhaustiveResult.top(n, by)         -> pd.DataFrame
ExhaustiveResult.subsets_for_tab5() -> dict[str, list[str]]

Constants
---------
MAX_BANDS        = 20   hard cap on candidate bands (combinatorial guard)
DEFAULT_MIN_K    = 3
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations, islice
from math import comb
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .detection import resolve_class_label
from .jm import (
    DEFAULT_REGULARISATION,
    JM_BUCKETS,
    JM_MAX,
    JM_THRESHOLD_GOOD,
    JM_THRESHOLD_MODERATE,
    JM_THRESHOLD_POOR,
    class_statistics,
)

# ─── Public constants ─────────────────────────────────────────────────────────

#: Hard cap on the number of candidate bands. 20 bands with ``min_k = 3``
#: is ≈ 1.05 million combinations — the practical ceiling for an
#: interactive tool. Above this the search is refused with a clear error.
MAX_BANDS: int = 20

#: Smallest subset size searched by default (the user asked for "3 bands
#: and up" — RGB-sized subsets are the smallest that make sense for
#: land-cover work).
DEFAULT_MIN_K: int = 3

#: Target number of floats per batch when chunking combinations. Keeps the
#: ``(n_classes, M, k, k)`` covariance stack around ~25 MB.
_TARGET_FLOATS_PER_CHUNK: int = 3_000_000
_MIN_CHUNK: int = 64
_MAX_CHUNK: int = 8_192

RankCriterion = Literal["mean", "min"]

#: Separator used to join band names in ``ranking["bands"]``.
BAND_SEP: str = " + "

#: Column order of :attr:`ExhaustiveResult.ranking`.
RANKING_COLUMNS: tuple[str, ...] = (
    "rank_mean", "rank_min", "k", "bands",
    "mean_jm", "min_jm", "max_jm", "std_jm",
    "worst_pair",
    "count_poor", "count_moderate", "count_good", "count_excellent",
    "mean_bucket",
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def count_combinations(
    n_bands: int, min_k: int = DEFAULT_MIN_K, max_k: int | None = None
) -> int:
    """Number of band combinations the search would evaluate.

    ``sum(C(n, k) for k in min_k..max_k)``. Returns 0 when the range is
    empty (e.g. ``min_k > n_bands``).
    """
    if max_k is None or max_k > n_bands:
        max_k = n_bands
    if min_k < 1 or min_k > max_k:
        return 0
    return sum(comb(n_bands, k) for k in range(min_k, max_k + 1))


_BUCKET_EDGES = np.asarray(
    [JM_THRESHOLD_POOR, JM_THRESHOLD_MODERATE, JM_THRESHOLD_GOOD], dtype=np.float64,
)


def _bucket_codes(values: np.ndarray) -> np.ndarray:
    """Vectorised 4-bucket interpretation as ``int8`` codes into ``JM_BUCKETS``.

    0 = Poor, 1 = Moderate, 2 = Good, 3 = Excellent. NaN → Poor, matching
    :func:`src.core.jm.interpret_jm`. Integer codes keep the hot loop free
    of Python string objects; names are materialised only for display.
    """
    v = np.where(np.isfinite(values), values, -1.0)
    return np.searchsorted(_BUCKET_EDGES, v, side="right").astype(np.int8)


def _batched_jm_lu(
    mu_a: np.ndarray, cov_a: np.ndarray,
    mu_b: np.ndarray, cov_b: np.ndarray,
) -> np.ndarray:
    """Reference (slow-path) JM for a batch of Gaussian pairs — LU + slogdet.

    Mirrors :func:`src.core.jm.bhattacharyya_distance` element-wise and
    returns ``nan`` wherever a covariance is not positive-definite. Used
    as the fallback when the Cholesky fast path hits a non-PD matrix.

    Parameters
    ----------
    mu_a, mu_b : (M, k)
    cov_a, cov_b : (M, k, k)
    """
    diff = mu_a - mu_b
    cov_avg = 0.5 * (cov_a + cov_b)
    with np.errstate(all="ignore"):
        try:
            z = np.linalg.solve(cov_avg, diff[..., None])[..., 0]
        except np.linalg.LinAlgError:
            z = np.stack([
                np.linalg.lstsq(c, d, rcond=None)[0]
                for c, d in zip(cov_avg, diff)
            ])
        s_avg, ld_avg = np.linalg.slogdet(cov_avg)
        s_a, ld_a = np.linalg.slogdet(cov_a)
        s_b, ld_b = np.linalg.slogdet(cov_b)
        b = 0.125 * np.einsum("mk,mk->m", diff, z) + 0.5 * (ld_avg - 0.5 * (ld_a + ld_b))
        jm = 2.0 * (-np.expm1(-b))
    bad = (s_avg <= 0) | (s_a <= 0) | (s_b <= 0) | ~np.isfinite(jm)
    jm = np.clip(jm, 0.0, JM_MAX)
    jm[bad] = np.nan
    return jm


def _tri_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``L w = b`` for batches of lower-triangular ``L``.

    ``np.linalg.solve`` (batched LAPACK ``gesv``) is used on purpose: SciPy's
    batched ``solve_triangular`` loops in Python and measured ~5× slower
    for the small ``k`` typical here, despite the lower flop count.
    """
    return np.linalg.solve(L, b[..., None])[..., 0]


def _jm_all_pairs(
    mu_sub: np.ndarray, cov_sub: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray,
) -> np.ndarray:
    """JM for every class pair of a chunk of combinations — Cholesky fast path.

    Parameters
    ----------
    mu_sub : (C, M, k)       class means restricted to the subset
    cov_sub : (C, M, k, k)   class covariances restricted to the subset
    pair_i, pair_j : (P,)    class indices of each unique pair

    Returns
    -------
    (M, P) array of JM ∈ [0, 2] (``nan`` for non-PD cases).

    Notes
    -----
    With Σ̄ = L Lᵀ the Mahalanobis term is ``‖L⁻¹(μ₁−μ₂)‖² / 8`` and
    ``ln|Σ̄| = 2 Σ ln diag(L)``, so one Cholesky factorisation per pair
    replaces one LU solve plus one LU log-determinant. Per-class
    ``ln|Σ_c|`` are computed once per chunk (C factorisations instead of
    2·P). If any matrix in the chunk is not positive-definite,
    ``np.linalg.cholesky`` raises and we fall back to the element-wise
    LU path for that chunk.
    """
    with np.errstate(all="ignore"):
        try:
            L_c = np.linalg.cholesky(cov_sub)                    # (C, M, k, k)
            ld_c = 2.0 * np.log(np.diagonal(L_c, axis1=-2, axis2=-1)).sum(-1)  # (C, M)
            cov_avg = 0.5 * (cov_sub[pair_i] + cov_sub[pair_j])  # (P, M, k, k)
            diff = mu_sub[pair_i] - mu_sub[pair_j]               # (P, M, k)
            L = np.linalg.cholesky(cov_avg)
            ld_avg = 2.0 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)  # (P, M)
            w = _tri_solve(L, diff)                              # (P, M, k)
            b = 0.125 * np.einsum("pmk,pmk->pm", w, w) + 0.5 * (
                ld_avg - 0.5 * (ld_c[pair_i] + ld_c[pair_j])
            )
            jm = np.clip(2.0 * (-np.expm1(-b)), 0.0, JM_MAX)
            jm[~np.isfinite(jm)] = np.nan
            return jm.T                                          # (M, P)
        except np.linalg.LinAlgError:
            pass

    # Slow path — element-wise NaN handling for non-PD matrices.
    m = mu_sub.shape[1]
    out = np.empty((m, len(pair_i)), dtype=np.float64)
    for p, (i, j) in enumerate(zip(pair_i, pair_j)):
        out[:, p] = _batched_jm_lu(mu_sub[i], cov_sub[i], mu_sub[j], cov_sub[j])
    return out


def _unique_class_names(labels: Sequence, class_labels: Mapping | None) -> dict:
    """``{class_id: display name}`` with duplicates disambiguated.

    Two ids mapping to the same display name (e.g. both "Other") would
    collide in the ``worst_pair`` categories, so such names get the id
    appended: ``"Other (3)"``.
    """
    names = {lab: resolve_class_label(lab, class_labels) for lab in labels}
    seen: dict[str, int] = {}
    for nm in names.values():
        seen[nm] = seen.get(nm, 0) + 1
    return {
        lab: (f"{nm} ({lab})" if seen[nm] > 1 else nm)
        for lab, nm in names.items()
    }


def _chunk_size(n_classes: int, n_pairs: int, k: int) -> int:
    """Combinations per batch so the covariance stacks stay ~25 MB."""
    floats_per_combo = max(1, (n_classes + 2 * n_pairs) * k * k)
    m = _TARGET_FLOATS_PER_CHUNK // floats_per_combo
    return int(min(_MAX_CHUNK, max(_MIN_CHUNK, m)))


def _iter_index_chunks(n: int, k: int, size: int):
    """Yield ``(M, k)`` int arrays of band-index combinations, M ≤ size."""
    it = combinations(range(n), k)
    while True:
        block = list(islice(it, size))
        if not block:
            return
        yield np.asarray(block, dtype=np.intp)


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class ExhaustiveResult:
    """Outcome of :func:`exhaustive_search`.

    Attributes
    ----------
    ranking : pd.DataFrame
        One row per combination, columns :data:`RANKING_COLUMNS`, sorted by
        ``mean_jm`` desc then ``min_jm`` desc. ``bands`` is a ``" + "``-joined
        string; use :meth:`bands_of` to recover the list.
    bands : list[str]
        Candidate bands, in the order given.
    labels : list
        Class ids in the order used for pair enumeration.
    class_names : dict
        ``{class_id: display name}`` used in ``worst_pair``.
    min_k, max_k : int
    n_combinations : int
    elapsed_s : float
    kept_index : pd.Index
        Index labels of the rows actually used (finite across every
        candidate band). Downstream per-subset computations should be run
        on exactly these rows so their numbers match the ranking.
    n_rows_dropped : int
        Rows excluded because of a non-finite value in some candidate band.
    """
    ranking: pd.DataFrame
    bands: list[str]
    labels: list
    class_names: dict
    min_k: int
    max_k: int
    n_combinations: int
    elapsed_s: float
    kept_index: pd.Index = field(default_factory=lambda: pd.Index([]))
    n_rows_dropped: int = 0
    # ── accessors ────────────────────────────────────────────────────────
    def bands_of(self, bands_str: str) -> list[str]:
        """Recover the band list from a ``ranking['bands']`` string."""
        return [b for b in str(bands_str).split(BAND_SEP) if b]

    def best_per_k(self, by: RankCriterion = "mean") -> pd.DataFrame:
        """Best combination at every subset size ``k`` (hierarchical view).

        ``by="mean"`` ranks by mean JM (tie-break min JM);
        ``by="min"`` ranks by worst-pair JM (tie-break mean JM).
        """
        df = self._sorted(by)
        return (
            df.groupby("k", sort=True).head(1)
            .sort_values("k")
            .reset_index(drop=True)
        )

    def top(self, n: int = 20, by: RankCriterion = "mean") -> pd.DataFrame:
        """Top-``n`` combinations overall under the chosen criterion."""
        return self._sorted(by).head(n).reset_index(drop=True)

    def subsets_for_tab5(
        self, max_subsets: int | None = None, by: RankCriterion = "mean",
    ) -> dict[str, list[str]]:
        """Best-per-k combinations as a ``{name: bands}`` subset mapping.

        Names are ``"Best-k"`` (e.g. ``Best-3``). When ``max_subsets`` is
        smaller than the number of k levels, the smallest k values are kept
        (band *reduction* is the usual question) and the full set is always
        appended last so the ceiling is visible.
        """
        best = self.best_per_k(by)
        rows = list(best.itertuples(index=False))
        if max_subsets is not None and len(rows) > max_subsets:
            full = [r for r in rows if r.k == self.max_k]
            head = [r for r in rows if r.k != self.max_k][: max(0, max_subsets - len(full))]
            rows = head + full
        return {f"Best-{r.k}": self.bands_of(r.bands) for r in rows}

    def _sorted(self, by: RankCriterion) -> pd.DataFrame:
        """View of ``ranking`` ordered by the criterion.

        ``ranking`` is stored in ``by="mean"`` order and carries both rank
        columns, so no multi-key re-sort of the (possibly million-row)
        table is needed: the mean view is the table itself and the min
        view is a single integer sort on ``rank_min``.
        """
        if by == "mean":
            return self.ranking
        if by == "min":
            return self.ranking.sort_values("rank_min", kind="mergesort")
        raise ValueError(f"by must be 'mean' or 'min', got {by!r}")


# ─── Main entry point ─────────────────────────────────────────────────────────

def exhaustive_search(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    min_k: int = DEFAULT_MIN_K,
    max_k: int | None = None,
    class_labels: Mapping[Any, str] | None = None,
    regularisation: float = DEFAULT_REGULARISATION,
    progress: Callable[[float, str], Any] | None = None,
) -> ExhaustiveResult:
    """Evaluate JM separability for every band combination of size k.

    Parameters
    ----------
    df : pd.DataFrame
        Samples; must contain ``class_col`` and every name in ``bands``.
    class_col : str
    bands : sequence of str
        Candidate bands (2 ≤ len ≤ :data:`MAX_BANDS`). Rows with any
        non-finite value across these bands are dropped once, up front
        (``ExhaustiveResult.kept_index`` / ``n_rows_dropped`` report
        this). Note that :func:`src.core.jm_matrix` drops NaN rows per
        subset instead, so to reproduce a ranking value with
        ``jm_matrix`` run it on ``df.loc[result.kept_index]``.
    min_k, max_k : int
        Subset-size range (inclusive). ``max_k`` defaults to ``len(bands)``.
    class_labels : mapping, optional
        ``{class_id: display name}`` for the ``worst_pair`` column.
    regularisation : float
        Diagonal jitter ε added to each class covariance (as in ``jm.py``).
    progress : callable, optional
        ``progress(fraction, message)`` called after each batch — plug in
        ``gr.Progress()`` or a tqdm wrapper.

    Returns
    -------
    ExhaustiveResult

    Raises
    ------
    ValueError
        On bad inputs: too few / too many bands, unknown columns, fewer
        than 2 classes, or an empty k range.
    """
    bands = [str(b) for b in bands]
    n = len(bands)
    if n < 2:
        raise ValueError(f"Need at least 2 candidate bands, got {n}.")
    if n > MAX_BANDS:
        raise ValueError(
            f"Too many candidate bands ({n}); the exhaustive search is "
            f"capped at {MAX_BANDS} to keep the number of combinations "
            f"tractable. Deselect some bands or raise min_k."
        )
    if len(set(bands)) != n:
        raise ValueError("Candidate bands contain duplicates.")
    clash = [b for b in bands if BAND_SEP in b]
    if clash:
        raise ValueError(
            f"Band names must not contain {BAND_SEP!r}: {clash}"
        )
    missing = [b for b in bands if b not in df.columns]
    if missing:
        raise ValueError(f"Bands not found in data: {missing}")
    if class_col not in df.columns:
        raise ValueError(f"Class column {class_col!r} not found in data.")

    if max_k is None or max_k > n:
        max_k = n
    min_k = max(1, int(min_k))
    if min_k > max_k:
        raise ValueError(
            f"min_k ({min_k}) exceeds the number of candidate bands ({max_k})."
        )

    # ── Per-class statistics on the FULL band set (once) ─────────────────
    X = df[bands].to_numpy(dtype=np.float64)
    y = df[class_col].to_numpy()
    finite = np.all(np.isfinite(X), axis=1)
    X, y = X[finite], y[finite]
    kept_index = df.index[finite]
    n_rows_dropped = int((~finite).sum())
    if X.shape[0] == 0:
        raise ValueError("No rows with finite values across the candidate bands.")

    stats = class_statistics(X, y, regularisation=regularisation)
    labels = sorted(stats.keys(), key=lambda x: (type(x).__name__, str(x)))
    n_classes = len(labels)
    if n_classes < 2:
        raise ValueError(f"JM needs ≥ 2 classes; found {n_classes}.")

    MU = np.stack([stats[lab][0] for lab in labels])    # (C, n)
    COV = np.stack([stats[lab][1] for lab in labels])   # (C, n, n)

    class_names = _unique_class_names(labels, class_labels)
    pair_idx = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
    pair_i = np.asarray([i for i, _ in pair_idx], dtype=np.intp)
    pair_j = np.asarray([j for _, j in pair_idx], dtype=np.intp)
    pair_names = [f"{class_names[labels[i]]} – {class_names[labels[j]]}" for i, j in pair_idx]
    n_pairs = len(pair_idx)

    total = count_combinations(n, min_k, max_k)
    done = 0
    t0 = time.perf_counter()

    # Accumulators (filled per chunk, concatenated at the end)
    acc_k: list[np.ndarray] = []
    acc_bands: list[list[str]] = []
    acc_mean: list[np.ndarray] = []
    acc_min: list[np.ndarray] = []
    acc_max: list[np.ndarray] = []
    acc_std: list[np.ndarray] = []
    acc_worst: list[np.ndarray] = []
    acc_counts: list[np.ndarray] = []   # (M, 4) in JM_BUCKETS order

    for k in range(min_k, max_k + 1):
        size = _chunk_size(n_classes, n_pairs, k)
        for idx in _iter_index_chunks(n, k, size):          # (M, k)
            m = idx.shape[0]
            # Gather μ[S] and Σ[S, S] for every class and every combo.
            mu_sub = MU[:, idx]                                      # (C, M, k)
            cov_sub = COV[:, idx[:, :, None], idx[:, None, :]]       # (C, M, k, k)

            jm_pairs = _jm_all_pairs(mu_sub, cov_sub, pair_i, pair_j)   # (M, P)

            with np.errstate(all="ignore"):
                mean = np.nanmean(jm_pairs, axis=1)
                mn = np.nanmin(jm_pairs, axis=1)
                mx = np.nanmax(jm_pairs, axis=1)
                sd = np.nanstd(jm_pairs, axis=1)
            # argmin of NaN-filled rows: treat NaN as -inf so it is "worst".
            worst = np.argmin(np.where(np.isnan(jm_pairs), -np.inf, jm_pairs), axis=1)

            codes = _bucket_codes(jm_pairs)                         # (M, P) int8
            counts = np.stack(
                [(codes == c).sum(axis=1) for c in range(len(JM_BUCKETS))], axis=1,
            )

            names_chunk = [BAND_SEP.join(bands[t] for t in row) for row in idx]

            acc_k.append(np.full(m, k, dtype=np.int64))
            acc_bands.append(names_chunk)
            acc_mean.append(mean); acc_min.append(mn)
            acc_max.append(mx);    acc_std.append(sd)
            acc_worst.append(worst.astype(np.int32))
            acc_counts.append(counts)

            done += m
            if progress is not None and total:
                progress(done / total, f"{done:,} / {total:,} combinations (k={k})")

    elapsed = time.perf_counter() - t0

    counts_all = np.concatenate(acc_counts) if acc_counts else np.zeros((0, 4), int)
    mean_all = np.concatenate(acc_mean) if acc_mean else np.zeros(0)
    ranking = pd.DataFrame({
        "k": np.concatenate(acc_k) if acc_k else np.zeros(0, int),
        "bands": [s for chunk in acc_bands for s in chunk],
        "mean_jm": mean_all,
        "min_jm": np.concatenate(acc_min) if acc_min else np.zeros(0),
        "max_jm": np.concatenate(acc_max) if acc_max else np.zeros(0),
        "std_jm": np.concatenate(acc_std) if acc_std else np.zeros(0),
        "worst_pair": pd.Categorical.from_codes(
            np.concatenate(acc_worst) if acc_worst else np.zeros(0, np.int32),
            categories=pair_names,
        ),
        "count_poor": counts_all[:, 0],
        "count_moderate": counts_all[:, 1],
        "count_good": counts_all[:, 2],
        "count_excellent": counts_all[:, 3],
    })
    ranking["mean_bucket"] = pd.Categorical.from_codes(
        _bucket_codes(mean_all), categories=list(JM_BUCKETS),
    )

    # Ranks under both criteria (1 = best). NaN means sort last.
    ranking = ranking.sort_values(
        ["mean_jm", "min_jm"], ascending=[False, False],
        kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    ranking["rank_mean"] = np.arange(1, len(ranking) + 1)
    order_min = ranking.sort_values(
        ["min_jm", "mean_jm"], ascending=[False, False],
        kind="mergesort", na_position="last",
    ).index
    rank_min = np.empty(len(ranking), dtype=np.int64)
    rank_min[order_min] = np.arange(1, len(ranking) + 1)
    ranking["rank_min"] = rank_min
    ranking = ranking[list(RANKING_COLUMNS)]

    return ExhaustiveResult(
        ranking=ranking,
        bands=bands,
        labels=list(labels),
        class_names=class_names,
        min_k=min_k,
        max_k=max_k,
        n_combinations=int(total),
        elapsed_s=float(elapsed),
        kept_index=kept_index,
        n_rows_dropped=n_rows_dropped,
    )


# ─── Self-test entry point ────────────────────────────────────────────────────

def _selftest() -> None:
    """Run the search on the bundled demo CSV and print the best-per-k tables."""
    from pathlib import Path

    csv_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "examples" / "spectral_samples.csv"
    )
    if not csv_path.exists():
        print(f"[exhaustive self-test] demo CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    bands = ["Blue", "Green", "Red", "RedEdge", "NIR", "nDSM_m", "Thermal_C"]
    names = df.groupby("class_id")["class_name"].first().to_dict()

    res = exhaustive_search(df, "class_id", bands, class_labels=names)
    print("=" * 72)
    print(f"[exhaustive self-test] {res.n_combinations} combinations "
          f"(k = {res.min_k}..{res.max_k}) in {res.elapsed_s:.2f} s "
          f"→ {res.elapsed_s / res.n_combinations * 1e6:.0f} µs / combination")
    print("=" * 72)
    pd.set_option("display.width", 160)
    cols = ["k", "bands", "mean_jm", "min_jm", "worst_pair", "count_excellent"]
    print("\nBest per k — by mean JM")
    print(res.best_per_k("mean")[cols].to_string(index=False, float_format="%.3f"))
    print("\nBest per k — by min JM (worst pair)")
    print(res.best_per_k("min")[cols].to_string(index=False, float_format="%.3f"))

    # Regression against canonical thesis subsets
    canon = {
        "Blue + Green + Red": 1.288,
        "Blue + Green + Red + RedEdge + NIR": 1.551,
        "Blue + Green + Red + RedEdge + NIR + nDSM_m + Thermal_C": 1.838,
    }
    print("\nCanonical subsets")
    ok_all = True
    for s, exp in canon.items():
        got = float(res.ranking.loc[res.ranking["bands"] == s, "mean_jm"].iloc[0])
        ok = abs(got - exp) < 0.01
        ok_all &= ok
        print(f"  {s:<58s} {got:6.3f}  (expected {exp:.3f}) {'✅' if ok else '❌'}")
    print("\n✅ all canonical values match" if ok_all else "\n❌ mismatch — investigate")


if __name__ == "__main__":
    _selftest()
