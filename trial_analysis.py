"""
trial_analysis.py — downstream helpers for the single-trial `trial_counts_*.npz`
files (layout in DATA.md).

The files hold raw spike counts per trial; everything that reduces them (trial
averaging, rebinning, smoothing, baseline normalisation, CV folds) lives here, so
each analysis states its own choices.  Used by `tutorial.ipynb` and
`plot_population_rates.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
BASELINE_S = (-0.5, -0.1)   # paper's normalisation window


def load(animal_id: str, bin_samples: int = 30, session: int = 1) -> dict[str, np.ndarray]:
    """One session's trial table, as written by `regenerate_population_rates.py`."""
    return dict(np.load(RESULTS / f"trial_counts_b{bin_samples}_{animal_id}_s{session}.npz"))


def bin_s(d: dict) -> float:
    return float(d["bin_samples"] / d["sampling_freq_hz"])


def select(d: dict, **criteria) -> np.ndarray:
    """Trial mask for conditions matching every criterion, e.g.
    `select(d, exp_type="paired_EE", ipi_ms=50, contrast=0)` (keys are `cond_*`
    without the prefix).  Unnamed columns are pooled over.  NaN never matches,
    so select on `tp_ms`/`seed` only with real values."""
    rows = np.ones(d["cond_exp_type"].size, bool)
    for col, val in criteria.items():
        rows &= d[f"cond_{col}"] == val
    return np.isin(d["cond_idx"], np.flatnonzero(rows))


def rebin(counts: np.ndarray, time_axis: np.ndarray,
          factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum `factor` adjacent bins (exact for counts); a trailing remainder is dropped."""
    n = counts.shape[-1] // factor * factor
    c = counts[..., :n].reshape(*counts.shape[:-1], -1, factor).sum(-1)
    t = time_axis[:n].reshape(-1, factor).mean(-1)
    return c, t


def psth(counts: np.ndarray, time_axis: np.ndarray, bin_width_s: float,
         baseline: tuple[float, float] = BASELINE_S,
         hamming_ms: float = 0.0) -> np.ndarray:
    """
    Trial-averaged E/I rate (2, n_bins), optionally Hamming-smoothed, divided by
    its own mean over `baseline` — the paper's population activity (40 ms Hamming).

    Averages *before* normalising: single-trial baselines hold few spikes, and a
    mean of ratios is not the paper's ratio of means.
    """
    rate = counts.mean(axis=0) / bin_width_s
    if hamming_ms > 0:
        w = max(1, round(hamming_ms / 1e3 / bin_width_s)) | 1   # odd -> symmetric
        k = np.hamming(w)
        rate = np.stack([np.convolve(r, k / k.sum(), mode="same") for r in rate])
    bl = (time_axis >= baseline[0]) & (time_axis < baseline[1])
    return rate / rate[:, bl].mean(axis=1, keepdims=True)


def fold_labels(cond_idx: np.ndarray, n_folds: int = 3, seed: int = 0) -> np.ndarray:
    """
    Fold id per trial: each condition's trials randomly split as evenly as possible.

    Same RNG procedure as the legacy loader.  It reproduced the legacy
    `population_rates` folds while conditions were keyed on (pulse type, intT,
    duration) only; with the full key (contrast, Tp, seed) the conditions differ,
    so the splits do too.
    """
    rng = np.random.default_rng(seed)
    fold = np.empty(cond_idx.size, int)
    for k in np.unique(cond_idx):
        idx = np.flatnonzero(cond_idx == k)
        for f, part in enumerate(np.array_split(rng.permutation(idx), n_folds)):
            fold[part] = f
    return fold
