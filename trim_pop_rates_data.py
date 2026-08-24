"""
trim_pop_rates_data.py — drop late-pulse conditions across every mouse, then cut
M151020's richer session down to the same per-type sub-condition counts as the
others, so a downstream step sees identical conditions in every file.

Two operations, in order:

1. Late-pulse filter.  A condition's first pulse is at t=0 and its second pulse
   (paired only) lands at `ipi_ms` after t=0, so a condition "has a pulse more
   than CUTOFF_MS after 0" exactly when `ipi_ms > CUTOFF_MS`.  We drop those.
   The three reference sessions carry 12 such conditions each (ipi 590/800/1000
   ms across the 4 paired types), so each falls 52 -> 40.  M151020 has none
   (max ipi 200 ms), so the filter alone leaves it untouched.

2. Count-match.  M151020_ICTP1 sweeps more pulse durations / ITIs, so after the
   filter it still holds many more sub-conditions than the others.  We keep only
   the first N sub-conditions of each type, where N is that type's *post-filter*
   count in the reference session (COMPARE) — bringing M151020 to 40 as well.

Layout of each npz (see regenerate_population_rates.py):
    {type}__responses  : float64 (n_cond, n_folds, 2, n_bins)
    {type}__time_axis  : (n_bins,)
    {type}__conditions : 0-d str array holding json.dumps(list-of-dicts), len n_cond
Plus provenance scalars: ei_cache_key, n_folds, fold_seed.

Note the `__conditions` field is a *single* 0-d JSON string, not an array of
n_cond entries — so it has to be json-decoded, sliced as a list, then re-encoded.
Slicing the 0-d array directly (the original bug) does nothing.
"""
import json
from pathlib import Path

import numpy as np

all_conditions = ["single_E",
                  "single_I",
                  "paired_EE",
                  "paired_II",
                  "paired_EI",
                  "paired_IE"]

RESULTS = Path("results")
ANIMALS = ["M150605_ICTP1", "M150609_ICTP1", "M150609_ICTP2", "M151020_ICTP1"]
COMPARE = "M150609_ICTP1"  # reference session that fixes the per-type keep counts
CUTOFF_MS = 350            # drop conditions whose second pulse lands after this


def _keep_mask(conds):
    """Boolean mask, True where a condition has no pulse beyond CUTOFF_MS.

    Pulse 1 is at t=0; the second pulse (paired only) is at ipi_ms, so the only
    way a pulse lands past the cutoff is ipi_ms > CUTOFF_MS. `== CUTOFF_MS` (the
    350 ms ITI) is kept — it is not *more than* the cutoff.
    """
    return np.array([c["ipi_ms"] <= CUTOFF_MS for c in conds], dtype=bool)


def _filter_type(data, c):
    """Return (responses, conditions-list) for type `c` with late pulses removed."""
    conds = json.loads(str(data[f"{c}__conditions"]))
    mask = _keep_mask(conds)
    kept = [cc for cc, m in zip(conds, mask) if m]
    return data[f"{c}__responses"][mask], kept


# Pass 1: filter the reference session and record its post-filter per-type counts.
compare = np.load(RESULTS / f"population_rates_{COMPARE}_s1.npz", allow_pickle=True)
keep_counts = {c: len(_filter_type(compare, c)[1]) for c in all_conditions}
print(f"reference {COMPARE} post-filter counts: {keep_counts} "
      f"(total {sum(keep_counts.values())})")

# Pass 2: apply the filter to every mouse, count-match the richer sessions, write.
for animal in ANIMALS:
    src = np.load(RESULTS / f"population_rates_{animal}_s1.npz", allow_pickle=True)
    new_data = {}
    n_conds = 0
    print(f"=== {animal} ===")
    for c in all_conditions:
        resp, kept = _filter_type(src, c)
        n_keep = keep_counts[c]
        if len(kept) > n_keep:
            # Richer session (M151020) — trim to the reference count.
            resp, kept = resp[:n_keep], kept[:n_keep]
        elif len(kept) < n_keep:
            # Don't invent conditions — keep what we have and flag it.
            print(f"  Warning: {c} has {len(kept)} < reference {n_keep}; keeping all.")
        print(f"  {c:12s} keep {len(kept)}")

        new_data[f"{c}__responses"] = resp
        new_data[f"{c}__time_axis"] = src[f"{c}__time_axis"]
        new_data[f"{c}__conditions"] = np.array(json.dumps(kept))
        n_conds += len(kept)

    # Carry provenance through unchanged so each trimmed file still records how
    # its E/I labels and folds were produced.
    for k in ("ei_cache_key", "n_folds", "fold_seed"):
        new_data[k] = src[k]

    out_path = RESULTS / f"population_rates_{animal}_s1_trimmed.npz"
    np.savez(out_path, **new_data)
    print(f"  total sub-conditions: {n_conds}  ->  {out_path.name}")
