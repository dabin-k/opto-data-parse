"""
Regenerate the saved single-trial files `results/trial_counts_b<BIN_SAMPLES>_<animal>_s<SESSION>.npz`
from `data_loader.get_trial_counts`.

Each file is one session's flat trial table (layout in DATA.md): raw E/I
population spike counts per trial, `counts (n_trials, 2, n_bins)`, plus
per-trial and per-condition arrays linked by `cond_idx`.  No averaging,
smoothing, baseline normalisation or cross-validation folding happens here —
those are downstream choices (see tutorial.ipynb).

Every hyperparameter that shapes the output is set below; the values are also
stamped into each file.
"""
from pathlib import Path

import numpy as np

import data_loader

ROOT = Path("/mnt/scratch")
ANIMALS = ["M150605_ICTP1", "M150609_ICTP1", "M150609_ICTP2", "M151020_ICTP1", "M150823_ICTP2"]
SESSION = 1
OUT_DIR = Path(__file__).parent / "results"

# Trial window around the first pulse's onset, in seconds.  One fixed window for
# every trial; trials whose window crosses an experiment boundary are dropped
# (count stamped as `n_trials_dropped`).
PRE_S = 0.5
POST_S = 1.5
# Bin width in sampling intervals of the 30 kHz recording clock
# (data_loader.SAMPLE_RATE_HZ), so 300 = 10 ms.  Must divide PRE_S * 30 000 so
# that onset is a bin edge.
BIN_SAMPLES = 300
# Neighbour pool for the clusterless E/I split: 2 = good clusters only (see
# data_loader.get_population_responses).  Stamped via `ei_cache_key`.
MIN_CLUSTER_GROUP = 2


def regenerate(animal_id: str) -> None:
    out = data_loader.get_trial_counts(
        base_dir=ROOT / animal_id, session=SESSION, animal_id=animal_id,
        pre_s=PRE_S, post_s=POST_S, bin_samples=BIN_SAMPLES,
        min_cluster_group=MIN_CLUSTER_GROUP,
    )
    if not out:
        print(f"SKIP {animal_id}: no laser trials — nothing written.", flush=True)
        return
    out_path = OUT_DIR / f"trial_counts_b{BIN_SAMPLES}_{animal_id}_s{SESSION}.npz"
    np.savez(out_path, **out)
    print(f"wrote {out_path}  ({out['counts'].shape[0]} trials, "
          f"{out['cond_exp_type'].size} conditions, "
          f"{int(out['n_trials_dropped'])} dropped, n_bins={out['time_axis'].size})",
          flush=True)


if __name__ == "__main__":
    for a in ANIMALS:
        print(f"=== {a} ===", flush=True)
        regenerate(a)
