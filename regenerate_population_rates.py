"""
Regenerate the saved `results/population_rates_<animal>_s1.npz` files from
`data_loader.get_population_responses` (unsmoothed by default — set
`HAMMING_MS = 40.0` for the paper's Hamming-smoothed PSTHs).  Bin width is
`BIN_SAMPLES` sampling intervals of the 30 kHz recording clock.

Output: `results/[smoothed_]b<BIN_SAMPLES>_population_rates_<animal>_s1.npz`.

Flat npz layout expected by `figures.plot_population_rates`:
    {type}__responses  : float64 (n_cond, n_folds, 2, n_bins)
    {type}__time_axis  : (n_bins,)
    {type}__conditions : 0-d str array holding json.dumps(list-of-dicts)
Plus provenance scalars: ei_cache_key, n_folds, fold_seed, bin_samples.
"""
import json
from pathlib import Path

import numpy as np

import data_loader

ROOT = Path("/mnt/scratch")
ANIMALS = ["M150605_ICTP1", "M150609_ICTP1", "M150609_ICTP2", "M151020_ICTP1", "M150823_ICTP2"]
OUT_DIR = Path(__file__).parent / "results"
N_FOLDS = 3       # paper's k for the mice we parse (see README exclusions)
FOLD_SEED = 0     # fixes the random trial->fold split for reproducibility
# Store raw trial-averaged rates by default; a symmetric Hamming window smears the
# evoked transient ~half its width before the pulse (see the onset-alignment check),
# so we keep the unsmoothed rates and leave any smoothing to downstream consumers.
# Set to 40.0 to reproduce the paper's 40 ms Hamming-smoothed PSTHs.
HAMMING_MS = 0.0
# Bin width in sampling intervals of the recording clock
# (data_loader.SAMPLE_RATE_HZ = 30 kHz, i.e. 1/30 ms per sample).
# Must divide the 0.5 s pre-onset window (15000 samples) so onset is a bin edge.
SAMPLING_FREQ_HZ = 30000  # 30 kHz recording clock
BIN_SAMPLES = 300


def regenerate(animal_id: str, hamming_ms: float = HAMMING_MS,
               bin_samples: int = BIN_SAMPLES) -> None:
    base_dir = ROOT / animal_id
    out = data_loader.get_population_responses(
        base_dir=base_dir, animal_id=animal_id,
        n_folds=N_FOLDS, fold_seed=FOLD_SEED, hamming_ms=hamming_ms,
        bin_samples=bin_samples,
    )
    if not out:
        print(f"SKIP {animal_id}: no pulse experiments / no responses — "
              f"nothing written.", flush=True)
        return
    flat: dict[str, np.ndarray] = {}
    for exp_type, res in out.items():
        flat[f"{exp_type}__responses"] = res["responses"]
        flat[f"{exp_type}__time_axis"] = res["time_axis"]
        flat[f"{exp_type}__conditions"] = np.array(json.dumps(res["conditions"]))
    # Provenance: the exact classified-cache key this run used (session + boxes +
    # classification version), so the file records how its E/I labels were produced.
    flat["ei_cache_key"] = np.array(data_loader.ei_cache_key(base_dir, animal_id=animal_id))
    # Fold provenance: n_folds + the seed that fixed the random trial split.
    flat["n_folds"] = np.array(N_FOLDS)
    flat["fold_seed"] = np.array(FOLD_SEED)
    flat["bin_samples"] = np.array(bin_samples)
    flat["sampling_freq_hz"] = np.array(SAMPLING_FREQ_HZ)
    if hamming_ms > 0.0:
        # out_path = OUT_DIR / f"h{hamming_ms:.0f}_smoothed_population_rates_{animal_id}_s1.npz"
        out_path = OUT_DIR / f"smoothed_b{bin_samples}_population_rates_{animal_id}_s1.npz"
    elif hamming_ms == 0.0 and bin_samples > 1:
        out_path = OUT_DIR / f"b{bin_samples}_population_rates_{animal_id}_s1.npz"
    np.savez(out_path, **flat)
    n_bins = next(iter(out.values()))["time_axis"].shape[0]
    print(f"wrote {out_path}  ({len(out)} types, n_bins={n_bins}, "
          f"bin_samples={bin_samples}, n_folds={N_FOLDS})", flush=True)


if __name__ == "__main__":
    for a in ANIMALS:
        print(f"=== {a} ===", flush=True)
        regenerate(a)
