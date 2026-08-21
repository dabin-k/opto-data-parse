"""
Regenerate the saved `results/population_rates_<animal>_s1.npz` files from
`data_loader.get_population_responses` (now 1 ms default bin).

Flat npz layout expected by `figures.plot_population_rates`:
    {type}__responses  : float64 (n_cond, n_folds, 2, n_bins)
    {type}__time_axis  : (n_bins,)
    {type}__conditions : 0-d str array holding json.dumps(list-of-dicts)
Plus provenance scalars: ei_cache_key, n_folds, fold_seed.
"""
import json
from pathlib import Path

import numpy as np

import data_loader

ROOT = Path("/mnt/scratch/IChunData4Dabin")
ANIMALS = ["M150605_ICTP1", "M150609_ICTP1", "M150609_ICTP2", "M151020_ICTP1"]
OUT_DIR = Path(__file__).parent / "results"
N_FOLDS = 3       # paper's k for the mice we parse (see README exclusions)
FOLD_SEED = 0     # fixes the random trial->fold split for reproducibility


def regenerate(animal_id: str) -> None:
    base_dir = ROOT / animal_id
    out = data_loader.get_population_responses(
        base_dir=base_dir, animal_id=animal_id,
        n_folds=N_FOLDS, fold_seed=FOLD_SEED,
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
    out_path = OUT_DIR / f"population_rates_{animal_id}_s1.npz"
    np.savez(out_path, **flat)
    n_bins = next(iter(out.values()))["time_axis"].shape[0]
    print(f"wrote {out_path}  ({len(out)} types, n_bins={n_bins}, "
          f"n_folds={N_FOLDS})", flush=True)


if __name__ == "__main__":
    for a in ANIMALS:
        print(f"=== {a} ===", flush=True)
        regenerate(a)
