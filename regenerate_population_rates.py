"""
Regenerate the saved `results/population_rates_<animal>_s1.npz` files from
`data_loader.get_population_responses` (now 1 ms default bin).

Flat npz layout expected by `figures.plot_population_rates`:
    {type}__responses  : float64 (n_cond, 2, n_bins)
    {type}__time_axis  : (n_bins,)
    {type}__conditions : 0-d str array holding json.dumps(list-of-dicts)
"""
import json
from pathlib import Path

import numpy as np

import data_loader

ROOT = Path("/mnt/scratch/IChunData4Dabin")
ANIMALS = ["M150605_ICTP1", "M150609_ICTP1"]
OUT_DIR = Path(__file__).parent / "results"


def regenerate(animal_id: str) -> None:
    base_dir = ROOT / animal_id
    out = data_loader.get_population_responses(base_dir=base_dir, animal_id=animal_id)
    flat: dict[str, np.ndarray] = {}
    for exp_type, res in out.items():
        flat[f"{exp_type}__responses"] = res["responses"]
        flat[f"{exp_type}__time_axis"] = res["time_axis"]
        flat[f"{exp_type}__conditions"] = np.array(json.dumps(res["conditions"]))
    out_path = OUT_DIR / f"population_rates_{animal_id}_s1.npz"
    np.savez(out_path, **flat)
    n_bins = next(iter(out.values()))["time_axis"].shape[0]
    print(f"wrote {out_path}  ({len(out)} types, n_bins={n_bins})", flush=True)


if __name__ == "__main__":
    for a in ANIMALS:
        print(f"=== {a} ===", flush=True)
        regenerate(a)
