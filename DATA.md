# Structure of trial_counts npz files

Single-trial E/I population spike counts, one file per (mouse, session).
Produced by `regenerate_population_rates.py` from `data_loader.get_trial_counts`.
Live in `results/`. Worked examples: `tutorial.ipynb`.

No trial averaging, smoothing, baseline normalisation or CV folding — those are
downstream choices (helpers in `trial_analysis.py`). Time is only binned: each bin
holds the number of spikes in it (a sum, not an average). (The older `*population_rates_*.npz` files hold per-fold
trial averages; their format is described in git history of this file.)

## Naming convention
- `trial_counts_b<N>_<animal_id>_s<session>.npz`, e.g. `trial_counts_b30_M150605_ICTP1_s1.npz`.
- `<animal_id>` = mouse + protocol, e.g. `M150605_ICTP1`, `M150609_ICTP2`, `M151020_ICTP1`.
- `b<N>` = bin width in 30 kHz sampling intervals (`b30` = 1 ms, the default; `b300` = 10 ms).
- Saved with `np.savez_compressed` (1 ms counts are mostly zeros); `np.load` reads it as usual.
- Session is always `s1` so far.

## Design: a flat trial table
Conditions have very different repeat counts (10 – 308 trials per condition, and
M151020 has a median of 10 but a max of 184), so a `(n_cond, n_repeats, 2, n_bins)`
cube would be mostly NaN padding. Instead the file is two *tables* — sets of
parallel arrays sharing axis 0 — linked by an index:

- **Trial table** (axis 0 = trial, in recording order): `counts` and per-trial labels.
- **Condition table** (axis 0 = condition): `cond_*` arrays.
- `cond_idx[i]` = the condition row of trial `i` (like a database foreign key).

All experiment types live in one table; the type is a column (`cond_exp_type`), not a
key prefix. Every file has the same keys, whatever the mouse.

`data_loader.get_trial_counts` returns a plain `dict[str, np.ndarray]` whose keys are
exactly the npz keys, so `np.savez(path, **out)` and `dict(np.load(path))` round-trip.
Strings are stored as NumPy unicode arrays, so `allow_pickle` is not needed.

Counts are stored rather than rates because the paper baseline-normalises the
*trial-averaged* PSTH: normalising single trials would divide by near-zero
baselines, and a mean of ratios is not a ratio of means.

## Fields

### Trial table — axis 0 = `n_trials`
| key | dtype, shape | meaning |
|---|---|---|
| `counts` | uint16 `(n_trials, 2, n_bins)` | spike counts per bin; axis 1: `0 = E` (wide), `1 = I` (narrow) |
| `cond_idx` | int64 `(n_trials,)` | row of the condition table |
| `exp_num` | int64 `(n_trials,)` | experiment number within the session |
| `trial_in_exp` | int64 `(n_trials,)` | index among that experiment's matched laser trials |
| `onset_sample` | int64 `(n_trials,)` | first-pulse onset, experiment-local 30 kHz sample |

### Condition table — axis 0 = `n_cond`
| key | dtype | meaning |
|---|---|---|
| `cond_exp_type` | str | `single_E`, `single_I`, `paired_EE`, `paired_II`, `paired_EI`, `paired_IE` |
| `cond_pulse_type` | int64 | Protocol `pulseType` code (1 = BB, 2 = GG, 3 = BG, 4 = GB) |
| `cond_ipi_ms` | int64 | Protocol `intT`: gap from the **end** of pulse 1 to the **start** of pulse 2; 0 for single (README "Protocol parameters") |
| `cond_onset_ipi_ms` | float64 | onset-to-onset interval = `intT` + `dur1`; 0 for single |
| `cond_dur1_ms` | float64 | first-pulse duration |
| `cond_dur2_ms` | float64 | second-pulse duration; NaN for single |
| `cond_first_pop` | str | `"E"`/`"I"` driven by pulse 1 |
| `cond_second_pop` | str | `"E"`/`"I"` driven by pulse 2; `""` for single |
| `cond_contrast` | int64 | RandNoise screen-stimulus contrast `c`: 0 = laser only (also 0 for protocols without `c`), 100 = laser during a screen stimulus (README "RandNoise") |
| `cond_tp_ms` | float64 | `Tp`: laser onset after the trial's Timeline StimStart; NaN for protocols without it |
| `cond_seed` | float64 | screen-stimulus seed `seedw`; NaN for protocols without it |

A condition is one distinct combination of all these columns, so the same laser-only
stimulus can appear as several rows (e.g. `Tp` 500 vs NaN from different protocols) —
pool with `trial_analysis.select`, which ignores columns you don't name. **Select
`contrast=0` for the paper's blank-screen paradigm.** Sorted by experiment type (order
above), then the other columns. Only conditions with at least one kept trial appear;
the exact set is per-mouse.

### Scalars (0-d) and axes
| key | meaning |
|---|---|
| `time_axis` | float64 `(n_bins,)`, bin centres in s relative to onset |
| `bin_samples` | bin width in 30 kHz sampling intervals |
| `sampling_freq_hz` | 30 000 |
| `pre_s`, `post_s` | trial window `[-pre_s, +post_s)` |
| `animal_id`, `session` | which session |
| `ei_cache_key` | classified-spike cache key (session + E/I boxes + classification version) that produced the E/I labels |
| `n_trials_dropped` | trials excluded because their window crossed an experiment boundary |
| `n_trials_no_ttl` | Timeline laser trials with no laser TTL (excluded) |
| `n_trials_unlabelled` | laser TTL trials with no Timeline trial (excluded) |
| `n_trials_failed_check` | trials whose TTL pulses disagreed with their Protocol label (excluded) |

## How trials are found (sources of truth)
- **When / which laser fired:** rising edges on the `.ns5` laser TTL channels (ain131 =
  445 nm, ain132 = 561 nm), same clock as the spikes. Pulses are grouped into trials by
  their gaps: every gap must be ≤ the protocol's longest within-trial interval
  (max `intT + dur1`) or ≥ 1.5 s (paper: 1.6–5 s between trials), otherwise loading
  **raises**. `trial_in_exp` / `onset_sample` refer to these groups.
- **What the trial was:** Timeline `StimStart` order + the full Protocol condition. A
  trial's laser is predicted at `StimStart + Tp`; groups are matched one-to-one to
  predictions within 0.25 s after fitting the clock offset/rate.
- **Per-trial check:** pulse count, laser order, widths (±0.15 ms) and onset-to-onset
  interval (`intT + dur1`, ±0.2 ms + 0.1 %) must agree with the label. More than 5 %
  unlabelled or failing trials in an experiment **raises**.

## Trial window and dropped trials
Every trial uses one fixed window `[-pre_s, +post_s)` (set in
`regenerate_population_rates.py`; default −0.5 to +1.5 s). Onset (0 s) is always a
bin edge; a trailing partial bin is dropped. A trial whose window runs past its
experiment's span in the concatenated recording is **dropped**, not zero-filled: the
spikes there belong to the neighbouring experiment. A window *can* contain the next
trial's pulse if the inter-trial interval is shorter than `post_s`.

## Bin width
Spikes are timestamped on the **30 kHz** recording clock as integer sample indices,
so bin width is an integer number of 1/30 ms sampling intervals (`bin_samples`) and
binning is exact. `bin_samples` must divide `pre_s × 30 000`, or the loader raises.
Default 1 ms. Coarser bins are an exact sum of finer ones (`trial_analysis.rebin`), so
the files keep the fine resolution and analyses rebin as needed. At 10 ms, paired IPIs
of 5 or 8 ms would put both pulses in one bin.

## How to pull in data
```python
import numpy as np

d = dict(np.load("results/trial_counts_b30_M150605_ICTP1_s1.npz"))

# all paired_EE trials
cond_rows = np.flatnonzero(d["cond_exp_type"] == "paired_EE")
x = d["counts"][np.isin(d["cond_idx"], cond_rows)]        # (n, 2, n_bins)

# trial-averaged, baseline-normalised E/I rate for one condition (the paper's PSTH)
k = cond_rows[0]
bin_s = d["bin_samples"] / d["sampling_freq_hz"]
rate = d["counts"][d["cond_idx"] == k].mean(axis=0) / bin_s          # (2, n_bins) Hz
bl = (d["time_axis"] >= -0.5) & (d["time_axis"] < -0.1)
psth = rate / rate[:, bl].mean(axis=1, keepdims=True)
```
