# ichun_opto — how the data is structured

Reverse-engineering the raw data of Lin, Okun, Carandini & Harris (2020),
*"Equations governing dynamics of excitation and inhibition in the mouse
corticothalamic network"* (bioRxiv 2020.06.03.132688). Goal: understand how the
recordings were stored, and reproduce paper figures to prove we read them right.

Session in hand: **M150605_ICTP1**, series/session 1. One mouse exercised so far.

> **Session 1 only (current convention).** Every pipeline entry point defaults to
> `session=1`, all outputs are named `_s1`, and no session-2 data has been built —
> even where a dir has two (e.g. `M151020_ICTP1` has `/1` and `/2`; we cache `/1`).
> `session` is a parameter throughout, so this is a convention for now. Maybe we'll change it in the future. 

## The experiment (V1, awake mouse)

- Optogenetic pulses drive excitatory (E) or PV-inhibitory (I) populations.
- Single pulses and paired pulses in all four orders (EE, II, EI, IE).
- Extracellular 32-ch silicon probe, 30 kHz, Blackrock amplifier. Paired LGN recording (separate pipeline, not handled here).
- **This session: 445 nm blue → E, 561 nm green → I** (mouse-line dependent; don't assume).

## Dataset layout (external — `/mnt/scratch/.../M150605_ICTP1/`, NOT in repo)

Session dir `…/1/`:

| file | what it is |
|---|---|
| `*_s1_not9.kwik` | detected spikes: sample indices (30 kHz) + cluster labels. HDF5. |
| `*_s1_not9.kwx` | per-spike waveforms `(n_spikes, 20, 8)` + PCA features. HDF5. |
| `*_s1_not9.raw.kwd` | continuous concatenated voltage. HDF5. |
| `*_s1_not9.mat` / `*_s1_V1.mat` | **manifest** (preprocessing recipe), not results. |
| `*.dat`, `*_CAR.dat` | flat int16 continuous voltage (SpikeDetekt input). |

Per-experiment dirs `…/1/<exp>/`:

| file | what it is |
|---|---|
| `<animal>_1_<exp>.ns5` | raw 30 kHz continuous, **NEURALSG (NSx 2.1)** format. 36 ch = 32 neural + 4 analog. |
| `<animal>_1_<exp>.nev` | Blackrock events — here **only online spike events**, no digital input. |
| `1_<exp>_<animal>_Timeline.mat` | stimulus condition order + timing, in the **Timeline DAQ clock**. |
| `Protocol.mat` | intended condition parameters (`pulseType`, `intT`, `durT`). |

## Data model (how it fits together)

- **Spikes** live in `.kwik` as one **concatenated recording of 13 experiments** (exp 9 dropped → "not9"). Times are 30 kHz sample indices in that concatenated space.
- **Segment boundaries**: manifest `lims` = per-experiment sample lengths. `exp_start = cumsum([0, lims])` maps concatenated samples ↔ experiment. Verified: `lims` == `.ns5` lengths exactly.
- **Stimulus timing**: the laser TTL is recorded on the **`.ns5` analog inputs** (same clock as spikes), NOT in Timeline. See below — this is the key fact.
- **Condition labels**: `Timeline.mat` (order) + `Protocol.mat` (params). Timeline gives *what/order*, the `.ns5` gives *when*.
- **E/I split**: read off each spike's **waveform width** (trough-to-peak, from `.kwx`), not from clustering and not from which laser fired. Faithful V1 representation is `(trials, 2 populations, time)`. This approximates the paper's clusterless method (we skip its LSH denoising).

## Timing — the important bit

- Timeline and Blackrock are **two separate DAQs on different clocks**, started seconds apart. Timeline's `mpepUDP` StimStart times are in the wrong clock → using them to align spikes gives flat PSTHs (this was the original bug).
- Fix: laser pulses appear as **saturating TTL markers on `.ns5` analog inputs**:
  - `ain 129` = sawtooth ramp (Timeline sync) — not laser
  - `ain 130` = continuous square clock — not laser
  - **`ain 131` = E laser (445 nm), `ain 132` = I laser (561 nm)** — the pulse markers
- Detect rising edges on 131/132 → onsets in Blackrock samples → map to concatenated space with `exp_start`. No cross-clock conversion.
- Match each analog onset to a Timeline trial by **relative timing** (drift < ~50 ms) to attach its condition label.

## Experiment layout (session 1)

- `1, 14` — visual tuning (no laser, skipped by the loader)
- `2, 3, 6, 7` — paired-pulse opto (`stim2PulsesRandNoise`)
- `4, 5, 13` — single-pulse opto (`stim2Pulses2Waves`)
- `8, 10–12` — regular periodic pulses (different paradigm)
- `DEFAULT_PULSE_EXPS = [2,3,4,5,6,7,13]` — the E/I analysis subset.
- Pulse-type map: `1=BB(EE)`, `2=GG(II)`, `3=BG(EI)`, `4=GB(IE)`; `intT==0` ⇒ single.

## Code

- `data_loader.py` — public API:
  - `get_trial_counts(...)` → **single-trial** E/I population spike counts for one session as a flat trial table: `counts (n_trials, 2, n_bins)` + per-trial and per-condition arrays linked by `cond_idx`. No averaging, smoothing, normalisation or folds — those are downstream. Saved by `regenerate_population_rates.py`; layout in `DATA.md`, worked examples in `tutorial.ipynb`.
  - `get_trial_margins(...)`, `window_trial_loss(...)` → help choose the trial window (time available around each onset; trials a window would drop).
  - `load_data(...)` → `(trials, units, bins)` spike counts + stimulus + trial/unit info.
  - `get_population_responses(...)` → *legacy* per-fold trial-averaged E/I PSTHs `(n_cond, n_folds, 2, n_bins)`; still read by `figures.plot_population_rates` / `plot_population_rates.py` and the old `*population_rates_*.npz` files.
- `laser_timing.py` — parse `.ns5` (NEURALSG), auto-detect laser channels, return per-trial onsets.
- `figures.py` — `plot_raw_traces_around_pulse()` reproduces Fig S1.B.
- Run against miniconda **base** python. Needs the dataset mounted.

## Cache (local, gitignored — lives on scratch)

`data_loader.CACHE_DIR = <repo>/cache` holds the expensive derived caches
(`knn_*`, `wide_*`, `ei_*` npz/npy — several GB per mouse). The repo `cache/` is
**a symlink onto `/mnt/scratch`**, because the root fs (`/dev/sdd`, holds `/home`)
runs ~99 % full and a prior build hit ENOSPC there. The chain:

```
<repo>/cache  ->  /mnt/scratch/_ichun_opto_cache      (live caches)
```

Archived older caches (mcg0/mcg1 knn/wide, moved off root to free space) sit
alongside at `/mnt/scratch/_ichun_opto_cache_archive/`. `cache/`
is in `.gitignore`, so only the symlink would ever be seen by git (and it isn't
tracked). To relocate, repoint the symlink — code always resolves `CACHE_DIR`
through it, so nothing else changes.

## Cross-mouse (how much generalises)

Checked on 2 mice so far: **M150605A** and **M150609A** (both `M15060x_ICTP1`). Not yet verified on the rest.

**Never parse `M150909C`, `M150823B`, `M150303B`.** The legacy population-rates pipeline hard-coded 3-fold cross-validation (paper S1.10), but these three sessions used a different number of folds (paper p22), so they were excluded. *Open:* the single-trial files no longer bake in folds, so this reason no longer applies — revisit whether to include them.

Same across both (promising, unverified elsewhere):
- `.ns5` = 36 ch = 32 neural + 4 analog (ids 129–132), **NEURALSG** format.
- **Laser channels identical: `ain131` = E laser, `ain132` = I laser.**
- Manifest `not*` + `lims` scheme; `lims` == `.ns5` lengths.

Differs per mouse (must be handled, not hardcoded):
- Excluded experiment differs (`not9` vs `not1`) → manifest filename differs. Handled by matching the `_s{n}_*.mat` manifest that has a sibling `.kwik` (also covers the `_all`/`_sel` naming of M150609_ICTP2 / M151020).
- Experiment numbers / layout differ → **pulse-exp list is derived per session** from each `Protocol.mat` `xfile` (`_OPTO_PULSE_XFILES`), not hardcoded.
- Mouse line differs → **wavelength→E/I map comes from `mouse_lines.py`**, not a fixed B→E. (M150609 happens to match M150605; other lines differ, e.g. `PVcre;Ai32` = 445→I only.)
- **Pulse-duration encoding differs.** Most sessions use `stim2PulsesRandNoise.x` /
  `stim2Pulses2Waves.x` with a single `durT` (both pulses equal). M151020 uses
  **`stim2Pulses2DurRandNoise.x`**, which has `durT1`/`durT2` — per-pulse durations.
  **Assumption (undocumented in the data, taken as given):** this is the same E/I
  single/paired paradigm, only the two pulses of a *pair* may differ in duration.
  Loader treatment: single (`intT==0`) uses `durT1` only (durT2 is a vestigial
  default); paired keeps a scalar `dur_ms` when `durT1==durT2`, else a `(d1, d2)`
  tuple. Downstream (`conditions` json, plots) must accept `dur_ms` as scalar *or*
  2-tuple.

## Known issues / uncertainties

- **E/I waveform threshold** (0.4 ms) is unvalidated — current split is lopsided (~113k wide vs ~1.09M narrow). Needs the per-session bimodal-width check.
- **~20 % of single_E trials in exp 5 fired no analog TTL** (all 1 ms pulses); cause unknown. Loader matches by relative timing, so these trials are just dropped.
- **Clusterless denoising** (paper's LSH over PCA features) is not reimplemented; we use raw per-spike waveform width.
- **Probe depth ordering** not applied — `figures.py` shows sites in acquisition order, not shank depth.
- First ~20 ms after a strong E pulse: detection underestimates activity (overlapping spikes / field). Treat with suspicion — paper caveat.
- `scratch.ipynb` / `fig1c_rasters.png` predate the timing fix and should be regenerated.
