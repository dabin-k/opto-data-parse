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
- `2, 3, 6, 7` — mixed single + paired-pulse opto (`stim2PulsesRandNoise`, 96 conditions:
  paired *and* `intT == 0` single pulses, some during a screen stimulus — see "RandNoise" below)
- `4, 5, 13` — single-pulse opto only (`stim2Pulses2Waves`)
- `8, 10–12` — regular periodic pulses (different paradigm)
- `DEFAULT_PULSE_EXPS = [2,3,4,5,6,7,13]` — the E/I analysis subset.
- Pulse-type map: `1=BB(EE)`, `2=GG(II)`, `3=BG(EI)`, `4=GB(IE)`; `intT==0` ⇒ single.
- Experiment numbers are the mpep run order within series 1: dir `1/<exp>/`, files
  `<animal>_1_<exp>.*`. The manifest's `ns5files2unify` / `SELECTED_EXPERIMENTS` give the
  order they were concatenated into the `.kwik` (with start times).

## Protocol parameters (what `Protocol.mat` `pars` mean)

| param | meaning | confidence |
|---|---|---|
| `pulseType` | 1 = BB, 2 = GG, 3 = BG, 4 = GB (laser order) | verified against TTL channels |
| `durT` / `durT1`, `durT2` | pulse duration(s), stored as ms × 10 | verified: TTL widths match (1–10 ms) |
| `intT` | **gap from the END of pulse 1 to the START of pulse 2**, ms; 0 = single pulse | verified, see below |
| `Vamp` | laser drive; 0 = no laser (blank trial, no TTL) | verified: no TTL for `Vamp = 0` |
| `Tp` | laser onset relative to the trial's Timeline `StimStart`, ms | verified: onset − (StimStart + Tp) is constant to ±65 ms (M150605 exp 2); matching on it labels 2103/2104 M150605 trials with 0 pulse-check failures |
| `c` | screen-stimulus contrast (0 or 100) | conjecture, see RandNoise |
| `x1 x2 y1 y2 sqsz nfr ncs seedw` | screen-stimulus geometry / squares / frames / seed | conjecture, see RandNoise |

**`intT` is not onset-to-onset.** Measured on the laser TTLs, onset-to-onset = `intT` + `dur1`
(plus ~0.05 % stimulus-PC vs Blackrock clock drift). Decisive case, M151020 exp 8: `intT` = 35 ms
gives 37.0 ms apart with a 2 ms first pulse and 45.0 ms with a 10 ms one; at `intT` = 70 the
excess equals `dur1` for all of 1, 2, 3, 4, 8, 10 ms (±0.07 ms). The paper's IPIs (Table S3) are
these `intT` numbers; for its 1–2 ms pulses the two definitions differ by 1–2 ms.

### RandNoise — laser pulses during a screen stimulus (finding, 2026-10-05)

The `stim2Pulses*RandNoise` protocols interleave, within one experiment:
- `c = 0` conditions — laser only (screen presumably blank), `Tp = 500`.
- `c = 100` conditions — laser fired `Tp` (400/500/900/1300) ms into a screen stimulus.
- `Vamp = 0` conditions — the screen stimulus with no laser.

**What the screen stimulus is, is a CONJECTURE.** The parameters (screen region `x1..y2`, square
size `sqsz`, frame count `nfr`, random seed `seedw`, contrast `c`) and the protocol name suggest a
random-checkerboard ("noise") movie. Nothing in the data or the paper documents it. The paper
(S1.5) says optogenetic stimulation was delivered "while the mouse was facing a blank gray
screen"; its only visual stimuli are LED flashes in *wild-type* mice, in separate sessions. So the
`c = 100` trials were most likely **not** part of the paper's analysis.

Share of laser trials with `c > 0`: 42 % in M150605_ICTP1 / M150609_ICTP1 / M150609_ICTP2,
20 % in M150823_ICTP2, 6 % in M151020_ICTP1. The loader keys conditions on `c`, `Tp` and
`seedw` as well, so they are never pooled with laser-only trials (filter on `cond_contrast`).
Supporting evidence that they differ: in M150605 paired_EE, `c = 100` conditions show extra
sharp E/I peaks at ~0.17 s and ~0.35 s after the laser that laser-only conditions lack.

### IPI-1000 trials were aligned to the second pulse (fixed 2026-10-05)
Pulses used to be split into trials at any gap > 1.0 s. A 1000 ms pair is 1001.5 ms
onset-to-onset, so it was cut in two, and the Timeline label landed on the *second* pulse.
The loader now groups by the protocol's own longest within-trial interval, raises on
ambiguous gaps, matches on `StimStart + Tp`, and checks every trial's pulses against its label
(`data_loader._experiment_trials`). Older outputs (`*population_rates_*.npz`, plots) predate
this and also pool `c = 100` trials.

## Code

- `data_loader.py` — public API:
  - `get_trial_counts(...)` → **single-trial** E/I population spike counts for one session as a flat trial table: `counts (n_trials, 2, n_bins)` + per-trial and per-condition arrays linked by `cond_idx`. No averaging, smoothing, normalisation or folds — those are downstream. Saved by `regenerate_population_rates.py`; layout in `DATA.md`, worked examples in `tutorial.ipynb`.
  - `get_trial_margins(...)`, `window_trial_loss(...)` → help choose the trial window (time available around each onset; trials a window would drop).
  - `load_data(...)` → `(trials, units, bins)` spike counts + stimulus + trial/unit info.
  - `get_population_responses(...)` → *legacy* per-fold trial-averaged E/I PSTHs `(n_cond, n_folds, 2, n_bins)`; still read by `figures.plot_population_rates` and the old `*population_rates_*.npz` files.
- `trial_analysis.py` — downstream helpers for the trial files: `load`, `select`, `rebin` (exact count sums), `psth` (trial-average → optional Hamming → baseline-normalise), `fold_labels` (seeded k-fold; defaults reproduce the legacy folds).
- `plot_population_rates.py` — QC figures per mouse from the trial files (folds thin, all-trials bold), written to `results/population_rates_plots/<mouse>_TRIALS/`.
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
- ~~20 % of single_E trials in exp 5 fired no analog TTL~~ — resolved: those are Protocol `Vamp = 0` blank (no-laser) trials.
- **Clusterless denoising** (paper's LSH over PCA features) is not reimplemented; we use raw per-spike waveform width.
- **Probe depth ordering** not applied — `figures.py` shows sites in acquisition order, not shank depth.
- First ~20 ms after a strong E pulse: detection underestimates activity (overlapping spikes / field). Treat with suspicion — paper caveat.
- `scratch.ipynb` / `fig1c_rasters.png` predate the timing fix and should be regenerated.
