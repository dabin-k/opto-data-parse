# ichun_opto — how the data is structured

Reverse-engineering the raw data of Lin, Okun, Carandini & Harris (2020),
*"Equations governing dynamics of excitation and inhibition in the mouse
corticothalamic network"* (bioRxiv 2020.06.03.132688). Goal: understand how the
recordings were stored, and reproduce paper figures to prove we read them right.

Session in hand: **M150605_ICTP1**, series/session 1. One mouse exercised so far.

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
  - `load_data(...)` → `(trials, units, bins)` spike counts + stimulus + trial/unit info.
  - `get_population_responses(...)` → trial-averaged, Hamming-smoothed, baseline-normalised E/I PSTHs per condition.
- `laser_timing.py` — parse `.ns5` (NEURALSG), auto-detect laser channels, return per-trial onsets.
- `figures.py` — `plot_raw_traces_around_pulse()` reproduces Fig S1.B.
- Run against miniconda **base** python. Needs the dataset mounted.

## Known issues / uncertainties

- **E/I waveform threshold** (0.4 ms) is unvalidated — current split is lopsided (~113k wide vs ~1.09M narrow). Needs the per-session bimodal-width check.
- **~20 % of single_E trials in exp 5 fired no analog TTL** (all 1 ms pulses); cause unknown. Loader matches by relative timing, so these trials are just dropped.
- **Clusterless denoising** (paper's LSH over PCA features) is not reimplemented; we use raw per-spike waveform width.
- **Probe depth ordering** not applied — `figures.py` shows sites in acquisition order, not shank depth.
- First ~20 ms after a strong E pulse: detection underestimates activity (overlapping spikes / field). Treat with suspicion — paper caveat.
- `scratch.ipynb` / `fig1c_rasters.png` predate the timing fix and should be regenerated.
