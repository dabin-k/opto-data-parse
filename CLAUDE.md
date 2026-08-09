# CLAUDE.md — Working conventions for this repo

How Dabin and Claude collaborate on **ichun_opto**. Auto-loaded as context every session.
A new agent should read this + `M150605_ICTP1_data_file_conjectures.md` + the latest
`journal/YYYY-MM-DD.md` and be caught up.

## What this project is

Reverse-engineering the raw data of an **unpublished 2019/2020 computational-neuroscience
paper** so Dabin can understand *how the data was saved*. No original analysis code survives,
so we read the paper and infer backwards how the recordings were stored and processed.

**Paper:** Lin, Okun, Carandini & Harris (2020), *"Equations governing dynamics of excitation
and inhibition in the mouse corticothalamic network"*, bioRxiv `2020.06.03.132688`
(PDF: `lin_harris_mouse_corticothalamic_dynamical_equation.pdf`). Awake mouse V1 was
optogenetically stimulated with single and paired light pulses to drive excitatory (pyramidal)
and/or PV-expressing inhibitory populations; activity was recorded extracellularly, with paired
LGN recordings. **Which wavelength drives which population depends on the mouse line** — see the
data-model section below; do not assume "blue = E".

**Method of validation:** recreate figures from the paper (starting with Fig. 1C rasters) to
*confirm* the raw data has been parsed correctly. A reproduced figure that matches the paper is
the evidence that our reading of the file format is right. The deliverable is understanding, not
a pipeline.

**Final aim (where this is heading):** make the loader driven by **mouse ID and experiment type**
as first-class, easily-configured inputs, so any session in this dataset — not just the hardcoded
`M150605_ICTP1` — can be loaded and its figures reproduced. The current `data_loader.py` hardcodes
this one session (single `animal_id`, fixed wavelength→population map, `DEFAULT_PULSE_EXPS`, etc.).
**We are not refactoring it yet** — this is the direction, stated here so new work bends toward
generality (parameterise mouse/experiment rather than baking in M150605 assumptions) rather than
away from it.

## Where things live
| Concern | Location |
|---|---|
| Raw dataset (external, NOT in repo) | `/mnt/scratch/M150605_ICTP1/` |
| Session-level files | `/mnt/scratch/M150605_ICTP1/1/` (manifest `.mat`, `.kwik`, `.kwx`) |
| Per-experiment files | `/mnt/scratch/M150605_ICTP1/1/<exp>/` (`Protocol.mat`, `*_Timeline.mat`, `.ns5`, `.nev`) |
| Data-loading code | `data_loader.py` |
| Standing notes on file formats & interpretations | `M150605_ICTP1_data_file_conjectures.md` |
| Exploration / figure work | `scratch.ipynb` |
| Reproduced figures | `fig1c_rasters.png` (and future `figNx_*.png`) |
| The paper | `lin_harris_mouse_corticothalamic_dynamical_equation.pdf` |
| Daily journal | `journal/YYYY-MM-DD.md` *(convention to adopt; dir not yet created)* |

The single session in hand is **M150605_ICTP1**, series/session 1. Code defaults assume this
one animal; `animal_id` / `base_dir` are parameterised but only this session has been exercised.

## Environment
**Python env:** miniconda **base** (`/home/dabin/miniconda3/bin/python`, Python 3, NumPy 2.5.1).
No project virtualenv — run against base.
**Dependencies:** `numpy`, `scipy`, `h5py`, `matplotlib` (all present in base). There is **no
`requirements.txt`** yet — add one only if the user asks.

## How to run
The loader is a library, driven from `scratch.ipynb` or a REPL:
```python
import data_loader
# Cluster-level trial-aligned spike counts: (n_trials, n_units, n_bins)
responses, stimulus, time_axis, trial_info, unit_info = data_loader.load_data(bin_s=0.01)
# Trial-averaged, baseline-normalised E/I population PSTHs, grouped by experiment type
pop = data_loader.get_population_responses()
```
Requires `/mnt/scratch/M150605_ICTP1/` to be mounted. Reading `.ns5` continuous voltage is a
last resort — prefer the already-detected spikes in `.kwik`.

## Git workflow
- Working branch: **master** (repo has **no commits yet** — the first commit will establish it).
- Don't push to anything other than `master` without confirming first.
- Commits: small, focused, descriptive. Use HEREDOC for multi-line commit messages.
- Don't commit the dataset or large artifacts: `/mnt/scratch/...` is external; keep `__pycache__/`,
  `.ns5`/`.dat`/`.kwd` and other large binaries out. There is no `.gitignore` yet — add one before
  the first commit (at minimum `__pycache__/`).
- Commit or push only when the user asks.

## Conventions for this codebase

### The data model we've inferred (see conjectures doc for full reasoning)
- Spikes come from a **SpikeDetekt/KlustaKwik `.kwik`** (HDF5) file covering **13 concatenated
  experiments** (experiment 9 was interrupted and excluded — hence the `not9` filename). Spike
  times are **sample indices at 30 kHz**, in concatenated-recording space.
- The manifest `.mat` (`..._s1_not9.mat` / `..._s1_V1.mat`) is a **preprocessing manifest**, not
  a results file: `lims` (per-segment sample lengths, cumsum → segment boundaries),
  `SELECTED_EXPERIMENTS`, `SELECTED_CHANNELS`, `CHANNELS_ORDER`.
- Per-experiment `Timeline.mat` holds **measured** stimulus timing as `mpepUDP` `StimStart`/
  `StimEnd` event strings (experiment-local seconds). `Protocol.mat` holds **intended** condition
  parameters (`pulseType`, `intT` = interpulse interval, `durT` = duration×10). Align by event.
- Pulse-type map: `1=BB`, `2=GG`, `3=BG`, `4=GB` (`intT==0` ⇒ single pulse). **Blue↔E / Green↔I
  is mouse-line dependent** — see Table S1 of the paper:
  - `Thy18`: ChR2 @ **445 nm** → **E** (pyramidal); no I opsin.
  - `PVᶜʳᵉ;Ai32`: ChR2 @ **445 nm** → **I** (PV) — here blue drives inhibition, not excitation.
  - `PVᶜʳᵉ;Thy18 + C1V1`: ChR2 @ **445 nm** → **E**, C1V1 @ **561 nm** → **I**.
  - **This session (M150605A = `PVᶜʳᵉ;Thy18`+C1V1, per Table S2):** 445 nm (blue) → E,
    561 nm (green) → I. So `data_loader._PULSE_TYPE_MAP` (`B→E, G→I`) is correct *for M150605*,
    but it is a per-mouse-line assumption that must become configurable per the final aim above.
- **E/I classification is clusterless in the paper**: each detected spike is labelled wide
  (putative excitatory) vs narrow (fast-spiking inhibitory) by waveform trough-to-peak time, from
  `.kwx` filtered waveforms. Our loader approximates this per-spike (paper additionally denoised
  via locality-sensitive hashing over PCA features — not reimplemented). So the faithful V1
  representation is `(trials, 2 populations, time)`, **not** `(trials, n_cells, time)`.
- LGN used a *different* pipeline (KiloSort + Phy → pooled MUA); not the V1 E/I path.

### Experiment layout for session 1 (from `scratch.ipynb` cell 8)
- Exp 1, 14 — visual tuning checks (`oglTwoGratings` / `ogltuning`)
- Exps 2, 3, 6, 7 — main paired-pulse optogenetic (`stim2PulsesRandNoise`)
- Exps 4, 5, 13 — single-pulse optogenetic (`stim2Pulses2Waves`)
- Exps 8, 10–12 — regular periodic pulses (`stimRegPulsesWave`)
- `DEFAULT_PULSE_EXPS = [2,3,4,5,6,7,13]` is the TTL-pulse subset used for E/I analysis.

### Style
- Match `data_loader.py`: module-level docstrings explaining *data flow*, private helpers prefixed
  `_`, a small public API, type hints, NumPy-vectorised binning. Comment the *why* (format quirks,
  paper caveats), not the *what*.
- When a new file/field is decoded, record the finding in `M150605_ICTP1_data_file_conjectures.md`
  with a confidence level — that doc is the project's memory of what each file means.

### Analysis cautions (from the paper — respect these when validating figures)
- Extracellular detection **severely underestimates** activity in roughly the first **20 ms** after
  strong excitatory optogenetic pulses (overlapping spikes / field fluctuations). Treat the earliest
  trial-aligned response with suspicion.
- Fine `chrono` Timeline↔Blackrock clock sync is **not** applied; residual drift < ~1 ms / 1000 s.
- The E/I trough-to-peak threshold (default 0.4 ms) should be checked against the bimodal
  waveform-width histogram per session before it's trusted.

---
<!-- Always keep the sections below -->

## How we work

### Cadence
- Each day starts with creating `journal/YYYY-MM-DD.md` with the aims list.
- After each substantive task, update the journal's "Done" section with what changed and why —
  especially any new conclusion about how a file is stored.
- End-of-day: write the "Next" section so the next session (or agent) knows where to pick up.

### Before non-trivial changes
- Propose the approach in 2-3 sentences and confirm before implementing. Especially for:
  - Anything that changes the public API of `data_loader.py`
  - Anything that touches dependencies / adds a `requirements.txt`
  - Reprocessing raw `.ns5`/`.dat` voltage (expensive — avoid unless higher-level data is insufficient)
- For lookups / file edits / small fixes / notebook exploration: just do it.

### Verify before claiming done
- A figure "reproduces" a paper figure only when you've actually rendered it and compared it to the
  PDF. Say what matches and what doesn't — a near-miss is itself evidence about the parsing.
- If something isn't checkable in this environment (data not mounted, etc.), say so explicitly.
- Distinguish the empirical result from its interpretation; flag untested assumptions about the
  file format rather than asserting them.

### Don't
- Add CLI flags, error handling, or abstractions "for the future." Add them when needed.
- Add comments that re-state what the code does. Comment the *why* when non-obvious.
- Assert a file/field meaning as fact when it's a conjecture — mark confidence.
- Bypass the user — no `--no-verify`, `--force`, or amending pushed commits without asking.
- Re-read a file Claude just edited; the harness tracks state.

### Communication
- Short responses by default. Match the question's depth.
- When proposing options, list the recommended one first with `(Recommended)`.
- Surface risks before acting (e.g. "this reprocesses 30 kHz voltage and will be slow").

---

## When the session ends

1. In-session tasks are either done or noted in the journal under "Open / next".
2. Journal's "Next" section lists concrete handoff items (file paths, function names, exact commands).
3. New format findings are folded into `M150605_ICTP1_data_file_conjectures.md`.
4. If we made commits, note the SHA range in the journal.
