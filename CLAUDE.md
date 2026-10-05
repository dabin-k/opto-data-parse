# CLAUDE.md — Working conventions for this repo

How Dabin and Claude collaborate on **ichun_opto**. Auto-loaded as context every session.
A new agent should read this + `README.md` + the latest
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
LGN recordings. 

**Method of validation:** recreate figures from the paper to *confirm* the raw data has been 
parsed correctly. A reproduced figure that matches the paper is the evidence that our reading 
of the file format is right. The deliverable is understanding, and then a pipeline.

**Final aim (where this is heading):** make the loader driven by **mouse ID and experiment type**
as first-class, easily-configured inputs, so any session in this dataset — not just the hardcoded
`M150605_ICTP1` — can be loaded and its figures reproduced. 

## Where things live
| Concern | Location |
|---|---|
| Raw dataset (external, NOT in repo) | `/mnt/scratch/M150605_ICTP1/` |
| Session-level files | `/mnt/scratch/M150605_ICTP1/1/` (manifest `.mat`, `.kwik`, `.kwx`) |
| Per-experiment files | `/mnt/scratch/M150605_ICTP1/1/<exp>/` (`Protocol.mat`, `*_Timeline.mat`, `.ns5`, `.nev`) |
| Data-loading code | `data_loader.py` |
| The paper | `lin_harris_mouse_corticothalamic_dynamical_equation.pdf` |
| Daily journal | `journal/YYYY-MM-DD.md` *(convention to adopt; dir not yet created)* |

The single session in hand is **M150605_ICTP1**, series/session 1. Code defaults assume this
one animal; `animal_id` / `base_dir` are parameterised but only this session has been exercised.

## Environment
**Python env:** miniconda **base** (`/home/dabin/miniconda3/bin/python`, Python 3, NumPy 2.5.1).
No project virtualenv — run against base.
**Dependencies:** `numpy`, `scipy`, `h5py`, `matplotlib` (all present in base). There is **no
`requirements.txt`** yet — add one only if the user asks.

## Git workflow
- Working branch: **master** (repo has **no commits yet** — the first commit will establish it).
- Don't push to anything other than `master` without confirming first.
- Commits: small, focused, descriptive. Use HEREDOC for multi-line commit messages.
- Don't commit the dataset or large artifacts: `/mnt/scratch/...` is external; keep `__pycache__/`,
  `.ns5`/`.dat`/`.kwd` and other large binaries out. There is no `.gitignore` yet — add one before
  the first commit (at minimum `__pycache__/`).
- Commit or push only when the user asks.

### Style
- Match `data_loader.py`: module-level docstrings explaining *data flow*, private helpers prefixed
  `_`, a small public API, type hints, NumPy-vectorised binning. Comment the *why* (format quirks,
  paper caveats), not the *what*.

---
<!-- Always keep the sections below -->

## How we work

### Cadence
- Each day starts with creating `journal/YYYY-MM-DD.md` with the aims list.
- After each substantive task, update the journal's "Done" section with what changed and why —
  especially any new conclusion about how a file is stored.
- When it is required to run a new task that is expected to take many minutes (> 5mins), check with
  the user first whether it is worth running it. It might be the case that we want to skip it / reorder 
  the plan (e.g. run the time consuming jobs later/ just run a proof of concept check, etc)
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
