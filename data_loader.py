"""
data_loader.py — M150605_ICTP1 data loader

Data flow
---------
Spike times come from the SpikeDetekt/KlustaKwik `.kwik` file, which covers a
single concatenated recording of 13 experiments (experiment 9 was interrupted
and excluded, hence the "not9" filename).  The `lims` array in the manifest
`.mat` gives the length of each segment in 30 kHz samples; its cumulative sum
maps spike sample indices back to individual experiment timelines.

Stimulus timing comes from the **laser TTL markers on the Blackrock `.ns5`
analog inputs** (see `laser_timing.py`), NOT from `Timeline.mat`.  Timeline's
`mpepUDP` StimStart times live in a separate DAQ clock that is offset from the
Blackrock clock by seconds, so aligning spikes to them scrambles every trial
(flat PSTHs).  The `.ns5` analog TTLs are recorded in the *same* 30 kHz clock as
the spikes, so their onsets map to concatenated-recording space with the same
`exp_start` offset the spikes use — no cross-clock conversion.  `Timeline.mat`
and `Protocol.mat` are still used, but only to label *what* each trial was:
each measured laser onset is matched to a Timeline trial by relative timing, and
that trial's `cond_id` is looked up in `Protocol.mat`.

Two public functions
--------------------
load_data()
    Returns per-unit (cluster-level) spike counts aligned to trial onsets.
    Shape: (n_trials, n_units, n_bins).

get_population_responses()
    Returns trial-averaged, baseline-normalised, Hamming-smoothed population
    activity for E and I populations, organised by experiment type.
    Shape per type: (n_conditions, 2, n_bins).

    Population activity is defined as in the paper (Lin & Harris 2020):
      E(t) = total rate of all wide-spike events (putative excitatory)
      I(t) = total rate of all narrow-spike events (putative fast-spiking)
    Wide vs. narrow is classified per detected spike by the trough-to-peak
    time of its filtered waveform (from the .kwx file).  This is an
    approximation of the paper's clusterless method, which additionally
    denoised each spike via locality-sensitive hashing before measuring
    waveform features.  (Which laser drove a trial is irrelevant to this split —
    the E/I population signal is read off spike waveforms, not the stimulus.)

    Experiment types returned (keys in output dict):
      'single_E'   BB pulse, intT = 0  (single blue/excitatory pulse)
      'single_I'   GG pulse, intT = 0  (single green/inhibitory pulse)
      'paired_EE'  BB pulse, intT > 0  (two excitatory pulses)
      'paired_II'  GG pulse, intT > 0  (two inhibitory pulses)
      'paired_EI'  BG pulse            (E first, then I)
      'paired_IE'  GB pulse            (I first, then E)
      'flash'      visual LED flash    (not available for M150605A)

Caveats
-------
- Onsets are matched to Timeline trials by relative timing; the residual (the
  small Timeline/Blackrock clock drift) is < ~50 ms and does not affect the
  onset itself, which comes from the analog TTL.  Some Timeline trials produce
  no analog TTL (in exp 5, 10 of 30 single_E did), so the number of returned
  trials can be slightly less than the Timeline trial count; those trials are
  simply dropped.
- The trough-to-peak threshold for E/I classification (default 0.4 ms) should
  be validated by inspecting the bimodal waveform-width distribution for each
  session; pass ei_threshold_ms to override.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from scipy.signal import windows as signal_windows

import ei_classification_specification as ei_spec
import laser_timing
import mouse_lines

SAMPLE_RATE_HZ: int = 30_000  # Blackrock / kwik spike-sample rate
SERIES: int = 1               # always 1 for this animal/session

# Clusterless spike smoothing (paper S1.6): each spike's waveform is denoised by
# averaging the filtered waveforms of its nearest neighbours in the SpikeDetekt
# PCA feature space, before measuring trough-to-peak width.  The paper used LSH to
# approximate this; we do exact k-NN (fast enough, cached).
KNN_K: int = 100          # neighbours cached per spike (incl. self)
KNN_RADIUS: float = 100.0 # dummy radius — ~keep-all in this space; tune from d100
CACHE_DIR = Path(__file__).parent / "cache"

# Wide raw snippet re-extracted from the .dat (the .kwx 20-sample window is too
# short for the E-defining features).  time_sample sits at offset WIDE_PRE; WIDE_POST
# samples follow the trough (~1.3 ms) — covers E-duration (trough+25) and the late
# gradient (trough+15).  The first 20 samples (= dat[t-10:t+10]) match the .kwx window.
WIDE_PRE: int = 10
WIDE_POST: int = 40

# ADC -> microvolt scaling.  The NEURALSG (NSx 2.1) header carries no per-channel
# analog/digital range, so this is NOT recoverable from the file; 0.25 uV/count is
# the Blackrock Cerebus default.  CONJECTURE — verify per rig.  Only the amplitude
# and cross-channel-variability QC thresholds depend on it.
UV_PER_ADC: float = 0.25

# Clusterless quality control (paper S1.6).  Thresholds are the paper's.  On the
# already-clean M150605_ICTP1 they reject ~nothing; they guard noisier
# mice/sessions now that cluster labels no longer pre-filter the spikes.
QC_FIT_ERROR_MAX: float = 0.45     # normalized RMS(filtered - smoothed) / ptp(smoothed)
QC_AMP_MIN_UV: float = 25.0        # min filtered peak-to-trough amplitude
# 5th-/1st-NN normalized distance: rejects spikes with no tight cluster of similar
# spikes (isolated -> the average blends other neurons -> unreliable smoothing).
# The paper's absolute thresholds (0.305 / 0.0305) are in ITS PCA-feature scale; our
# SpikeDetekt features run ~15x smaller relative to ptp (same scale mismatch as the
# gradients), so those values reject ~0%.  We use data-driven, session-tunable
# thresholds instead (M150605 cg0: d5_norm ~0.022 median, 0.042 at 99th pct).
QC_D5_NORM_MAX: float = 0.042      # 5th-NN distance / ptp(smoothed) (our units)
QC_D1_NORM_MAX: float = 0.020      # 1st-NN distance / ptp(smoothed), paired with...
QC_FIT_ERROR_MAX2: float = 0.305   # ...fitting error; reject only if BOTH exceeded
# Cross-channel variability (artifact rejection).  The paper's wording is terse
# ("variability < 2.53 uV^2, and < 5.66 uV^2 if variability/amp < 0.05 uV"); the
# interpretation below (mean over samples of the across-channel variance) is a
# CONJECTURE — flagged, easy to retune/disable via these constants.
QC_XCH_VAR_MIN_UV2: float = 2.53
QC_XCH_VAR_MIN_UV2_COND: float = 5.66
QC_XCH_VAR_AMP_RATIO_UV: float = 0.05

# E/I classification boxes (paper S1.6 p26, Fig S2C).  A spike is E only if inside
# the E-box on every listed feature, I only if inside the I-box, else DISCARDED.
# Duration + FW3M are in ms (calibration-independent) and reproduce S2C-top's two
# blobs; ranges are the paper's first-session values (it tunes them per session).
#
# Gradients: measured at 0.07 ms (trough gradient) and 0.50 ms (peak gradient) after
# the trough.  The paper's absolute thresholds (early 1.16, late 0 uV/ms) can't be
# matched — the .dat has no recoverable uV scale (NEURALSG header; assumed
# UV_PER_ADC=0.25).  But the E/I separation is scale-free and matches the paper's
# signs (I steeper early upstroke; E still rising / I already falling at 0.50 ms), so
# we split with our own data-driven thresholds (in the same uV/ms units
# `_spike_features` returns).  Session-tunable, like the paper's per-session boxes.
EI_GRAD_EARLY_MS: float = 0.07
EI_GRAD_LATE_MS: float = 0.50
# Duration = trough -> "following peak" (paper S1.6).  We take the max of the
# smoothed waveform within this post-trough window rather than the first local
# max: it skips small early shoulders (which underestimate the E cells) without
# reaching end-of-window baseline drift (which overestimates).  Covers the E box
# top (0.83 ms) with headroom; WIDE_POST=40 samples (~1.33 ms) bounds it.
DURATION_PEAK_WINDOW_MS: float = 0.9

# Bump whenever the E/I feature extraction or classification *logic* changes (the
# boxes are hashed separately).  Folded into the classified-spike cache key
# (`_ei_cache_key`) so a code change never silently reuses a stale ei_ cache.
#   1: bounded-window duration ("following peak") on clusterless-smoothed
#      unfiltered waveforms; two-box duration/FW3M/gradient classification.
_EI_CLASSIFY_VERSION: int = 1
EI_GRAD_EARLY_SPLIT: float = 270.0   # E below (slow upstroke), I above (fast)
EI_GRAD_LATE_SPLIT: float = 0.0      # E above (still rising), I below (falling)
# FW3M ranges: the paper lists several per-session values and picks manually per
# session to "maximize detected spikes, consistent with conservative criteria".  A
# scan of all M150605 cg0 candidates spans 39.7-51.0% kept; we take the conservative
# end (wide 0.18-0.30, narrow 0.10-0.18) which yields 39.7% ~ the paper's headline
# ~40%.  Session-tunable.  (Looser ranges keep more but catch more boundary spikes;
# the effect on the E/I *populations* is second-order.)
EI_BOX_E = dict(duration=(0.47, 0.83), fw3m=(0.18, 0.30),
                early=(-np.inf, EI_GRAD_EARLY_SPLIT), late=(EI_GRAD_LATE_SPLIT, np.inf))
EI_BOX_I = dict(duration=(0.13, 0.33), fw3m=(0.10, 0.18),
                early=(EI_GRAD_EARLY_SPLIT, np.inf), late=(-np.inf, EI_GRAD_LATE_SPLIT))


def _mouse_ei_boxes(animal_id: str) -> tuple[dict, dict]:
    """
    (box_e, box_i) for an animal: the full four-metric per-mouse E/I spec from
    `ei_classification_specification.mouse_ei_boxes` (each metric tuned per mouse).
    Falls back to the module-default boxes (`EI_BOX_E`/`EI_BOX_I`) for a mouse
    without a bespoke spec, so unknown animals keep the previous behaviour.
    """
    try:
        return ei_spec.mouse_ei_boxes(animal_id)
    except KeyError:
        return dict(EI_BOX_E), dict(EI_BOX_I)  # no per-mouse spec -> module defaults


def _find_session_files(
    session_dir: Path,
    animal_id: str,
    session: int,
) -> tuple[Path, Path, Path]:
    """
    Locate manifest, kwik, and kwx files for a session.

    Manifest (*.mat) and kwik/kwx may live in session_dir or _klustakwik/.
    The manifest suffix varies per session (`not9`, `all`, `sel`, …): it names
    the experiment-selection the spikes were sorted on, and its `lims` must match
    the `.kwik`'s concatenation.  So we pick the `{animal}_s{n}_*.mat` whose stem
    has a matching `.kwik`, deprioritising the `V1` (visual-only) manifest, which
    is a different selection and normally carries no kwik.

    Returns (kwik_path, kwx_path, manifest_path).
    """
    mats = sorted(session_dir.glob(f"{animal_id}_s{session}_*.mat"))
    # Stable sort pushing the V1 manifest last; matching-kwik requirement below
    # then selects the sorted-on manifest (not9/all/sel/…) over V1.
    mats.sort(key=lambda p: p.stem.endswith("_V1"))
    if not mats:
        raise FileNotFoundError(
            f"No manifest MAT found in {session_dir} for {animal_id} s{session}"
        )

    for manifest_path in mats:
        stem = manifest_path.stem   # e.g. M150605_ICTP1_s1_not9, …_s1_all
        for search_dir in (session_dir, session_dir / "_klustakwik"):
            kwik = search_dir / f"{stem}.kwik"
            if kwik.exists():
                return kwik, search_dir / f"{stem}.kwx", manifest_path

    stems = ", ".join(p.stem for p in mats)
    raise FileNotFoundError(
        f"No manifest MAT with a matching .kwik in {session_dir} or "
        f"_klustakwik/ for {animal_id} s{session} (tried: {stems})"
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_kwik_spikes(
    kwik_path: Path,
    min_cluster_group: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Load spike times and cluster assignments from a kwik (HDF5) file.

    Parameters
    ----------
    kwik_path : Path
    min_cluster_group : int
        Keep clusters whose KlustaKwik group label is >= this value.
        0 = noise (excluded by default), 1 = MUA, 2 = good, 3 = unsorted.
        klusta/phy default cluster-group convention (naming table in .kwik file)

    Returns
    -------
    time_samples : (n_spikes,) int64
        Concatenated-recording sample index for every retained spike.
    unit_ids : (n_spikes,) int32
        Index into unit_info for every retained spike.
    unit_info : list of dicts
        One entry per unit: channel_group, cluster_id, cluster_group, unit_idx.
    """
    all_times: list[np.ndarray] = []
    all_uids: list[np.ndarray] = []
    unit_info: list[dict] = []

    with h5py.File(str(kwik_path), "r") as f:
        for cg_key in sorted(f["channel_groups"].keys()):
            cg = int(cg_key)
            spikes_grp = f[f"channel_groups/{cg_key}/spikes"]
            times = spikes_grp["time_samples"][()]        # (n_spikes,) uint64
            cluster_ids = spikes_grp["clusters/main"][()]  # (n_spikes,) uint32

            # Collect cluster-group labels for this channel group
            cls_meta = f[f"channel_groups/{cg_key}/clusters/main"]
            cg_label: dict[int, int] = {}
            for cid_key in cls_meta.keys():
                cid = int(cid_key)
                grp = int(cls_meta[cid_key].attrs.get("cluster_group", 3))
                cg_label[cid] = grp

            # Build per-cluster uid map (only retained clusters)
            for cid, grp in sorted(cg_label.items()):
                if grp >= min_cluster_group:
                    unit_info.append(
                        dict(
                            channel_group=cg,
                            cluster_id=cid,
                            cluster_group=grp,
                            unit_idx=len(unit_info),
                        )
                    )

            # Vectorised filter and uid assignment
            uid_for_cluster = {
                u["cluster_id"]: u["unit_idx"]
                for u in unit_info
                if u["channel_group"] == cg
            }
            # Map every spike's cluster to its uid (-1 = excluded)
            uids = np.full(len(cluster_ids), -1, dtype=np.int32)
            for cid, uid in uid_for_cluster.items():
                uids[cluster_ids == cid] = uid

            mask = uids >= 0
            all_times.append(times[mask].astype(np.int64))
            all_uids.append(uids[mask])

    time_samples = np.concatenate(all_times)
    unit_ids = np.concatenate(all_uids)

    # Sort by time so searchsorted works later
    order = np.argsort(time_samples, kind="stable")
    return time_samples[order], unit_ids[order], unit_info


def _parse_timeline_trials(timeline_path: Path) -> list[dict]:
    """
    Parse mpepUDP StimStart/StimEnd events from a Timeline.mat file.

    Returns a list of trial dicts with keys:
        onset_s, offset_s, exp, block, cond_id
    Times are in seconds relative to Timeline DAQ start (t = 0).

    NOTE: onset_s is in the *Timeline* clock, which is offset from the Blackrock
    spike clock — it is used only as a per-trial ordering / relative-timing key
    to match against the analog laser onsets, never as an absolute spike-aligned
    time.  See the module docstring.
    """
    mat = scipy.io.loadmat(str(timeline_path), squeeze_me=True, struct_as_record=False)
    tl = mat["Timeline"]
    n_ev = int(tl.mpepUDPCount)
    events: np.ndarray = tl.mpepUDPEvents[:n_ev]
    times: np.ndarray = tl.mpepUDPTimes[:n_ev]

    start_pattern = re.compile(
        r"StimStart\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    )
    end_pattern = re.compile(
        r"StimEnd\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    )

    open_trials: dict[tuple, dict] = {}
    trials: list[dict] = []

    for ev, t in zip(events, times):
        m = start_pattern.match(ev)
        if m:
            series, exp, block, cond = map(int, m.groups())
            open_trials[(block, cond)] = dict(
                onset_s=float(t), exp=exp, block=block, cond_id=cond
            )
            continue
        m = end_pattern.match(ev)
        if m:
            series, exp, block, cond = map(int, m.groups())
            key = (block, cond)
            if key in open_trials:
                trial = open_trials.pop(key)
                trial["offset_s"] = float(t)
                trials.append(trial)

    return trials


def _match_onsets_to_timeline(
    onsets: np.ndarray,
    trials: list[dict],
    tol_s: float = 1.0,
) -> list[tuple[int, dict]]:
    """
    Match measured laser onsets (Blackrock samples) to Timeline trials.

    The two clocks share the same rate to within negligible drift over one
    experiment, so a matching analog onset and Timeline trial differ by a
    constant time shift.  We anchor the first analog onset to whichever of the
    first few Timeline trials maximises the number of consistent matches (robust
    to a missing pulse at the start), then assign each onset to its nearest
    Timeline trial under that shift.

    Returns (onset_sample, trial) pairs, sorted by onset, for onsets that matched
    a trial within `tol_s`.
    """
    if onsets.size == 0 or not trials:
        return []

    a = onsets.astype(np.float64) / SAMPLE_RATE_HZ
    b = np.array([t["onset_s"] for t in trials], dtype=np.float64)

    best_nearest: np.ndarray | None = None
    best_resid: np.ndarray | None = None
    best_score = -1
    for j0 in range(min(8, b.size)):
        shift = a[0] - b[j0]
        nearest = np.abs((b[np.newaxis, :] + shift) - a[:, np.newaxis]).argmin(axis=1)
        resid = np.abs(b[nearest] + shift - a)
        score = int(np.sum(resid < tol_s))
        if score > best_score:
            best_score, best_nearest, best_resid = score, nearest, resid

    # Assign, keeping the closest onset when two map to the same trial.
    chosen: dict[int, tuple[int, float]] = {}
    for i in range(a.size):
        if best_resid[i] >= tol_s:
            continue
        j = int(best_nearest[i])
        if j not in chosen or best_resid[i] < chosen[j][1]:
            chosen[j] = (i, float(best_resid[i]))

    matches = [(int(onsets[i]), trials[j]) for j, (i, _) in chosen.items()]
    matches.sort(key=lambda m: m[0])
    return matches


def _load_experiment_trials(
    session_dir: Path,
    exp_num: int,
    animal_id: str,
    iti_gap_s: float = 1.0,
) -> list[dict]:
    """
    Trials for one experiment: measured laser onset + condition label.

    Combines the analog-TTL onsets (timing, Blackrock samples, experiment-local)
    with the matched Timeline trial and its Protocol condition (labels).  Returns
    an empty list for experiments with no detectable laser pulses (e.g. visual
    tuning experiments), which cannot be aligned by this method.

    Each returned dict has:
        onset_sample : int   experiment-local Blackrock sample of the first pulse
        exp, block, cond_id
        pulse_type, ipi_ms, dur_ms, exp_type   (from Protocol)
    """
    ns5 = laser_timing.find_ns5(session_dir, exp_num, animal_id)
    if ns5 is None:
        return []
    onsets = laser_timing.experiment_trial_onsets(ns5, iti_gap_s=iti_gap_s)
    if onsets.size == 0:
        return []

    timeline_path = session_dir / str(exp_num) / f"1_{exp_num}_{animal_id}_Timeline.mat"
    protocol_path = session_dir / str(exp_num) / "Protocol.mat"
    if not timeline_path.exists() or not protocol_path.exists():
        return []

    trials = _parse_timeline_trials(timeline_path)
    if not trials:
        return []
    cond_map = _load_protocol_conditions(protocol_path, _wavelength_to_pop(animal_id))

    out: list[dict] = []
    for onset_sample, trial in _match_onsets_to_timeline(onsets, trials):
        cond = cond_map.get(trial["cond_id"])
        if cond is None:
            continue   # pulse type we don't model
        out.append(
            dict(
                onset_sample=int(onset_sample),
                exp=int(exp_num),
                block=trial["block"],
                cond_id=trial["cond_id"],
                pulse_type=cond["pulse_type"],
                ipi_ms=cond["ipi_ms"],
                dur_ms=cond["dur_ms"],
                exp_type=cond["exp_type"],
            )
        )
    return out


def _pulse_mask(trial: dict, n_bins: int, pre_s: float, bin_s: float) -> np.ndarray:
    """
    Binary laser-on indicator for one trial, from Protocol pulse timing.

    The pulse(s) are anchored at the measured onset (bin for t = 0) using the
    intended duration and interpulse interval from Protocol.
    """
    s = np.zeros(n_bins, dtype=np.float32)
    # dur_ms is a scalar (both pulses equal / single pulse) or a (d1, d2) tuple
    # when the two pulses have different durations (2Dur variant).
    dur = trial["dur_ms"]
    d1_ms, d2_ms = dur if isinstance(dur, tuple) else (dur, dur)

    def mark(t0_s: float, dur_ms: float) -> None:
        dur_s = dur_ms / 1000.0
        lo = int(round((pre_s + t0_s) / bin_s))
        hi = int(round((pre_s + t0_s + dur_s) / bin_s))
        hi = max(hi, lo + 1)   # a sub-bin pulse (< bin_s) still marks one bin
        s[max(lo, 0):min(hi, n_bins)] = 1.0

    mark(0.0, d1_ms)
    if trial["ipi_ms"] > 0:
        mark(trial["ipi_ms"] / 1000.0, d2_ms)
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    base_dir: str | Path = "/mnt/scratch/M150605_ICTP1",
    session: int = 1,
    pre_s: float = 0.5,
    post_s: float = 1.5,
    bin_s: float = 0.005,
    min_cluster_group: int = 1,
    selected_exps: list[int] | None = None,
    animal_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    """
    Load trial-aligned spike counts and stimulus traces for one session.

    Parameters
    ----------
    base_dir : str or Path
        Root directory containing the session subdirectory.
    session : int
        Session number (always 1 for this dataset).
    pre_s : float
        Seconds before stimulus onset included in each trial window.
    post_s : float
        Seconds after stimulus onset included in each trial window.
    bin_s : float
        Spike-count bin width in seconds.
    min_cluster_group : int
        Minimum KlustaKwik cluster group to retain
        (0 = noise, 1 = MUA, 2 = good, 3 = unsorted).
        Default 1 keeps everything except noise.
    selected_exps : list of int, optional
        Restrict to these experiment numbers (1-indexed, matching the log file).
        None (default) loads every experiment in the manifest; those without
        detectable laser pulses (e.g. visual tuning experiments) are skipped,
        because trial alignment here relies on the laser TTL.

    Returns
    -------
    responses : ndarray, shape (n_trials, n_units, n_bins), float32
        Spike counts per bin.  Units are ordered by (channel_group, cluster_id).
    stimulus : ndarray, shape (n_trials, n_bins), float32
        Binary laser-on indicator built from the Protocol pulse timing, anchored
        at the measured onset.  1 during each laser pulse, 0 otherwise.
    time_axis : ndarray, shape (n_bins,), float64
        Bin centres in seconds relative to stimulus onset (negative = pre).
    trial_info : list of dicts, length n_trials
        Per-trial metadata: exp, block, cond_id, onset_s (experiment-local
        seconds), onset_sample (concatenated Blackrock sample), pulse_type,
        ipi_ms, dur_ms, exp_type.
    unit_info : list of dicts, length n_units
        Per-unit metadata: channel_group, cluster_id, cluster_group, unit_idx.

    Notes
    -----
    Memory: with default settings and ~3 000 trials, responses will be roughly
    (3000, 315, 400) float32 ≈ 1.5 GB.  Increase bin_s or narrow the window
    to reduce memory use.
    """
    base_dir = Path(base_dir)
    if animal_id is None:
        animal_id = base_dir.name
    session_dir = base_dir / str(session)

    kwik_path, _kwx_path, manifest_path = _find_session_files(
        session_dir, animal_id, session
    )

    # --- segment boundaries ------------------------------------------------
    manifest = scipy.io.loadmat(str(manifest_path), squeeze_me=True)
    lims: np.ndarray = manifest["lims"].astype(np.int64)
    all_exps: np.ndarray = manifest["SELECTED_EXPERIMENTS"].astype(int)
    exp_start_samples = np.concatenate([[0], np.cumsum(lims)])

    if selected_exps is not None:
        exp_mask = np.isin(all_exps, selected_exps)
    else:
        exp_mask = np.ones(len(all_exps), dtype=bool)

    # --- load spikes -------------------------------------------------------
    print("Loading spike data from kwik file …")
    time_samples, unit_ids, unit_info = _load_kwik_spikes(kwik_path, min_cluster_group)
    n_units = len(unit_info)
    print(f"  {n_units} units, {len(time_samples):,} spikes retained")

    # --- time axis ---------------------------------------------------------
    n_bins = int(round((pre_s + post_s) / bin_s))
    bin_edges = np.linspace(-pre_s, post_s, n_bins + 1)
    time_axis = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- per-unit sorted spike arrays (concatenated-sample space) ----------
    # Group spike indices by unit so each unit has a sorted time array.
    unit_spike_samples: list[np.ndarray] = []
    for uid in range(n_units):
        mask = unit_ids == uid
        unit_spike_samples.append(time_samples[mask])  # already sorted

    # --- iterate over experiments and trials --------------------------------
    all_responses: list[np.ndarray] = []
    all_stimuli: list[np.ndarray] = []
    all_trial_info: list[dict] = []

    for exp_i, exp_num in enumerate(all_exps):
        if not exp_mask[exp_i]:
            continue

        trials = _load_experiment_trials(session_dir, int(exp_num), animal_id)
        if not trials:
            print(f"  exp {exp_num}: no laser trials, skipping")
            continue

        print(f"  exp {exp_num}: {len(trials)} trials")

        exp_start = exp_start_samples[exp_i]
        exp_end = exp_start_samples[exp_i + 1]

        for trial in trials:
            onset_sample = exp_start + trial["onset_sample"]
            win_start_sample = onset_sample - int(pre_s * SAMPLE_RATE_HZ)
            win_end_sample = onset_sample + int(post_s * SAMPLE_RATE_HZ)

            # Clamp to experiment boundaries
            win_start_sample = max(win_start_sample, exp_start)
            win_end_sample = min(win_end_sample, exp_end)

            # --- responses: (n_units, n_bins) --------------------------------
            r = np.zeros((n_units, n_bins), dtype=np.float32)

            for uid, st in enumerate(unit_spike_samples):
                lo = int(np.searchsorted(st, win_start_sample))
                hi = int(np.searchsorted(st, win_end_sample))
                if lo >= hi:
                    continue
                # Spike times relative to onset, in seconds
                spike_t_s = (st[lo:hi].astype(np.float64) - onset_sample) / SAMPLE_RATE_HZ
                # Bin index: shift by pre_s so that t = -pre_s → bin 0
                bin_idx = ((spike_t_s + pre_s) / bin_s).astype(np.int32)
                valid = (bin_idx >= 0) & (bin_idx < n_bins)
                np.add.at(r[uid], bin_idx[valid], 1.0)

            all_responses.append(r)

            # --- stimulus: binary laser-on indicator from Protocol -----------
            all_stimuli.append(_pulse_mask(trial, n_bins, pre_s, bin_s))

            all_trial_info.append(
                dict(
                    exp=trial["exp"],
                    block=trial["block"],
                    cond_id=trial["cond_id"],
                    onset_s=trial["onset_sample"] / SAMPLE_RATE_HZ,
                    onset_sample=int(onset_sample),
                    pulse_type=trial["pulse_type"],
                    ipi_ms=trial["ipi_ms"],
                    dur_ms=trial["dur_ms"],
                    exp_type=trial["exp_type"],
                )
            )

    if not all_responses:
        raise RuntimeError("No trials found. Check selected_exps and file paths.")

    responses = np.stack(all_responses)   # (n_trials, n_units, n_bins)
    stimulus = np.stack(all_stimuli)      # (n_trials, n_bins)

    return responses, stimulus, time_axis, all_trial_info, unit_info


# ---------------------------------------------------------------------------
# E / I population helpers
# ---------------------------------------------------------------------------

# Protocol pulseType integer → the two pulse wavelengths (nm).
# B = Blue = 445 nm, G = Green = 561 nm.  These are physical constants; which
# population each wavelength drives is mouse-line dependent (see mouse_lines.py).
_PULSE_TYPE_WAVELENGTHS = {
    1: (445, 445),   # BB
    2: (561, 561),   # GG
    3: (445, 561),   # BG
    4: (561, 445),   # GB
}


def _wavelength_to_pop(animal_id: str) -> dict[int, str]:
    """
    {wavelength_nm → 'E'|'I'} for this animal, from its mouse line.

    e.g. PVcre;Thy18+C1V1 → {445: 'E', 561: 'I'};  PVcre;Ai32 → {445: 'I'}.
    A wavelength that drives no opsin in this line is simply absent.
    """
    e_nm, i_nm = mouse_lines.mouse_wavelengths(animal_id)
    wl: dict[int, str] = {}
    if e_nm is not None:
        wl[e_nm] = "E"
    if i_nm is not None:
        wl[i_nm] = "I"
    return wl


def _pulse_populations(pulse_type: int, wl_to_pop: dict[int, str]) -> tuple[str, str] | None:
    """(first_pop, second_pop) for a pulseType, or None if a wavelength is not
    drivable in this mouse line."""
    w1, w2 = _PULSE_TYPE_WAVELENGTHS[pulse_type]
    if w1 not in wl_to_pop or w2 not in wl_to_pop:
        return None
    return wl_to_pop[w1], wl_to_pop[w2]


def _exp_type_key(pulse_type: int, ipi_ms: int, wl_to_pop: dict[int, str]) -> str | None:
    """Experiment-type key ('single_E', 'paired_EI', …), or None if untargetable."""
    pops = _pulse_populations(pulse_type, wl_to_pop)
    if pops is None:
        return None
    first, second = pops
    if ipi_ms == 0:
        return f"single_{first}"
    return f"paired_{first}{second}"

def _knn_features(
    features: np.ndarray,
    k: int = KNN_K,
    block: int = 1000,
    verbose: bool = True,
    query_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact k-nearest-neighbour search over one shank's spikes (plain Euclidean on
    the 24 SpikeDetekt PCA features).  The pool is always the full `features`; by
    default every point is also a query (so a spike's own nearest neighbour is
    itself, distance 0).  Pass `query_idx` to search neighbours for only that subset
    of query points against the full pool — the score matrix becomes
    len(query_idx) x n instead of n x n, so a 1000-spike sample is ~1000/n of the
    work while still finding each sample's true 100 nearest from all spikes.

    Blocked brute-force: all-pairs distance is O(n^2), but the 24-d dot product is
    cheap in BLAS, so the cost is memory bandwidth over the block x n score matrix
    — we process `block` query rows at a time to bound it.  For ranking within a
    query row the constant |query|^2 term drops out, so we rank on the score
    `dot(q,x) - 0.5*|x|^2`.  That score is produced by a *single* GEMM by augmenting
    each pool point with a |x|^2 column and each query with a constant -0.5 column,
    which folds the former `- 0.5|x|^2` broadcast into the matmul.  True distances
    fall straight out of the winners' scores: |q-x|^2 = |q|^2 - 2*score.

    Returns
    -------
    nbr_idx  : (nq, k) int32   pool indices of each query's k nearest (col 0 = self)
    nbr_dist : (nq, k) float32 Euclidean distance to each, sorted near -> far
                (nq = n, or len(query_idx) when a subset is given)
    """
    X = np.ascontiguousarray(features, dtype=np.float32)
    n, d = X.shape
    k = min(k, n)
    sq = np.einsum("ij,ij->i", X, X).astype(np.float32)   # |x|^2 per pool point

    Q = X if query_idx is None else X[query_idx]
    sq_q = sq if query_idx is None else sq[query_idx]
    nq = Q.shape[0]

    # Augmented pool: [x, |x|^2]; query block gets a constant -0.5 in the extra
    # column, so q_aug @ pool_aug.T == dot(q,x) - 0.5|x|^2 == the ranking score.
    pool_aug = np.empty((n, d + 1), dtype=np.float32)
    pool_aug[:, :d] = X
    pool_aug[:, d] = sq
    pool_aug_T = np.ascontiguousarray(pool_aug.T)

    nbr_idx = np.empty((nq, k), dtype=np.int32)
    nbr_dist = np.empty((nq, k), dtype=np.float32)
    n_blocks = (nq + block - 1) // block
    report_every = max(1, n_blocks // 20)                 # ~5% steps
    q_aug = np.empty((block, d + 1), dtype=np.float32)
    q_aug[:, d] = -0.5
    for bi, lo in enumerate(range(0, nq, block)):
        hi = min(lo + block, nq)
        b = hi - lo
        q_aug[:b, :d] = Q[lo:hi]
        score = q_aug[:b] @ pool_aug_T                     # (b, n)  dot - 0.5|x|^2
        part = np.argpartition(score, n - k, axis=1)[:, n - k:]  # (b, k) nearest, unordered
        score_w = np.take_along_axis(score, part, axis=1)        # (b, k)
        d2 = sq_q[lo:hi, np.newaxis] - 2.0 * score_w           # |q-x|^2 = |q|^2 - 2*score
        dd = np.sqrt(np.clip(d2, 0.0, None)).astype(np.float32)
        order = np.argsort(dd, axis=1)                     # nearest first
        nbr_idx[lo:hi] = np.take_along_axis(part, order, axis=1)
        nbr_dist[lo:hi] = np.take_along_axis(dd, order, axis=1)
        if verbose and (bi % report_every == 0 or hi == nq):
            print(f"  knn {hi}/{nq} ({100 * hi / nq:.0f}%)", flush=True)
    return nbr_idx, nbr_dist


def _load_or_build_knn(
    cache_key: str, features: np.ndarray, k: int = KNN_K
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cache wrapper for `_knn_features`.  If the cache exists it is loaded (a shank is
    never rebuilt — the basis of the restartable `prebuild_knn_caches`); otherwise
    it is built once (~1 h for a ~900k-spike shank) and saved.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"knn_{cache_key}_k{k}.npz"
    if path.exists():
        print(f"knn: using cached {path.name}", flush=True)
        d = np.load(path)
        return d["nbr_idx"], d["nbr_dist"]
    print(f"knn: building {path.name} ({len(features)} spikes)...", flush=True)
    nbr_idx, nbr_dist = _knn_features(features, k)
    np.savez(path, nbr_idx=nbr_idx, nbr_dist=nbr_dist)
    return nbr_idx, nbr_dist


def prebuild_knn_caches(
    base_dir: str | Path,
    animal_id: str | None = None,
    session: int = 1,
    min_cluster_group: int = 2,
) -> None:
    """
    Build the k-NN cache for every channel group (shank) of a session, skipping any
    already cached.  Safe to re-run and to interrupt — each shank is written
    atomically by `_load_or_build_knn`, so an overnight "build everything" run is
    restartable.  `min_cluster_group=0` matches the paper's clusterless pool.
    """
    base_dir = Path(base_dir)
    animal_id = animal_id or base_dir.name
    session_dir = base_dir / str(session)
    kwik_path, kwx_path, _ = _find_session_files(session_dir, animal_id, session)
    with h5py.File(str(kwik_path), "r") as fk, h5py.File(str(kwx_path), "r") as fx:
        for cg_key in sorted(fk["channel_groups"].keys()):
            clusters = fk[f"channel_groups/{cg_key}/spikes/clusters/main"][()]
            cls_meta = fk[f"channel_groups/{cg_key}/clusters/main"]
            cg_label = {int(k): int(cls_meta[k].attrs.get("cluster_group", 3))
                        for k in cls_meta.keys()}
            mask = np.array([cg_label.get(int(c), 3) >= min_cluster_group
                             for c in clusters])
            features = fx[f"channel_groups/{cg_key}/features_masks"][:, :, 0][mask]
            cache_key = f"{kwx_path.stem}_cg{cg_key}_mcg{min_cluster_group}"
            _load_or_build_knn(cache_key, features)


def _infer_dat_shape(dat_path: Path, max_time_sample: int) -> tuple[int, int]:
    """
    Infer (n_samples, n_channels) of a headerless int16 `.dat`.

    The file is a flat (n_samples x n_channels) int16 array with no header, so the
    shape is recovered from the byte count: pick the largest channel count whose
    n_samples still exceeds the largest spike `time_sample` (i.e. the tightest fit).
    For M150605_ICTP1_s1_not9.dat this gives 36 channels, n_samples exceeding the
    max spike by 66 — an exact, unambiguous fit (verified bit-for-bit against
    `waveforms_raw`).
    """
    total = dat_path.stat().st_size // 2                  # int16 values
    best = None
    for n_ch in range(1, 257):
        if total % n_ch:
            continue
        n_samples = total // n_ch
        if n_samples <= max_time_sample:
            continue
        excess = n_samples - max_time_sample
        if best is None or excess < best[2]:
            best = (n_samples, n_ch, excess)
    if best is None:
        raise ValueError(f"could not infer .dat shape for {dat_path.name}")
    return best[0], best[1]


def _extract_wide_waveforms(
    dat_path: Path,
    n_samples: int,
    n_channels: int,
    time_samples: np.ndarray,
    cols: np.ndarray,
    pre: int = 10,
    post: int = 40,
) -> np.ndarray:
    """
    Cut wide raw snippets from the continuous `.dat` voltage around each spike.

    The `.kwx` only stores 20-sample snippets (too short: a wide spike's peak and
    late gradient fall off the end).  The `.dat` is the full voltage those snippets
    were cut from, with `time_sample` at offset 10 in the stored 20-sample window;
    here we re-cut `dat[t-pre : t+post, cols]`, so the trough again sits at offset
    `pre` but with `post` samples after it (default 40 = ~1.3 ms, enough for
    E-duration at trough+25 and the late gradient at trough+15).

    Returns (n_spikes, pre+post, len(cols)) int16; windows are zero-padded at the
    recording edges (rare).
    """
    mm = np.memmap(dat_path, dtype="<i2", mode="r", shape=(n_samples, n_channels))
    cols = np.asarray(cols)
    w = pre + post
    out = np.zeros((len(time_samples), w, len(cols)), dtype=np.int16)
    for i, t in enumerate(time_samples):
        lo, hi = int(t) - pre, int(t) + post
        a, b = max(lo, 0), min(hi, n_samples)
        out[i, a - lo : (a - lo) + (b - a)] = mm[a:b][:, cols]
    return out


def _load_or_extract_wide(
    cache_key: str,
    dat_path: Path,
    n_samples: int,
    n_channels: int,
    time_samples: np.ndarray,
    cols: np.ndarray,
    pre: int = WIDE_PRE,
    post: int = WIDE_POST,
) -> np.ndarray:
    """Cache wrapper for `_extract_wide_waveforms` (one-time ~10 min / ~700 MB per shank)."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"wide_{cache_key}_pre{pre}post{post}.npy"
    if path.exists():
        return np.load(path)
    print(f"wide: extracting {path.name} ({len(time_samples)} spikes)...", flush=True)
    w = _extract_wide_waveforms(dat_path, n_samples, n_channels, time_samples, cols, pre, post)
    np.save(path, w)
    return w


def _clusterless_smooth_waveforms(
    waveforms: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_dist: np.ndarray,
    distance_threshold: float = KNN_RADIUS,
    max_neighbours: int = KNN_K,
) -> np.ndarray:
    """
    Denoise each spike by averaging the waveforms of its nearest neighbours in
    feature space (paper S1.6).  In the E/I path `waveforms` are the *unfiltered*
    wide snippets re-extracted from the `.dat`, matching the paper's "smoothed
    unfiltered waveform".

    Distance is measured on the PCA features (already computed into `nbr_dist`);
    averaging is over the passed-in `waveforms`.  Neighbours are pre-sorted
    near -> far, so capping at `max_neighbours` keeps the *nearest* ones; the
    radius then drops any beyond `distance_threshold` (self is always kept).

    Parameters
    ----------
    waveforms : (n_pool, n_samples, n_channels)  waveforms indexed by `nbr_idx`.
        Usually the whole pool (n_pool == n_query); in sample mode it is just the
        pool spikes that appear as neighbours, with `nbr_idx` already remapped to
        index into it.
    nbr_idx   : (n_query, k) int   pool indices of each query's k nearest (col 0 = self)
    nbr_dist  : (n_query, k) float Euclidean feature distance to each neighbour

    Returns
    -------
    smoothed : (n_query, n_samples, n_channels) float32
    """
    nq = nbr_idx.shape[0]                           # one smoothed waveform per query
    ns, nc = waveforms.shape[1:]
    k = min(max_neighbours, nbr_idx.shape[1])
    idx = nbr_idx[:, :k]
    keep = nbr_dist[:, :k] <= distance_threshold   # (nq, k) bool
    keep[:, 0] = True                              # always keep self
    wf = waveforms.astype(np.float32)

    smoothed = np.empty((nq, ns, nc), dtype=np.float32)
    block = 2000                                   # bound the (b, k, ns, nc) gather
    for lo in range(0, nq, block):
        hi = min(lo + block, nq)
        gathered = wf[idx[lo:hi]]                   # (b, k, ns, nc)
        w = keep[lo:hi].astype(np.float32)[:, :, np.newaxis, np.newaxis]
        smoothed[lo:hi] = (gathered * w).sum(1) / w.sum(1)
    return smoothed


def _baseline_subtract(waveforms: np.ndarray, n_base: int = 4) -> np.ndarray:
    """
    Zero each (spike, channel) by the mean of its first `n_base` samples.

    The paper averages the *unfiltered* waveforms, which carry a DC baseline (raw
    snippets sit ~366 ADC off zero vs ~16 for filtered).  Averaging preserves that
    offset, so the smoothed waveform must be baselined before width features (FW3M,
    trough-to-peak) or the fitting-error QC compares it to the ~zero-baseline
    filtered target.  The trough sits near sample ~9 of 20, so the first few
    samples are pre-spike baseline.
    """
    base = waveforms[:, :n_base, :].mean(axis=1, keepdims=True)
    return waveforms - base


def _waveform_qc(
    filtered: np.ndarray, smoothed: np.ndarray, nbr_dist: np.ndarray
) -> np.ndarray:
    """
    Clusterless quality control (paper S1.6): keep-mask over spikes.

    Rejects spikes whose smoothed waveform is unreliable (badly fit, low amplitude,
    no close neighbours) or looks like a movement/photoelectric artifact (waveform
    near-identical across recording sites).  `filtered` and `smoothed` are both
    baseline-subtracted (n, n_samples, n_channels); `nbr_dist` is the (n, k)
    Euclidean feature distances (col 0 = self).

    On the already-clean M150605_ICTP1 these reject ~nothing — they exist so the
    clusterless pipeline stays robust on noisier mice/sessions.
    """
    n = filtered.shape[0]
    filt = filtered.reshape(n, -1)
    sm = smoothed.reshape(n, -1)
    ptp = sm.max(1) - sm.min(1)                       # smoothed peak-to-trough (ADC)
    amp = filt.max(1) - filt.min(1)                   # filtered peak-to-trough (ADC)
    safe_ptp = np.where(ptp > 0, ptp, np.inf)
    safe_amp = np.where(amp > 0, amp, np.inf)

    fit_err = np.sqrt(((filt - sm) ** 2).mean(1)) / safe_ptp
    # nbr_dist col 0 is self, so the 1st / 5th *neighbours* are cols 1 and 5.
    d1 = nbr_dist[:, 1] / safe_ptp                    # 1st neighbour, ptp-normalized
    d5 = nbr_dist[:, min(5, nbr_dist.shape[1] - 1)] / safe_ptp   # 5th neighbour

    # Cross-channel variability: mean over samples of the variance across the
    # recording sites.  Low = same signal everywhere = far-field artifact.
    # (Interpretation is a conjecture — see QC_XCH_* constants.)
    xch_var = smoothed.var(axis=2).mean(axis=1) * (UV_PER_ADC ** 2)   # uV^2

    keep = (
        (fit_err <= QC_FIT_ERROR_MAX)
        & (amp >= QC_AMP_MIN_UV / UV_PER_ADC)
        & (d5 <= QC_D5_NORM_MAX)
        & ~((d1 > QC_D1_NORM_MAX) & (fit_err > QC_FIT_ERROR_MAX2))
        & (xch_var >= QC_XCH_VAR_MIN_UV2)
        & ~((xch_var < QC_XCH_VAR_MIN_UV2_COND)
            & (xch_var / safe_amp / UV_PER_ADC < QC_XCH_VAR_AMP_RATIO_UV))
    )
    return keep


def _dominant_trace(waveforms: np.ndarray) -> np.ndarray:
    """(n, n_samples) trace of the largest peak-to-trough channel, per spike."""
    w = waveforms.astype(np.float32)
    n = w.shape[0]
    dom = (w.max(1) - w.min(1)).argmax(1)
    return w[np.arange(n), :, dom]


def _fw3m_ms(dom_trace: np.ndarray, frac: float = 2 / 3) -> np.ndarray:
    """
    Full width at `frac` of trough depth (FW3M for frac=2/3), in ms.

    Measured on the trough (baseline = 0; waveforms are baseline-subtracted
    upstream), so it fits inside the snippet window.  The two threshold crossings
    (down before the trough, up after it) are linearly interpolated between samples.
    NaN if the trough never rises back above the level on one side.
    """
    n, ns = dom_trace.shape
    idx = np.arange(n)
    grid = np.arange(ns)[np.newaxis, :]
    trough = dom_trace.argmin(1)
    level = frac * dom_trace.min(1)                 # negative threshold
    above = dom_trace > level[:, np.newaxis]

    ml = above & (grid <= trough[:, np.newaxis])
    has_l = ml.any(1)
    li = np.clip(ns - 1 - np.argmax(ml[:, ::-1], 1), 0, ns - 2)
    a0, a1 = dom_trace[idx, li], dom_trace[idx, li + 1]
    t_left = li + (a0 - level) / np.where(a1 != a0, a0 - a1, 1e9)

    mr = above & (grid >= trough[:, np.newaxis])
    has_r = mr.any(1)
    ri = np.clip(np.argmax(mr, 1), 1, ns - 1)
    b0, b1 = dom_trace[idx, ri - 1], dom_trace[idx, ri]
    t_right = (ri - 1) + (level - b0) / np.where(b1 != b0, b1 - b0, 1e9)

    width = (t_right - t_left) / (SAMPLE_RATE_HZ / 1000.0)
    width[~(has_l & has_r)] = np.nan
    return width


def _spike_features(smoothed: np.ndarray) -> dict[str, np.ndarray]:
    """
    Four waveform features (paper S1.6) from a smoothed, baseline-subtracted
    waveform (n, n_samples, n_channels), all on the dominant channel:

      duration : trough -> first local max after it (ms), parabola-interpolated
      fw3m     : full width at 2/3 trough depth (ms)
      early    : gradient at 0.07 ms after trough (uV/ms)
      late     : gradient at 0.50 ms after trough (uV/ms)

    Duration is trough -> the max within DURATION_PEAK_WINDOW_MS after it (the
    paper's "following peak"), so an early shoulder can't collapse it short and
    end-of-window drift can't inflate it.
    """
    dw = _dominant_trace(smoothed)
    n, ns = dw.shape
    idx = np.arange(n)
    fs_khz = SAMPLE_RATE_HZ / 1000.0
    trough = dw.argmin(1)

    # "following peak" = max of the smoothed waveform within a physiological window
    # after the trough (paper S1.6).  A bounded-window max avoids two failure modes
    # of a bare first-local-max: an early shoulder just past the trough (which
    # collapses duration short and drops E cells) and end-of-window drift.
    win = int(round(DURATION_PEAK_WINDOW_MS * fs_khz))
    grid = np.arange(ns)[np.newaxis, :]
    in_win = (grid >= trough[:, np.newaxis]) & (grid <= (trough + win)[:, np.newaxis])
    peak = np.where(in_win, dw, -np.inf).argmax(1)          # following-peak sample
    # parabolic sub-sample refinement around the peak
    pj = np.clip(peak, 1, ns - 2)
    y0, y1, y2 = dw[idx, pj - 1], dw[idx, pj], dw[idx, pj + 1]
    denom = y0 - 2 * y1 + y2
    offset = np.where(denom != 0, 0.5 * (y0 - y2) / np.where(denom != 0, denom, 1), 0.0)
    offset = np.clip(offset, -1.0, 1.0)
    duration = (pj + offset - trough) / fs_khz

    def grad_at(ms: float) -> np.ndarray:
        j = np.clip(trough + int(round(ms * fs_khz)), 1, ns - 2)
        slope = (dw[idx, j + 1] - dw[idx, j - 1]) / 2.0     # ADC / sample
        return slope * UV_PER_ADC * fs_khz                  # uV / ms

    return dict(
        duration=duration,
        fw3m=_fw3m_ms(dw),
        early=grad_at(EI_GRAD_EARLY_MS),
        late=grad_at(EI_GRAD_LATE_MS),
    )


def _classify_ei_boxes(
    features: dict[str, np.ndarray],
    box_e: dict | None = None,
    box_i: dict | None = None,
) -> np.ndarray:
    """
    Two-box E/I classification (paper S1.6).  Returns labels int8:
    0 = E (in E-box on all four features), 1 = I (in I-box), -1 = discard.
    ~40% of spikes land in a box; the rest are excluded.

    `box_e`/`box_i` default to the module `EI_BOX_E`/`EI_BOX_I` (tuned on M150605).
    The paper tunes duration/FW3M boundaries per session, so pass per-mouse boxes
    here for other animals.
    """
    box_e = EI_BOX_E if box_e is None else box_e
    box_i = EI_BOX_I if box_i is None else box_i

    def in_box(box):
        m = np.ones(len(features["duration"]), dtype=bool)
        for key, (lo, hi) in box.items():
            v = features[key]
            m &= (v >= lo) & (v <= hi) & ~np.isnan(v)
        return m

    labels = np.full(len(features["duration"]), -1, dtype=np.int8)
    labels[in_box(box_e)] = 0
    labels[in_box(box_i)] = 1
    return labels

def _ei_cache_key(
    kwx_stem: str, min_cluster_group: int, box_e: dict | None, box_i: dict | None
) -> str:
    """
    Cache key for the classified `(time_samples, ei_labels)` output.  Captures the
    E/I boxes (with their module defaults resolved) *and* `_EI_CLASSIFY_VERSION` in
    an 8-char content hash, so editing a classification box OR the feature/logic
    version invalidates the cache rather than silently returning stale labels.
    """
    eff_e = EI_BOX_E if box_e is None else box_e
    eff_i = EI_BOX_I if box_i is None else box_i
    payload = json.dumps({"v": _EI_CLASSIFY_VERSION, "e": eff_e, "i": eff_i}, sort_keys=True)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    return f"ei_{kwx_stem}_mcg{min_cluster_group}_v{_EI_CLASSIFY_VERSION}_{digest}"


def ei_cache_key(
    base_dir: str | Path,
    session: int = 1,
    min_cluster_group: int = 2,
    animal_id: str | None = None,
    box_e: dict | None = None,
    box_i: dict | None = None,
) -> str:
    """
    Public: the exact classified-cache key a `get_population_responses` run would
    use for this session — session file stem + `min_cluster_group` + effective
    boxes + `_EI_CLASSIFY_VERSION`.  Resolves per-mouse boxes when not overridden.
    Used to stamp provenance into saved outputs (see `regenerate_population_rates`).
    """
    base_dir = Path(base_dir)
    animal_id = animal_id or base_dir.name
    if box_e is None or box_i is None:
        default_e, default_i = _mouse_ei_boxes(animal_id)
        box_e = default_e if box_e is None else box_e
        box_i = default_i if box_i is None else box_i
    _, kwx_path, _ = _find_session_files(base_dir / str(session), animal_id, session)
    return _ei_cache_key(kwx_path.stem, min_cluster_group, box_e, box_i)


def _load_population_spikes(
    kwik_path: Path,
    kwx_path: Path,
    min_cluster_group: int,
    box_e: dict | None = None,
    box_i: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load spikes with their E/I classification, using the paper's *clusterless*
    method — cluster labels are not trusted (fast-spiking I cells isolate poorly
    and get dumped into "noise"/MUA clusters, so filtering by cluster discards much
    of the I population).  Every spike is smoothed and kept unless it fails the
    waveform quality-control checks (`_waveform_qc`).

    `min_cluster_group` selects the neighbour pool (2 = good clusters only, our
    standing default; 0 = all spikes = the paper's clusterless pool).

    Returns
    -------
    time_samples : (n_spikes,) int64  — global sorted sample index (E/I only; spikes
                   outside both classification boxes are discarded — the paper keeps ~40%)
    ei_labels    : (n_spikes,) int8   — 0 = E (wide), 1 = I (narrow)
    """
    CACHE_DIR.mkdir(exist_ok=True)
    ei_path = CACHE_DIR / f"{_ei_cache_key(kwx_path.stem, min_cluster_group, box_e, box_i)}.npz"
    if ei_path.exists():
        print(f"ei: using cached {ei_path.name}", flush=True)
        d = np.load(ei_path)
        return d["time_samples"], d["ei_labels"]

    all_times: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    dat_path = kwx_path.with_suffix(".dat")
    with h5py.File(str(kwik_path), "r") as fk, h5py.File(str(kwx_path), "r") as fx:
        cg_keys = sorted(fk["channel_groups"].keys())
        max_t = max(int(fk[f"channel_groups/{cg}/spikes/time_samples"][()].max())
                    for cg in cg_keys)
        n_samples, n_channels = _infer_dat_shape(dat_path, max_t)

        for cg_key in cg_keys:
            spikes_grp = fk[f"channel_groups/{cg_key}/spikes"]
            times = spikes_grp["time_samples"][()]
            clusters = spikes_grp["clusters/main"][()]

            cls_meta = fk[f"channel_groups/{cg_key}/clusters/main"]
            cg_label = {
                int(k): int(cls_meta[k].attrs.get("cluster_group", 3))
                for k in cls_meta.keys()
            }
            mask = np.array([cg_label.get(int(c), 3) >= min_cluster_group
                             for c in clusters])
            times_m = times[mask]

            # Distance on PCA features; waveform denoised by averaging the *unfiltered*
            # neighbour snippets re-extracted WIDE from the .dat (the .kwx 20-sample
            # window truncates the E-defining features), then baseline-subtracted.
            features = fx[f"channel_groups/{cg_key}/features_masks"][:, :, 0][mask]
            filt = fx[f"channel_groups/{cg_key}/waveforms_filtered"][()][mask]
            cols = np.array(sorted(int(c) for c in fk[f"channel_groups/{cg_key}/channels"].keys()))

            cache_key = f"{kwx_path.stem}_cg{cg_key}_mcg{min_cluster_group}"
            nbr_idx, nbr_dist = _load_or_build_knn(cache_key, features)
            raw_wide = _load_or_extract_wide(cache_key, dat_path, n_samples, n_channels,
                                             times_m, cols)
            smoothed = _baseline_subtract(
                _clusterless_smooth_waveforms(raw_wide, nbr_idx, nbr_dist)
            )
            # QC on the .kwx-aligned window (first 20 samples = dat[t-10:t+10]).
            qc = _waveform_qc(_baseline_subtract(filt.astype(np.float32)),
                              smoothed[:, :20], nbr_dist)
            labels = _classify_ei_boxes(_spike_features(smoothed), box_e, box_i)  # 0=E,1=I,-1=discard
            keep = qc & (labels != -1)

            all_times.append(times_m[keep].astype(np.int64))
            all_labels.append(labels[keep])

    time_samples = np.concatenate(all_times)
    ei_labels = np.concatenate(all_labels)
    order = np.argsort(time_samples, kind="stable")
    time_samples, ei_labels = time_samples[order], ei_labels[order]
    np.savez(ei_path, time_samples=time_samples, ei_labels=ei_labels)
    return time_samples, ei_labels


def _load_protocol_conditions(
    protocol_path: Path,
    wl_to_pop: dict[int, str],
) -> dict[int, dict]:
    """
    Parse Protocol.mat and return a mapping:
        cond_id (1-indexed) → {pulse_type, ipi_ms, dur_ms, first_pop, second_pop, exp_type}

    `wl_to_pop` maps each pulse wavelength to E/I for this animal (see
    `_wavelength_to_pop`).  Returns an empty dict for non-TTL protocols (visual /
    regular-pulse), which have no `pulseType` parameter.  Conditions whose
    wavelength drives no opsin in this mouse line are skipped.
    """
    mat = scipy.io.loadmat(str(protocol_path), squeeze_me=True, struct_as_record=False)
    p = mat["Protocol"]
    parnames = list(p.parnames)
    if "pulseType" not in parnames:
        return {}
    pars = p.pars  # (n_params, n_conditions)

    pt_idx = parnames.index("pulseType")
    int_idx = parnames.index("intT")
    # Duration parameter: either a single `durT` (both pulses equal) or, in the
    # 2Dur variant, per-pulse `durT1`/`durT2` (durations stored as ms*10).
    if "durT" in parnames:
        dur1_idx = dur2_idx = parnames.index("durT")
    else:
        dur1_idx, dur2_idx = parnames.index("durT1"), parnames.index("durT2")

    conditions = {}
    for cond_i in range(p.npfilestimuli):
        cond_id = cond_i + 1   # 1-indexed to match mpepUDP events
        pt = int(pars[pt_idx, cond_i])
        ipi_ms = int(pars[int_idx, cond_i])
        d1 = float(pars[dur1_idx, cond_i]) / 10.0   # stored as ms*10
        d2 = float(pars[dur2_idx, cond_i]) / 10.0
        # Single pulse (intT==0): only durT1 is delivered — durT2 is a vestigial
        # default in the 2Dur variant, so drop it.  Paired: keep one value when
        # both durations match, else a (d1, d2) tuple.
        if ipi_ms == 0 or d1 == d2:
            dur_ms = d1
        else:
            dur_ms = (d1, d2)
        if pt not in _PULSE_TYPE_WAVELENGTHS:
            continue
        pops = _pulse_populations(pt, wl_to_pop)
        if pops is None:
            continue   # wavelength not drivable in this mouse line
        conditions[cond_id] = dict(
            pulse_type=pt,
            ipi_ms=ipi_ms,
            dur_ms=dur_ms,
            first_pop=pops[0],
            second_pop=pops[1],
            exp_type=_exp_type_key(pt, ipi_ms, wl_to_pop),
        )
    return conditions


# ---------------------------------------------------------------------------
# Public API: population-level responses
# ---------------------------------------------------------------------------

#: Example pulse-exp list for M150605 (single + paired opto).  The loader no
#: longer hardcodes this — it derives the list per session via
#: `_pulse_experiments`; kept only as a reference / for callers that want it.
DEFAULT_PULSE_EXPS = [2, 3, 4, 5, 6, 7, 13]

#: Protocol xfiles that are single/paired optogenetic pulse experiments (the E/I
#: analysis subset).  Excludes regular-pulse trains (`stimRegPulsesWave`) and
#: visual protocols (`ogl*`).
_OPTO_PULSE_XFILES = {
    "stim2Pulses2Waves.x",
    "stim2PulsesRandNoise.x",
    # Variant where the two pulses may have *different* durations (durT1/durT2
    # instead of a single durT).  Same E/I single/paired paradigm otherwise.
    "stim2Pulses2DurRandNoise.x",
}

#: All valid experiment-type keys, in a canonical order.
ALL_EXP_TYPES = ("single_E", "single_I", "paired_EE", "paired_II",
                 "paired_EI", "paired_IE", "flash")


def _pulse_experiments(
    session_dir: Path,
    all_exps: list[int],
    animal_id: str,
) -> list[int]:
    """
    Optogenetic single/paired pulse experiments for a session, from Protocol.

    Reads each experiment's `Protocol.mat` `xfile` and keeps those in
    `_OPTO_PULSE_XFILES`.  Replaces the hardcoded, M150605-specific
    `DEFAULT_PULSE_EXPS` so the loader works for any mouse's experiment layout.
    """
    out: list[int] = []
    for exp_num in all_exps:
        protocol_path = session_dir / str(exp_num) / "Protocol.mat"
        if not protocol_path.exists():
            continue
        proto = scipy.io.loadmat(
            str(protocol_path), squeeze_me=True, struct_as_record=False
        )["Protocol"]
        if str(getattr(proto, "xfile", "")) in _OPTO_PULSE_XFILES:
            out.append(int(exp_num))
    return out


def get_population_responses(
    base_dir: str | Path = "/mnt/scratch/M150605_ICTP1",
    session: int = 1,
    selected_exps: list[int] | None = None,
    pre_s: float = 0.5,
    post_s: float = 1.5,
    bin_s: float = 0.001,
    hamming_ms: float = 40.0,
    baseline_window: tuple[float, float] = (-0.5, -0.1),
    min_cluster_group: int = 2,
    animal_id: str | None = None,
    box_e: dict | None = None,
    box_i: dict | None = None,
    n_folds: int = 3,
    fold_seed: int = 0,
) -> dict[str, dict]:
    """
    Compute trial-averaged, baseline-normalised E/I population PSTHs grouped
    by experiment type and stimulus condition.

    Each condition's trials are randomly split into `n_folds` folds and averaged
    *within* each fold, so the output holds one PSTH per fold rather than a single
    all-trials mean.  This mirrors the paper's k-fold cross-validation (S1.10):
    downstream, train/test PSTHs are built by averaging held-in folds and holding
    out the remaining one.  Averaging the fold means (weighted by
    `n_trials_per_fold`) recovers the all-trials mean when needed.

    Population activity follows the paper's definition (Lin & Harris 2020,
    Supplemental Fig. S2): the total firing rate of all wide-spike (E) or
    narrow-spike (I) events detected by the probe, normalised to the mean
    rate in the baseline window and smoothed with a Hamming window.

    Parameters
    ----------
    base_dir : str or Path
    session : int
    selected_exps : list of int, optional
        Experiment numbers to include.  Default (None) derives the single/paired
        optogenetic pulse experiments for this session from each Protocol's xfile
        (`_pulse_experiments`), so it adapts to each mouse's layout — visual
        (`ogl*`) and regular-pulse (`stimRegPulsesWave`) experiments are excluded.
    pre_s, post_s : float
        Trial window around stimulus onset (seconds).
    bin_s : float
        Spike-count bin width (seconds).  Default 1 ms.  The paper bins at the
        1/30 ms sample period before smoothing; at the 40 ms Hamming window used
        here, 1 ms is empirically indistinguishable from 0.2 ms and finer, while
        5 ms visibly clips/mis-times the fast onset transient in short-IPI
        conditions.  The output keeps only 2 populations, so fine bins are cheap.
    hamming_ms : float
        Hamming smoothing window width (milliseconds).  Paper uses 40 ms.
    baseline_window : (float, float)
        Time interval (relative to onset) used for normalisation.
        Paper uses (–0.5, –0.1) s.
    min_cluster_group : int
        Neighbour-pool selection for the *clusterless* E/I split.  Default 2 keeps
        only good clusters.  0 keeps all spikes (the paper's clusterless method —
        noise rejected by `_waveform_qc`, not by cluster label); 1 = MUA+unsorted.
    box_e, box_i : dict, optional
        E/I classification boxes.  Default (None) resolves per-mouse boxes via
        `_mouse_ei_boxes(animal_id)` — the full four-metric per-mouse spec in
        `ei_classification_specification.py`.  Pass a dict to override explicitly.
    n_folds : int
        Number of cross-validation folds to split each condition's trials into.
        Default 3 (the paper's k for the mice we parse).  Folds are as even as
        possible; the seeded permutation makes the split deterministic.
    fold_seed : int
        Seed for the RNG that permutes trials before folding.  Fixed so the split
        is reproducible; stamped into saved outputs for provenance.

    Returns
    -------
    dict mapping experiment-type string → result dict, where result dict has:

        'responses' : ndarray (n_conditions, n_folds, 2, n_bins), float64
            Per-fold trial-averaged, Hamming-smoothed, baseline-normalised
            population activity.  Axis 2: 0 = E (wide), 1 = I (narrow).  A fold
            with no trials (condition with fewer trials than `n_folds`) is NaN.
        'conditions' : list of dicts, length n_conditions
            Per-condition stimulus parameters plus provenance:
              pulse_type, ipi_ms, dur_ms, first_pop, second_pop,
              n_trials, n_trials_per_fold
            n_trials is the total across folds; n_trials_per_fold is the list of
            per-fold trial counts (length n_folds), aligned to axis 1 of
            'responses'.
        'time_axis'  : ndarray (n_bins,)
            Bin centres in seconds relative to stimulus onset.

    Types with no trials in the selected experiments are omitted from the dict.
    'flash' is always absent for sessions without a visual-flash experiment.
    """
    base_dir = Path(base_dir)
    if animal_id is None:
        animal_id = base_dir.name
    session_dir = base_dir / str(session)

    # Per-mouse E/I boxes (full four-metric spec) unless the caller overrode a box.
    if box_e is None or box_i is None:
        default_e, default_i = _mouse_ei_boxes(animal_id)
        box_e = default_e if box_e is None else box_e
        box_i = default_i if box_i is None else box_i

    kwik_path, kwx_path, manifest_path = _find_session_files(
        session_dir, animal_id, session
    )

    # --- segment boundaries -------------------------------------------------
    manifest = scipy.io.loadmat(str(manifest_path), squeeze_me=True)
    lims: np.ndarray = manifest["lims"].astype(np.int64)
    all_exps: np.ndarray = manifest["SELECTED_EXPERIMENTS"].astype(int)
    exp_start_samples = np.concatenate([[0], np.cumsum(lims)])

    if selected_exps is None:
        selected_exps = _pulse_experiments(session_dir, list(all_exps), animal_id)
        print(f"pulse experiments (from Protocol xfiles): {selected_exps}")

    # --- load population spike trains (E and I) -----------------------------
    print("Loading and classifying spikes …")
    pop_times, pop_labels = _load_population_spikes(
        kwik_path, kwx_path, min_cluster_group, box_e, box_i
    )
    e_times = pop_times[pop_labels == 0]   # wide / excitatory
    i_times = pop_times[pop_labels == 1]   # narrow / inhibitory
    print(f"  {(pop_labels==0).sum():,} E spikes,  {(pop_labels==1).sum():,} I spikes")

    wl_to_pop = _wavelength_to_pop(animal_id)   # for per-condition E/I labels

    # --- time axis and Hamming kernel ---------------------------------------
    n_bins = int(round((pre_s + post_s) / bin_s))
    bin_edges = np.linspace(-pre_s, post_s, n_bins + 1)
    time_axis = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hamming_bins = max(1, int(round(hamming_ms / 1000.0 / bin_s)))
    if hamming_bins % 2 == 0:
        hamming_bins += 1   # keep odd so the window is symmetric
    kernel = signal_windows.hamming(hamming_bins)
    kernel /= kernel.sum()

    baseline_mask = (time_axis >= baseline_window[0]) & (time_axis < baseline_window[1])

    # Accumulator: exp_type → { condition_key → [list of (2, n_bins) spike-count arrays] }
    accum: dict[str, dict[tuple, list]] = {}

    # --- iterate experiments -------------------------------------------------
    for exp_i, exp_num in enumerate(all_exps):
        if exp_num not in selected_exps:
            continue

        trials = _load_experiment_trials(session_dir, int(exp_num), animal_id)
        if not trials:
            print(f"  exp {exp_num}: no laser trials, skipping")
            continue

        exp_start = exp_start_samples[exp_i]
        exp_end   = exp_start_samples[exp_i + 1]
        print(f"  exp {exp_num}: {len(trials)} trials")

        for trial in trials:
            exp_type = trial["exp_type"]
            onset_sample = exp_start + trial["onset_sample"]
            win_start = max(onset_sample - int(pre_s * SAMPLE_RATE_HZ), exp_start)
            win_end   = min(onset_sample + int(post_s * SAMPLE_RATE_HZ), exp_end)

            counts = np.zeros((2, n_bins), dtype=np.float32)
            for pop_idx, st in enumerate((e_times, i_times)):
                lo = int(np.searchsorted(st, win_start))
                hi = int(np.searchsorted(st, win_end))
                if lo < hi:
                    spike_t = (st[lo:hi].astype(np.float64) - onset_sample) / SAMPLE_RATE_HZ
                    bin_idx = ((spike_t + pre_s) / bin_s).astype(np.int32)
                    valid = (bin_idx >= 0) & (bin_idx < n_bins)
                    np.add.at(counts[pop_idx], bin_idx[valid], 1.0)

            # Condition key used to group trials: (pulse_type, ipi_ms, dur_ms)
            ckey = (trial["pulse_type"], trial["ipi_ms"], trial["dur_ms"])
            accum.setdefault(exp_type, {}).setdefault(ckey, []).append(counts)

    # --- fold, average, smooth, normalise -----------------------------------
    # One seeded RNG stream, consumed in a deterministic condition order, makes
    # the fold split reproducible from `fold_seed` alone.
    rng = np.random.default_rng(fold_seed)

    def _psth(counts_2xn: np.ndarray) -> np.ndarray:
        """Fold-mean spike counts (2, n_bins) -> smoothed, baseline-normalised."""
        rate = counts_2xn / bin_s                       # spikes / s
        smoothed = np.stack([
            np.convolve(rate[pop], kernel, mode="same")
            for pop in range(2)
        ])                                              # (2, n_bins)
        baseline_rate = smoothed[:, baseline_mask].mean(axis=1, keepdims=True)
        baseline_rate = np.maximum(baseline_rate, 1e-6)  # avoid /0
        return smoothed / baseline_rate

    output: dict[str, dict] = {}

    for exp_type in ALL_EXP_TYPES:
        if exp_type == "flash" or exp_type not in accum:
            continue

        cond_dict = accum[exp_type]
        # Sort conditions by (ipi_ms, dur_ms) for a consistent ordering.  dur_ms
        # (k[2]) may be a scalar or a (d1, d2) tuple; normalise to a tuple so
        # float-vs-tuple comparisons don't raise.
        sorted_keys = sorted(
            cond_dict.keys(),
            key=lambda k: (k[1], k[2] if isinstance(k[2], tuple) else (k[2],)),
        )

        responses_list = []
        conditions_list = []

        for ckey in sorted_keys:
            trial_counts = np.stack(cond_dict[ckey])   # (n_trials, 2, n_bins)
            n_trials = trial_counts.shape[0]

            # As-even-as-possible random split into n_folds (earlier folds absorb
            # the remainder).  Empty folds (n_trials < n_folds) -> NaN PSTH, count 0.
            perm = rng.permutation(n_trials)
            fold_idx = np.array_split(perm, n_folds)

            fold_psths = []
            n_per_fold = []
            for idx in fold_idx:
                n_per_fold.append(int(idx.size))
                if idx.size == 0:
                    fold_psths.append(np.full((2, n_bins), np.nan))
                    continue
                fold_psths.append(_psth(trial_counts[idx].mean(axis=0)))

            responses_list.append(np.stack(fold_psths))   # (n_folds, 2, n_bins)
            pt, ipi, dur = ckey
            first, second = _pulse_populations(pt, wl_to_pop)
            cond_meta = dict(pulse_type=pt, ipi_ms=ipi, dur_ms=dur,
                             first_pop=first, second_pop=second,
                             n_trials=n_trials,            # total across folds
                             n_trials_per_fold=n_per_fold)  # aligned to axis 1
            conditions_list.append(cond_meta)

        output[exp_type] = dict(
            responses=np.stack(responses_list),   # (n_cond, n_folds, 2, n_bins)
            conditions=conditions_list,
            time_axis=time_axis,
        )

    return output
