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

import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from scipy.signal import windows as signal_windows

import laser_timing

SAMPLE_RATE_HZ: int = 30_000  # Blackrock / kwik spike-sample rate
SERIES: int = 1               # always 1 for this animal/session


def _find_session_files(
    session_dir: Path,
    animal_id: str,
    session: int,
) -> tuple[Path, Path, Path]:
    """
    Locate manifest, kwik, and kwx files for a session.

    Manifest (*.mat) and kwik/kwx may live in session_dir or _klustakwik/.
    Prefers a 'not*' manifest over 'V1' when both exist, because the not*
    variant is the one that feeds SpikeDetekt.

    Returns (kwik_path, kwx_path, manifest_path).
    """
    not_mats = sorted(session_dir.glob(f"{animal_id}_s{session}_not*.mat"))
    v1_mats  = sorted(session_dir.glob(f"{animal_id}_s{session}_V1.mat"))
    candidates = not_mats or v1_mats
    if not candidates:
        raise FileNotFoundError(
            f"No manifest MAT found in {session_dir} for {animal_id} s{session}"
        )
    manifest_path = candidates[0]
    stem = manifest_path.stem   # e.g. M150605_ICTP1_s1_not9

    for search_dir in (session_dir, session_dir / "_klustakwik"):
        kwik = search_dir / f"{stem}.kwik"
        if kwik.exists():
            kwx = search_dir / f"{stem}.kwx"
            return kwik, kwx, manifest_path

    raise FileNotFoundError(
        f"kwik file '{stem}.kwik' not found in {session_dir} or _klustakwik/"
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
    cond_map = _load_protocol_conditions(protocol_path)

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
    dur_s = trial["dur_ms"] / 1000.0

    def mark(t0_s: float) -> None:
        lo = int(round((pre_s + t0_s) / bin_s))
        hi = int(round((pre_s + t0_s + dur_s) / bin_s))
        hi = max(hi, lo + 1)   # a sub-bin pulse (< bin_s) still marks one bin
        s[max(lo, 0):min(hi, n_bins)] = 1.0

    mark(0.0)
    if trial["ipi_ms"] > 0:
        mark(trial["ipi_ms"] / 1000.0)
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

# Maps Protocol pulseType integer to (first_pop, second_pop) labels.
# B = Blue = 445 nm → excitatory (E); G = Green = 561 nm → inhibitory (I).
_PULSE_TYPE_MAP = {
    1: ("E", "E"),   # BB
    2: ("I", "I"),   # GG
    3: ("E", "I"),   # BG
    4: ("I", "E"),   # GB
}

# Experiment-type key derived from (first_pop, second_pop, is_paired).
def _exp_type_key(pulse_type: int, ipi_ms: int) -> str:
    first, second = _PULSE_TYPE_MAP[pulse_type]
    if ipi_ms == 0:
        return f"single_{first}"
    return f"paired_{first}{second}"


def _classify_waveforms(waveforms: np.ndarray, threshold_ms: float) -> np.ndarray:
    """
    Classify each spike as wide/E (0) or narrow/I (1) from its filtered waveform.

    Parameters
    ----------
    waveforms : (n_spikes, n_samples, n_channels) int16
    threshold_ms : float
        Trough-to-peak time threshold.  Spikes with trough-to-peak >= threshold
        are classified as wide (E=0); those below as narrow (I=1).

    Returns
    -------
    labels : (n_spikes,) int8   0 = E, 1 = I
    """
    n_spikes, n_samples, n_channels = waveforms.shape
    wav = waveforms.astype(np.float32)

    # Dominant channel = channel with largest peak-to-trough amplitude
    pk_to_tr = wav.max(axis=1) - wav.min(axis=1)          # (n, n_ch)
    dom_ch = pk_to_tr.argmax(axis=1)                       # (n,)
    dom_wav = wav[np.arange(n_spikes), :, dom_ch]          # (n, n_samp)

    # Trough index (most negative point)
    trough_idx = dom_wav.argmin(axis=1)                    # (n,)

    # Peak index = first maximum AFTER the trough
    sample_grid = np.arange(n_samples)[np.newaxis, :]      # (1, n_samp)
    after_trough = sample_grid >= trough_idx[:, np.newaxis] # (n, n_samp)
    masked = np.where(after_trough, dom_wav, -np.inf)
    peak_idx = masked.argmax(axis=1)                       # (n,)

    trough_to_peak_ms = (peak_idx - trough_idx) / (SAMPLE_RATE_HZ / 1000.0)
    return (trough_to_peak_ms < threshold_ms).astype(np.int8)


def _load_population_spikes(
    kwik_path: Path,
    kwx_path: Path,
    ei_threshold_ms: float,
    min_cluster_group: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all non-noise spikes with their E/I classification.

    Returns
    -------
    time_samples : (n_spikes,) int64  — global sorted sample index
    ei_labels    : (n_spikes,) int8   — 0 = E (wide), 1 = I (narrow)
    """
    all_times: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with h5py.File(str(kwik_path), "r") as fk, h5py.File(str(kwx_path), "r") as fx:
        for cg_key in sorted(fk["channel_groups"].keys()):
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

            waveforms = fx[f"channel_groups/{cg_key}/waveforms_filtered"][()]
            labels = _classify_waveforms(waveforms, ei_threshold_ms)

            all_times.append(times[mask].astype(np.int64))
            all_labels.append(labels[mask])

    time_samples = np.concatenate(all_times)
    ei_labels = np.concatenate(all_labels)
    order = np.argsort(time_samples, kind="stable")
    return time_samples[order], ei_labels[order]


def _load_protocol_conditions(protocol_path: Path) -> dict[int, dict]:
    """
    Parse Protocol.mat and return a mapping:
        cond_id (1-indexed) → {pulse_type, ipi_ms, dur_ms, exp_type}

    Returns an empty dict for non-TTL protocols (visual / regular-pulse), which
    have no `pulseType` parameter.
    """
    mat = scipy.io.loadmat(str(protocol_path), squeeze_me=True, struct_as_record=False)
    p = mat["Protocol"]
    parnames = list(p.parnames)
    if "pulseType" not in parnames:
        return {}
    pars = p.pars  # (n_params, n_conditions)

    pt_idx = parnames.index("pulseType")
    int_idx = parnames.index("intT")
    dur_idx = parnames.index("durT")

    conditions = {}
    for cond_i in range(p.npfilestimuli):
        cond_id = cond_i + 1   # 1-indexed to match mpepUDP events
        pt = int(pars[pt_idx, cond_i])
        ipi_ms = int(pars[int_idx, cond_i])
        dur_ms = float(pars[dur_idx, cond_i]) / 10.0   # stored as ms*10
        if pt not in _PULSE_TYPE_MAP:
            continue
        conditions[cond_id] = dict(
            pulse_type=pt,
            ipi_ms=ipi_ms,
            dur_ms=dur_ms,
            exp_type=_exp_type_key(pt, ipi_ms),
        )
    return conditions


# ---------------------------------------------------------------------------
# Public API: population-level responses
# ---------------------------------------------------------------------------

#: Experiments to include by default (TTL pulse experiments only; excludes
#: visual characterisation exps 1 & 14, and regular-pulse exps 8, 10–12).
DEFAULT_PULSE_EXPS = [2, 3, 4, 5, 6, 7, 13]

#: All valid experiment-type keys, in a canonical order.
ALL_EXP_TYPES = ("single_E", "single_I", "paired_EE", "paired_II",
                 "paired_EI", "paired_IE", "flash")


def get_population_responses(
    base_dir: str | Path = "/mnt/scratch/M150605_ICTP1",
    session: int = 1,
    selected_exps: list[int] | None = None,
    pre_s: float = 0.5,
    post_s: float = 1.5,
    bin_s: float = 0.005,
    hamming_ms: float = 40.0,
    baseline_window: tuple[float, float] = (-0.5, -0.1),
    ei_threshold_ms: float = 0.4,
    min_cluster_group: int = 1,
    animal_id: str | None = None,
) -> dict[str, dict]:
    """
    Compute trial-averaged, baseline-normalised E/I population PSTHs grouped
    by experiment type and stimulus condition.

    Population activity follows the paper's definition (Lin & Harris 2020,
    Supplemental Fig. S2): the total firing rate of all wide-spike (E) or
    narrow-spike (I) events detected by the probe, normalised to the mean
    rate in the baseline window and smoothed with a Hamming window.

    Parameters
    ----------
    base_dir : str or Path
    session : int
    selected_exps : list of int, optional
        Experiment numbers to include.  Defaults to the TTL pulse experiments
        [2, 3, 4, 5, 6, 7, 13].  Experiments 1 & 14 are visual tuning checks;
        8, 10–12 are regular-pulse experiments with a different paradigm.
    pre_s, post_s : float
        Trial window around stimulus onset (seconds).
    bin_s : float
        Spike-count bin width (seconds).
    hamming_ms : float
        Hamming smoothing window width (milliseconds).  Paper uses 40 ms.
    baseline_window : (float, float)
        Time interval (relative to onset) used for normalisation.
        Paper uses (–0.5, –0.1) s.
    ei_threshold_ms : float
        Trough-to-peak duration threshold separating wide (E) from narrow (I)
        spikes.  Inspect the bimodal waveform-width histogram for your session
        before accepting the default 0.4 ms.
    min_cluster_group : int
        Minimum KlustaKwik cluster group to include (0 = noise excluded by
        default, 1 = MUA+unsorted, 2 = good only).

    Returns
    -------
    dict mapping experiment-type string → result dict, where result dict has:

        'responses' : ndarray (n_conditions, 2, n_bins), float64
            Trial-averaged, Hamming-smoothed, baseline-normalised population
            activity.  Axis 1: 0 = E (wide), 1 = I (narrow).
        'conditions' : list of dicts, length n_conditions
            Per-condition stimulus parameters:
              pulse_type, ipi_ms, dur_ms  (plus ipi2_ms for paired types)
        'time_axis'  : ndarray (n_bins,)
            Bin centres in seconds relative to stimulus onset.

    Types with no trials in the selected experiments are omitted from the dict.
    'flash' is always absent for sessions without a visual-flash experiment.
    """
    base_dir = Path(base_dir)
    if animal_id is None:
        animal_id = base_dir.name
    session_dir = base_dir / str(session)

    kwik_path, kwx_path, manifest_path = _find_session_files(
        session_dir, animal_id, session
    )

    if selected_exps is None:
        selected_exps = DEFAULT_PULSE_EXPS

    # --- segment boundaries -------------------------------------------------
    manifest = scipy.io.loadmat(str(manifest_path), squeeze_me=True)
    lims: np.ndarray = manifest["lims"].astype(np.int64)
    all_exps: np.ndarray = manifest["SELECTED_EXPERIMENTS"].astype(int)
    exp_start_samples = np.concatenate([[0], np.cumsum(lims)])

    # --- load population spike trains (E and I) -----------------------------
    print("Loading and classifying spikes …")
    pop_times, pop_labels = _load_population_spikes(
        kwik_path, kwx_path, ei_threshold_ms, min_cluster_group
    )
    e_times = pop_times[pop_labels == 0]   # wide / excitatory
    i_times = pop_times[pop_labels == 1]   # narrow / inhibitory
    print(f"  {(pop_labels==0).sum():,} E spikes,  {(pop_labels==1).sum():,} I spikes")

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

    # --- average, smooth, normalise -----------------------------------------
    output: dict[str, dict] = {}

    for exp_type in ALL_EXP_TYPES:
        if exp_type == "flash" or exp_type not in accum:
            continue

        cond_dict = accum[exp_type]
        # Sort conditions by (ipi_ms, dur_ms) for a consistent ordering
        sorted_keys = sorted(cond_dict.keys(), key=lambda k: (k[1], k[2]))

        responses_list = []
        conditions_list = []

        for ckey in sorted_keys:
            trial_counts = np.stack(cond_dict[ckey])   # (n_trials, 2, n_bins)
            mean_counts = trial_counts.mean(axis=0)     # (2, n_bins)
            rate = mean_counts / bin_s                  # spikes / s

            # Hamming smoothing per population
            smoothed = np.stack([
                np.convolve(rate[pop], kernel, mode="same")
                for pop in range(2)
            ])  # (2, n_bins)

            # Baseline normalisation
            baseline_rate = smoothed[:, baseline_mask].mean(axis=1, keepdims=True)
            baseline_rate = np.maximum(baseline_rate, 1e-6)  # avoid /0
            normalised = smoothed / baseline_rate

            responses_list.append(normalised)
            pt, ipi, dur = ckey
            first, second = _PULSE_TYPE_MAP[pt]
            cond_meta = dict(pulse_type=pt, ipi_ms=ipi, dur_ms=dur,
                             first_pop=first, second_pop=second)
            conditions_list.append(cond_meta)

        output[exp_type] = dict(
            responses=np.stack(responses_list),   # (n_cond, 2, n_bins)
            conditions=conditions_list,
            time_axis=time_axis,
        )

    return output
