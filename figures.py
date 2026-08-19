"""
figures.py — reproductions of figures from Lin & Harris (2020).

Each function pulls already-parsed timing/spike data (via `data_loader` and
`laser_timing`) and renders one paper figure, as evidence that our reading of
the file format is correct.

So far
------
plot_raw_traces_around_pulse()
    Fig S1.B — raw extracellular traces across N recording sites in a short
    window around a single laser pulse, with the pulse marked.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import data_loader as dl
import laser_timing as lt


def plot_raw_traces_around_pulse(
    base_dir: str | Path,
    exp_num: int,
    dur_ms: float = 2.0,
    exp_type: str = "single_E",
    n_sites: int = 16,
    pre_ms: float = 25.0,
    post_ms: float = 200.0,
    trial_index: int = 0,
    animal_id: str | None = None,
    session: int = 1,
    ax: plt.Axes | None = None,
):
    """
    Fig S1.B-style plot: raw ephys traces across sites around one laser pulse.

    Timing and raw voltage share the Blackrock 30 kHz clock, so the trial window
    is a plain slice of the `.ns5`: the pulse onset comes from the analog laser
    TTL (`data_loader._load_experiment_trials` -> `onset_sample`, experiment-local
    Blackrock samples) and the raw traces come from the same file's neural
    channels (`laser_timing.open_nsx`).  No cross-clock conversion is involved.

    Parameters
    ----------
    base_dir : str or Path
        Session root (its name is used as animal_id unless given).
    exp_num : int
        Experiment number to draw the pulse from (must be a laser experiment).
    dur_ms : float
        Pulse duration to match (e.g. 2.0).  The red bar spans this width.
    exp_type : str
        Which condition family to pull the trial from ('single_E', 'single_I',
        'paired_EE', ...).
    n_sites : int
        Number of neural channels (recording sites) to show, from the top.
    pre_ms, post_ms : float
        Window before/after pulse onset (defaults give the paper's ~225 ms).
    trial_index : int
        Which matching trial to plot (0 = first).
    ax : matplotlib Axes, optional
        Draw into this axis; a new figure/axis is created if omitted.

    Returns
    -------
    ax : matplotlib Axes
    trial : dict
        The plotted trial's metadata (from `_load_experiment_trials`).

    Notes
    -----
    Sites are shown in acquisition-channel order, NOT verified probe depth.  To
    match the paper's depth layout, reorder the neural columns by the probe
    geometry (`CHANNELS_ORDER` / `SELECTED_CHANNELS` in the manifest `.mat`).
    """
    base_dir = Path(base_dir)
    animal_id = animal_id or base_dir.name
    session_dir = base_dir / str(session)
    fs = lt.SAMPLE_RATE_HZ

    # 1. locate a matching trial -> E-pulse onset (experiment-local samples)
    trials = [
        t for t in dl._load_experiment_trials(session_dir, exp_num, animal_id)
        if t["exp_type"] == exp_type and abs(t["dur_ms"] - dur_ms) < 0.1
    ]
    if not trials:
        raise RuntimeError(
            f"no {exp_type} dur={dur_ms} ms trial found in exp {exp_num}"
        )
    trial = trials[trial_index]
    onset = trial["onset_sample"]

    # 2. raw traces from the .ns5 neural channels (ids < 129 are neural)
    ns5 = lt.find_ns5(session_dir, exp_num, animal_id)
    data, hdr = lt.open_nsx(ns5)
    neural_cols = [i for i, cid in enumerate(hdr["channel_ids"]) if cid < 129][:n_sites]
    pre_n = int(pre_ms / 1000 * fs)
    lo, hi = onset - pre_n, onset + int(post_ms / 1000 * fs)
    seg = np.asarray(data[lo:hi, :][:, neural_cols], dtype=np.float32)  # (n, n_sites)
    t_ms = (np.arange(seg.shape[0]) - pre_n) / fs * 1000

    # 3. stacked traces, one row per site, with a red bar over the pulse
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))
    # Offset each row by a robust multiple of the median absolute deviation so
    # traces are separated without clipping into each other.
    offset = 6 * np.median(np.abs(seg - np.median(seg, axis=0)))
    for j in range(len(neural_cols)):
        ax.plot(t_ms, seg[:, j] - np.median(seg[:, j]) - j * offset,
                color="k", lw=0.4)
    ax.axvspan(0, dur_ms, color="red", alpha=0.9, lw=0)   # laser pulse
    ax.annotate("", xy=(0, offset * 0.5), xytext=(0, offset * 1.8),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.set_yticks([])
    ax.set_xlabel("Time from pulse onset (ms)")
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_title(
        f"Raw traces, {n_sites} sites, {dur_ms}-ms {exp_type} pulse "
        f"(exp {exp_num}, {int(pre_ms + post_ms)} ms)"
    )
    return ax, trial
