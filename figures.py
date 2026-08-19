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
plot_fw3m_features()
    FW3M distribution, single-spike (unsmoothed) vs clusterless-smoothed — shows
    the denoising sharpening the waveform-width measurement.
plot_s2c()
    Fig S2.C both panels (FW3M vs duration; late vs early gradient) from the wide
    `.dat`-re-extracted, clusterless-smoothed waveforms, with the E/I boxes drawn.
plot_population_rates()
    E/I population PSTHs per experiment type, from a saved
    `population_rates_*.npz` (as produced by `data_loader.get_population_responses`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import data_loader as dl
import laser_timing as lt

import h5py


def _smoothed_cg(
    base_dir: str | Path,
    cg: str = "0",
    min_cluster_group: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    animal_id: str | None = None,
    session: int = 1,
) -> dict[str, np.ndarray]:
    """
    Wide raw snippets + their clusterless-smoothed, baseline-subtracted waveforms
    for one shank — the shared wide-`.dat` load behind `smoothed_features_cg`,
    `plot_s2c` and `plot_fw3m_features`.

    Mirrors `data_loader._load_population_spikes`: cached k-NN on the PCA features →
    average the *wide* raw neighbour snippets re-extracted from the `.dat` → baseline
    subtract.  `sample_size` restricts the queries to a random subset (the full pool
    is still searched for each one's neighbours); None = all spikes.

    Returns dict aligned to the queries:
      wide     (nq, W, C) int16   raw wide query snippets
      smoothed (nq, W, C) float32 baseline-subtracted smoothed waveforms
      filt     (nq, 20, C) int16  `.kwx` filtered snippets (QC target)
      nbr_dist (nq, k)  float32   neighbour feature distances
    """
    base_dir = Path(base_dir)
    animal_id = animal_id or base_dir.name
    session_dir = base_dir / str(session)
    kwik_path, kwx_path, _ = dl._find_session_files(session_dir, animal_id, session)
    dat_path = kwx_path.with_suffix(".dat")

    with h5py.File(str(kwik_path), "r") as fk, h5py.File(str(kwx_path), "r") as fx:
        clusters = fk[f"channel_groups/{cg}/spikes/clusters/main"][()]
        meta = fk[f"channel_groups/{cg}/clusters/main"]
        grp = {int(k): int(meta[k].attrs.get("cluster_group", 3)) for k in meta.keys()}
        mask = np.array([grp.get(int(c), 3) >= min_cluster_group for c in clusters])
        times = fk[f"channel_groups/{cg}/spikes/time_samples"][()][mask]
        max_t = int(fk[f"channel_groups/{cg}/spikes/time_samples"][()].max())
        features = fx[f"channel_groups/{cg}/features_masks"][:, :, 0][mask]
        filt = fx[f"channel_groups/{cg}/waveforms_filtered"][()][mask]
        cols = np.array(sorted(int(c) for c in fk[f"channel_groups/{cg}/channels"].keys()))

    n_samples, n_channels = dl._infer_dat_shape(dat_path, max_t)
    cache_key = f"{kwx_path.stem}_cg{cg}_mcg{min_cluster_group}"
    nbr_idx, nbr_dist = dl._load_or_build_knn(cache_key, features)
    wide = dl._load_or_extract_wide(cache_key, dat_path, n_samples, n_channels, times, cols)

    if sample_size is not None and sample_size < len(nbr_idx):
        q = np.random.default_rng(seed).choice(len(nbr_idx), sample_size, replace=False)
    else:
        q = np.arange(len(nbr_idx))

    smoothed = dl._baseline_subtract(dl._clusterless_smooth_waveforms(wide, nbr_idx[q], nbr_dist[q]))
    return dict(wide=wide[q], smoothed=smoothed, filt=filt[q], nbr_dist=nbr_dist[q])


def smoothed_features_cg(
    base_dir: str | Path,
    cg: str = "0",
    min_cluster_group: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    animal_id: str | None = None,
    session: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Clusterless four-feature measurement + E/I labels for one shank (wide `.dat` path).

    Loads the wide smoothed waveforms via `_smoothed_cg`, then applies
    `data_loader._spike_features` and `_classify_ei_boxes`.  `sample_size` runs a
    random subset of queries (full pool still searched); None = all spikes.

    Returns (features, labels).  labels: 0=E, 1=I, -1=discard.
    """
    s = _smoothed_cg(base_dir, cg=cg, min_cluster_group=min_cluster_group,
                     sample_size=sample_size, seed=seed, animal_id=animal_id, session=session)
    feats = dl._spike_features(s["smoothed"])
    labels = dl._classify_ei_boxes(feats)
    return feats, labels


def plot_fw3m_features(
    base_dir: str | Path,
    cg: str = "0",
    sample_size: int | None = None,
    axes: np.ndarray | None = None,
    **kw,
):
    """
    Show the FW3M denoising effect for one shank (wide `.dat` path): the
    single-spike (unsmoothed) FW3M distribution vs the clusterless-smoothed one,
    over the QC-passing spikes.  FW3M is measured on the trough, so it is well
    defined even where trough-to-peak is not.

    (For the full S2.C E/I feature space use `plot_s2c`.)

    Returns (axes, dict of fw3m_raw / fw3m_smoothed arrays in ms).
    """
    s = _smoothed_cg(base_dir, cg=cg, sample_size=sample_size, **kw)
    qc = dl._waveform_qc(dl._baseline_subtract(s["filt"].astype(np.float32)),
                         s["smoothed"][:, :20], s["nbr_dist"])
    raw_dom = dl._dominant_trace(dl._baseline_subtract(s["wide"].astype(np.float32)))
    fw3m_raw = dl._fw3m_ms(raw_dom)[qc]
    fw3m_sm = dl._fw3m_ms(dl._dominant_trace(s["smoothed"]))[qc]

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    bins = np.linspace(0, 0.5, 70)
    axes[0].hist(fw3m_raw[~np.isnan(fw3m_raw)], bins=bins, alpha=0.5,
                 density=True, label="raw (single spike)")
    axes[0].hist(fw3m_sm[~np.isnan(fw3m_sm)], bins=bins, alpha=0.5,
                 density=True, label="clusterless-smoothed")
    axes[0].set(title="FW3M: smoothing sharpens", xlabel="FW3M (ms)", ylabel="density")
    axes[0].legend()
    axes[1].hist(fw3m_sm[~np.isnan(fw3m_sm)], bins=bins)
    axes[1].set(title=f"FW3M smoothed (QC-kept {qc.mean()*100:.0f}%)",
                xlabel="FW3M (ms)", ylabel="count")
    return axes, dict(fw3m_raw=fw3m_raw, fw3m_smoothed=fw3m_sm)


def plot_s2c(
    base_dir: str | Path,
    cg: str = "0",
    sample_size: int | None = None,
    axes: np.ndarray | None = None,
    **kw,
):
    """
    Reproduce Fig S2.C (both panels) for one shank, with the E/I boxes drawn.
    Returns (axes, feats, labels).  Pass `sample_size` to run on a random subset.
    """
    feats, labels = smoothed_features_cg(base_dir, cg=cg, sample_size=sample_size, **kw)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colour = {0: "r", 1: "b", -1: "0.6"}
    for L in (-1, 0, 1):
        m = labels == L
        axes[0].scatter(feats["duration"][m], feats["fw3m"][m], s=2, c=colour[L], alpha=.2, lw=0)
        axes[1].scatter(feats["early"][m], feats["late"][m], s=2, c=colour[L], alpha=.2, lw=0)
    for box, c in [(dl.EI_BOX_E, "r"), (dl.EI_BOX_I, "b")]:
        (d0, d1), (f0, f1) = box["duration"], box["fw3m"]
        axes[0].add_patch(plt.Rectangle((d0, f0), d1 - d0, f1 - f0, fill=False, ec=c, lw=1.5))
    axes[0].set(xlim=(0, 1.0), ylim=(0, 0.4), xlabel="spike duration (ms)",
                ylabel="FW3M (ms)", title="S2.C top")
    axes[1].axvline(dl.EI_GRAD_EARLY_SPLIT, color="k", lw=.6)
    axes[1].axhline(dl.EI_GRAD_LATE_SPLIT, color="k", lw=.6)
    axes[1].set(xlim=(0, 600), ylim=(-150, 150), xlabel="early grad (uV/ms)",
                ylabel="late grad (uV/ms)", title="S2.C bottom")
    kept = np.mean(labels != -1) * 100
    axes[0].text(0.02, 0.95, f"E {np.mean(labels==0)*100:.0f}%  I {np.mean(labels==1)*100:.0f}%"
                 f"  kept {kept:.0f}%", transform=axes[0].transAxes, va="top", fontsize=9)
    return axes, feats, labels


def plot_population_rates(
    npz_path: str | Path,
    condition: int = 0,
    types: list[str] | None = None,
    axes: np.ndarray | None = None,
):
    """
    Plot E (wide) and I (narrow) population PSTHs for one stimulus condition of each
    experiment type, from a `population_rates_*.npz` saved by
    `data_loader.get_population_responses` (keys `{type}__responses` of shape
    (n_conditions, 2, n_bins), `{type}__time_axis`).

    Rates are trial-averaged, Hamming-smoothed and baseline-normalised (~1 at rest).

    Parameters
    ----------
    condition : int
        Which stimulus condition (row of `responses`) to draw for each type.

    Returns
    -------
    axes : np.ndarray of matplotlib Axes
    """
    d = np.load(str(npz_path), allow_pickle=True)
    if types is None:
        types = sorted({k.split("__")[0] for k in d.files})
    if axes is None:
        ncol = 3
        nrow = int(np.ceil(len(types) / ncol))
        _, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.4 * nrow),
                               squeeze=False)
    flat = np.ravel(axes)
    for ax, t in zip(flat, types):
        resp = d[f"{t}__responses"]                 # (n_cond, 2, n_bins)
        tax = d[f"{t}__time_axis"]
        c = min(condition, resp.shape[0] - 1)
        ax.plot(tax, resp[c, 0], "r", label="E (wide)")
        ax.plot(tax, resp[c, 1], "b", label="I (narrow)")
        ax.axvline(0, color="k", lw=0.5)
        ax.set(title=f"{t} (cond {c} of {resp.shape[0]})",
               xlabel="time from onset (s)", ylabel="norm. rate")
        ax.legend(fontsize=7)
    for ax in flat[len(types):]:
        ax.set_visible(False)
    return axes


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
