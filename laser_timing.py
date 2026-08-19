"""
laser_timing.py — recover laser-pulse onset times from Blackrock analog inputs.

Why this exists
---------------
Trial alignment must use a time base that lives in the SAME clock as the spikes
(the Blackrock 30 kHz recording behind the concatenated `.kwik`). The per-
experiment `Timeline.mat` `mpepUDP` StimStart times do NOT: they are logged by a
separate DAQ (the Timeline system) whose recording is started seconds apart from
Blackrock and ticks at a slightly different rate. Aligning spikes to Timeline
times therefore scrambles every trial-aligned analysis (flat PSTHs, invisible
raw traces, mis-aligned Fig 1C rasters).

The laser pulses ARE recorded in Blackrock's clock — as TTL markers on the
auxiliary analog inputs of the `.ns5`. For M150605_ICTP1 the `.ns5` holds 36
channels: 32 neural (IDs 1-32) + 4 analog inputs (IDs 129-132):

    ain 129  per-trial sawtooth ramp  (a Timeline sync signal)  -- NOT a laser
    ain 130  continuous square-wave clock                       -- NOT a laser
    ain 131  E laser TTL  (445 nm / blue -> excitatory)          -- pulse marker
    ain 132  I laser TTL  (561 nm / green -> inhibitory)         -- pulse marker

The laser channels sit at ~0 baseline and jump to a saturated ~21600 for the
pulse duration (0.5-4 ms). Detecting high-threshold rising edges on ain 131/132
gives onsets that produce sharp, short-latency (2-12 ms) evoked responses when
spikes are aligned to them — i.e. correct alignment.

Verified on exp 5 (single-pulse): ain131 -> 20 single_E, ain132 -> 20 single_I,
matched to Timeline trials with <=52 ms residual (the residual is exactly the
uncorrected Timeline/Blackrock clock drift). Note 10 of the 30 single_E trials
(all 1 ms) produced no analog TTL on either line — a real, still-unexplained
per-trial gap, so onset counts can be < Timeline trial counts; match by relative
timing rather than assuming a 1:1 count.

The `.nev` is NOT usable for this: it contains only online threshold-crossing
spike events (packet ids 1-32, all NEUEVWAV), with no digital-input packets.

Caveats
-------
- Which analog id carries E vs I is a per-mouse-line/session fact (see CLAUDE.md
  Table S1/S2 mapping); do not assume 131=E, 132=I for other animals.
- The `.ns5` here is the old NEURALSG (NSx 2.1) format: no per-channel labels,
  just channel IDs, then raw interleaved int16 with no data-packet headers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE_HZ: int = 30_000
_ANALOG_ID_MIN: int = 129  # Blackrock front-end: analog inputs are ids >= 129


def read_nsx_header(ns5_path: str | Path) -> dict:
    """
    Parse a NEURALSG (NSx 2.1) basic header.

    Returns dict with: label, period, fs, channel_ids (list), n_channels,
    header_bytes, n_samples.
    """
    p = Path(ns5_path)
    with open(p, "rb") as f:
        magic = f.read(8)
        if magic != b"NEURALSG":
            raise ValueError(f"{p.name}: expected NEURALSG (NSx 2.1), got {magic!r}")
        label = f.read(16).rstrip(b"\x00").decode("latin1")
        period = int(np.frombuffer(f.read(4), "<u4")[0])
        n_ch = int(np.frombuffer(f.read(4), "<u4")[0])
        ch_ids = np.frombuffer(f.read(4 * n_ch), "<u4").tolist()
    header_bytes = 8 + 16 + 4 + 4 + 4 * n_ch
    n_samples = (p.stat().st_size - header_bytes) // (2 * n_ch)
    return dict(
        label=label,
        period=period,
        fs=SAMPLE_RATE_HZ / period,
        channel_ids=ch_ids,
        n_channels=n_ch,
        header_bytes=header_bytes,
        n_samples=n_samples,
    )


def open_nsx(ns5_path: str | Path) -> tuple[np.memmap, dict]:
    """Memory-map an NSx 2.1 file as (n_samples, n_channels) int16, + header."""
    hdr = read_nsx_header(ns5_path)
    data = np.memmap(
        ns5_path,
        dtype="<i2",
        mode="r",
        offset=hdr["header_bytes"],
        shape=(hdr["n_samples"], hdr["n_channels"]),
    )
    return data, hdr


def _ttl_onsets(trace: np.ndarray, threshold: float, refractory_s: float) -> np.ndarray:
    """Rising-edge sample indices of a TTL, with a refractory to drop ringing."""
    high = np.asarray(trace, dtype=np.int32) > threshold
    rises = np.where((~high[:-1]) & (high[1:]))[0] + 1
    if rises.size == 0:
        return rises
    keep = [int(rises[0])]
    min_gap = refractory_s * SAMPLE_RATE_HZ
    for r in rises[1:]:
        if r - keep[-1] > min_gap:
            keep.append(int(r))
    return np.asarray(keep, dtype=np.int64)


def laser_onsets(
    ns5_path: str | Path,
    threshold: float = 2000.0,
    refractory_s: float = 0.05,
) -> dict[int, np.ndarray]:
    """
    Detect laser-pulse onsets (experiment-local samples) per analog channel.

    Returns { channel_id: onset_samples } for analog channels (id >= 129) that
    look TTL-like: sitting near a low baseline with sparse saturating pulses.
    Sync/clock analog channels (continuous or ramping) are excluded. Map each
    returned channel id to E/I using the session's wavelength assignment.
    """
    data, hdr = open_nsx(ns5_path)
    out: dict[int, np.ndarray] = {}
    for col, chid in enumerate(hdr["channel_ids"]):
        if chid < _ANALOG_ID_MIN:
            continue
        x = np.asarray(data[:, col], dtype=np.int32)
        # TTL-like = sits at a stable baseline almost all the time, with rare
        # saturating pulses. The pulses are so sparse (tens of ~1-2 ms events
        # over minutes) that even the 1st/99th percentiles are still baseline,
        # so detect the pulse via the extreme (max - median) while requiring the
        # bulk of samples (p1..p99) to occupy a tiny band. This rejects the
        # sawtooth-ramp and continuous-square sync channels, whose p1..p99 span
        # a large fraction of their full range.
        lo, med, hi = np.percentile(x, [1, 50, 99])
        full = int(x.max()) - int(x.min())
        if full < 5000:
            continue
        if (int(x.max()) - med) < 5000:      # no saturating pulse above baseline
            continue
        if (hi - lo) > 0.1 * full:           # baseline not stable -> sync/clock
            continue
        onsets = _ttl_onsets(x, threshold=threshold, refractory_s=refractory_s)
        if onsets.size:
            out[int(chid)] = onsets
    return out


def experiment_trial_onsets(
    ns5_path: str | Path,
    iti_gap_s: float = 1.0,
    **detect_kwargs,
) -> np.ndarray:
    """
    First-pulse onset (experiment-local samples) of each delivered trial.

    Pools rising edges across all detected laser channels and splits the train
    at gaps larger than `iti_gap_s`: within-trial pulses (paired-pulse interval
    <= 800 ms) stay together, while the >1.6 s inter-trial interval starts a new
    trial. The first edge of each group is that trial's onset.
    """
    chans = laser_onsets(ns5_path, **detect_kwargs)
    if not chans:
        return np.empty(0, dtype=np.int64)
    allon = np.sort(np.concatenate(list(chans.values())))
    if allon.size == 0:
        return allon
    gaps_s = np.diff(allon) / SAMPLE_RATE_HZ
    starts = np.concatenate([[0], np.where(gaps_s > iti_gap_s)[0] + 1])
    return allon[starts]


def find_ns5(session_dir: Path, exp_num: int, animal_id: str) -> Path | None:
    """Locate the .ns5 for one experiment (skip macOS AppleDouble sidecars)."""
    hits = sorted(
        p for p in (session_dir / str(exp_num)).glob("*.ns5")
        if not p.name.startswith("._")
    )
    return hits[0] if hits else None
