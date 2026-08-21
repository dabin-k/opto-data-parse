"""
E/I classification diagnostics — why we do or don't capture spikes of each class.

Waveforms are the clusterless-smoothed, baseline-subtracted *unfiltered* wide
snippets (dominant channel), shown in uV (ADC * UV_PER_ADC).

`plot_metric_distributions(name, base)` — one figure per mouse: histograms of the
  four classification metrics (duration, FW3M, trough grad, peak grad) with the E
  box (red) and I box (blue) ranges shaded, to see the distributions vs the cuts.

`plot_spikes_by_class(name, base)` — grid of example spikes grouped by class
  (putative E / I / neither), each with its smoothed waveform and colour-coded
  feature legend, to see *why* individual E-like spikes are or aren't kept.

Run: PYTHONPATH=. python ei_classification_diag.py
"""
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import data_loader as dl
import figures

FS = dl.SAMPLE_RATE_HZ / 1000.0          # samples per ms
PRE_MS = 0.33                             # always show this much before the trough
LABEL_COLOR = {"putative E": "r", "putative I": "b", "neither": "0.5"}
MICE = {"M150605": "/mnt/scratch/IChunData4Dabin/M150605_ICTP1",
        "M150609A": "/mnt/scratch/IChunData4Dabin/M150609_ICTP1",
        "M150609B": "/mnt/scratch/IChunData4Dabin/M150609_ICTP2",
        "M151020": "/mnt/scratch/IChunData4Dabin/M151020_ICTP1",
        }

def _uv(dw):
    """Dominant smoothed trace (ADC) -> uV."""
    return dw * dl.UV_PER_ADC


# The four classification features: (feature key, axis label, plot x-range).
METRICS = [("duration", "spike duration (ms)", (0.0, 1.2)),
           ("fw3m",     "FW3M (ms)",            (0.0, 0.80)),
           ("early",    "trough grad (uV/ms)",  (0.0, 900.0)),
           ("late",     "peak grad (uV/ms)",    (-150.0, 250.0))]


def classify_sample(s, animal_id):
    """Per-spike E/I label ('putative E'/'putative I'/'neither') for a _smoothed_cg
    result, applying the pipeline's box + QC keep logic (per-mouse boxes)."""
    box_e, box_i = dl._mouse_ei_boxes(animal_id)
    lab = dl._classify_ei_boxes(dl._spike_features(s["smoothed"]), box_e, box_i)  # 0=E,1=I,-1
    qc = dl._waveform_qc(dl._baseline_subtract(s["filt"].astype(np.float32)),
                         s["smoothed"][:, :20], s["nbr_dist"])
    out = np.full(len(lab), "neither", dtype=object)
    out[(lab == 0) & qc] = "putative E"
    out[(lab == 1) & qc] = "putative I"
    return out


def _feat_color(v, box_e, box_i, key):
    """red if value is in the E-box range for `key`, blue if in the I-box range,
    grey otherwise (or NaN)."""
    if np.isnan(v):
        return "0.5"
    lo_e, hi_e = box_e[key]
    lo_i, hi_i = box_i[key]
    if lo_e <= v <= hi_e:
        return "r"
    if lo_i <= v <= hi_i:
        return "b"
    return "0.5"


def _feature_legend(ax, feats, ei, box_e, box_i):
    """Legend of the 4 classification features for spike `ei`, each entry coloured
    by which box (E=red / I=blue / neither=grey) its value falls in."""
    rows = [("dur", "duration", "{:.2f} ms"),
            ("FW3M", "fw3m", "{:.2f} ms"),
            ("trough grad", "early", "{:.0f} uV/ms"),
            ("peak grad", "late", "{:.0f} uV/ms")]
    labels, colors = [], []
    for name, key, fmt in rows:
        v = float(feats[key][ei])
        labels.append(f"{name}: {fmt.format(v)}")
        colors.append(_feat_color(v, box_e, box_i, key))
    handles = [Line2D([], [], color=c, marker="s", ls="", ms=5) for c in colors]
    ax.legend(handles, labels, labelcolor=colors, fontsize=6.5, loc="lower right",
              handlelength=0, handletextpad=0.3, borderpad=0.3, framealpha=0.85)


def _plot_waveform(ax, w_uv):
    """One smoothed spike, x = ms from trough.  Shows up to PRE_MS of pre-trough
    baseline, but never past the start of the snippet (so no blank padding is
    drawn — the cached wide window is only ~WIDE_PRE/FS ms deep before the trough)."""
    trough = int(w_uv.argmin())
    t = (np.arange(len(w_uv)) - trough) / FS
    ax.plot(t, w_uv, "k", lw=1)
    ax.axvline(0, color="0.7", lw=.6)
    ax.set_xlim(max(-PRE_MS, float(t.min())), float(t.max()))
    ax.set(xlabel="ms from trough", ylabel="amplitude (uV)")


def plot_metric_distributions(name, base, min_cluster_group=2, sample_size=8000, seed=0):
    """One figure per mouse: histograms of the four classification metrics
    (duration, FW3M, trough grad, peak grad) over a sample of this mouse's spikes,
    with the E box range shaded red and the I box range shaded blue for each — so
    you can see the distributions against the cuts (e.g. where E candidates pile up
    just outside the E box)."""
    animal_id = Path(base).name
    s = figures._smoothed_cg(base, cg="0", min_cluster_group=min_cluster_group,
                             sample_size=sample_size, seed=seed)
    feats = dl._spike_features(s["smoothed"])
    box_e, box_i = dl._mouse_ei_boxes(animal_id)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, xlabel, xr) in zip(axes.ravel(), METRICS):
        v = feats[key]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=np.linspace(*xr, 80), color="0.6", edgecolor="none")
        for box, c in [(box_e, "r"), (box_i, "b")]:      # shade E and I ranges
            lo, hi = box[key]
            lo = xr[0] if lo == -np.inf else lo
            hi = xr[1] if hi == np.inf else hi
            ax.axvspan(lo, hi, color=c, alpha=0.15)
        ax.set(xlabel=xlabel, ylabel="count", xlim=xr)
    fig.suptitle(f"{name}: classification metric distributions (E box = red, I box = blue)",
                 fontsize=12)
    fig.tight_layout()
    out = f"results/metric_distributions_mcg_{min_cluster_group}_{name}.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


def plot_spikes_by_class(name, base, min_cluster_group=2    , counts=None, ncol=4, seed=1, sample_size=3000):
    """Grid of example spikes grouped by class (to see *why* E-like spikes are or
    aren't kept): `counts[class]` random members of each class, each with its
    smoothed waveform and colour-coded feature legend.  Extra neither spikes help
    survey the E candidates being rejected.  `counts` defaults to 4 E / 4 I / 8
    neither, laid out at `ncol` columns."""
    counts = counts or {"putative E": 4, "putative I": 4, "neither": 8}
    animal_id = Path(base).name
    s = figures._smoothed_cg(base, cg="0", min_cluster_group=min_cluster_group,
                             sample_size=sample_size, seed=seed)
    labels = classify_sample(s, animal_id)
    dw_uv = _uv(dl._dominant_trace(s["smoothed"]))   # smoothed waveform = what's classified
    feats = dl._spike_features(s["smoothed"])        # per-spike classification features
    box_e, box_i = dl._mouse_ei_boxes(animal_id)

    rng = np.random.default_rng(seed)
    rows_per = {cls: -(-c // ncol) for cls, c in counts.items()}   # ceil(c/ncol)
    total_rows = sum(rows_per.values())
    fig, axes = plt.subplots(total_rows, ncol, figsize=(3.4 * ncol, 3 * total_rows),
                             squeeze=False)
    r0 = 0
    for cls, count in counts.items():
        idx = np.where(labels == cls)[0]
        if len(idx) > count:
            idx = rng.choice(idx, count, replace=False)
        for j in range(rows_per[cls] * ncol):
            ax = axes[r0 + j // ncol][j % ncol]
            if j < len(idx):
                ei = idx[j]
                _plot_waveform(ax, dw_uv[ei])
                _feature_legend(ax, feats, ei, box_e, box_i)
                ax.set_title(cls, color=LABEL_COLOR[cls], fontsize=10)
            else:
                ax.set_visible(False)   # unfilled cell (class short, or last row)
        r0 += rows_per[cls]
    fig.suptitle(f"{name}: example spikes by class (per-mouse box + QC)", fontsize=12)
    fig.tight_layout()
    out = f"results/random_spikes_mcg_{min_cluster_group}_{name}.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    for name, base in MICE.items():
        plot_metric_distributions(name, base)
        plot_spikes_by_class(name, base)
