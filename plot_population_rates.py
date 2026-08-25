"""
plot_population_rates.py — eyeball E/I population rates for one mouse to spot
anomalous folds / sub-conditions worth dropping.

Reads a `results/population_rates_<animal_id>_s<session>.npz` (structure in
DATA.md) and renders one figure per experiment type. Each figure is a grid of
subplots, one per stimulus sub-condition; within a subplot E (wide) and I
(narrow) rates are overlaid, and — crucially for the QC purpose — every CV fold
is drawn as its own thin line with the fold-weighted all-trials mean bold on
top. A fold that wanders off the others is the thing we're hunting for, so we
show the folds rather than hiding them inside the mean.

Rates are already Hamming-smoothed and baseline-normalised (baseline ~= 1.0),
so a horizontal line at y=1 marks baseline and t=0 marks first-pulse onset.

Usage:
    python plot_population_rates.py                       # default mouse
    python plot_population_rates.py M150609_ICTP2         # another mouse
    python plot_population_rates.py M151020_ICTP1 --full  # the FULL_ file
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# axis-2 index -> (label, colour). 0 = E (wide), 1 = I (narrow); see DATA.md.
_POPS = [("E", "tab:blue"), ("I", "tab:red")]

_TYPES = ["single_E", "single_I", "paired_EE", "paired_II", "paired_EI", "paired_IE"]


def _load(animal_id, session=1, full=False, plot_smooth=False):
    prefix = "FULL_" if full else ""
    return np.load(
        f"results/{'smoothed_' if plot_smooth else ''}{prefix}population_rates_{animal_id}_s{session}.npz",
        allow_pickle=True,
    )


def _fold_mean(resp_cond, n_trials_per_fold):
    """All-trials mean over folds, weighting each fold by its trial count.

    resp_cond: (n_folds, n_bins) for one population. Empty folds are NaN (see
    DATA.md); nan-weighting them out recovers the true all-trials mean.
    """
    w = np.asarray(n_trials_per_fold, float)
    return np.nansum(resp_cond * w[:, None], axis=0) / w.sum()


def _cond_label(c):
    dur = c["dur_ms"]
    dur = f"{dur[0]}/{dur[1]}" if isinstance(dur, list) else f"{dur:g}"
    stim = "single" if c["ipi_ms"] == 0 else f"ipi {c['ipi_ms']}ms"
    return f"{stim}, dur {dur}ms  (n={c['n_trials']})"


def _stim_spans(c):
    """(start_s, end_s) grey strips for each laser pulse in a condition.

    Pulse 1 onset at t=0; pulse 2 (paired only) onset at the IPI (onset-to-onset).
    `dur_ms` is a scalar, or [d1, d2] when the two pulses differ (DATA.md).
    """
    dur = c["dur_ms"]
    d1, d2 = (dur[0], dur[1]) if isinstance(dur, list) else (dur, dur)
    spans = [(0.0, d1 / 1000.0)]
    if c["ipi_ms"] > 0:
        t2 = c["ipi_ms"] / 1000.0
        spans.append((t2, t2 + d2 / 1000.0))
    return spans


def _plot_type(d, type_name, animal_id, plot_smooth):
    resp = d[f"{type_name}__responses"]            # (n_cond, n_folds, 2, n_bins)
    t = d[f"{type_name}__time_axis"]
    conds = json.loads(str(d[f"{type_name}__conditions"]))
    n_cond, n_folds = resp.shape[0], resp.shape[1]

    ncols = min(4, n_cond)
    nrows = int(np.ceil(n_cond / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True, squeeze=False
    )
    axes = axes.ravel()

    for ci, c in enumerate(conds):
        ax = axes[ci]
        for pop_ax, (plabel, colour) in enumerate(_POPS):
            folds = resp[ci, :, pop_ax, :]         # (n_folds, n_bins)
            for f in range(n_folds):
                ax.plot(t, folds[f], color=colour, lw=0.6, alpha=0.35)
            mean = _fold_mean(folds, c["n_trials_per_fold"])
            ax.plot(t, mean, color=colour, lw=1.8, label=plabel)
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        for s0, s1 in _stim_spans(c):
            ax.axvspan(s0, s1, color="0.55", alpha=0.7, lw=0)
        ax.set_title(_cond_label(c), fontsize=8)
        # Match the paper's peristimulus window (Fig. panel A): -0.1 to 0.4 s.
        ax.set_xlim(-0.1, 0.4)
    for ci in range(n_cond, len(axes)):
        axes[ci].axis("off")

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{animal_id} — {type_name}  (thin = folds, bold = weighted mean)")
    fig.supxlabel("time from first pulse (s)")
    fig.supylabel("baseline-normalised rate")
    fig.tight_layout()
    # Grouped one dir per mouse; filename keeps the full animal_id so the two
    # M150609 protocols (ICTP1/ICTP2) stay distinct within the same folder.
    if plot_smooth:
        out_dir = f"results/population_rates_plots/{animal_id.split('_')[0]}"
    else:
        out_dir = f"results/population_rates_plots/{animal_id.split('_')[0]}_UNSMOOTHED"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/pop_rates_{animal_id}_{type_name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv[1:]
    plot_smooth = "--plot_smooth" in sys.argv[1:]
    animal_id = args[0] if args else "M150605_ICTP1"

    d = _load(animal_id, full=full, plot_smooth=plot_smooth)
    for type_name in _TYPES:
        if f"{type_name}__responses" not in d:
            continue
        print("wrote", _plot_type(d, type_name, animal_id, plot_smooth))


if __name__ == "__main__":
    main()
