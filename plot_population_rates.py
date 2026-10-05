"""
plot_population_rates.py — eyeball E/I population rates for one mouse to spot
anomalous folds / sub-conditions worth dropping.

Reads `results/trial_counts_b<BIN_SAMPLES>_<animal_id>_s<session>.npz` (single-trial
counts; layout in DATA.md) and renders one figure per experiment type. Each figure
is a grid of subplots, one per stimulus sub-condition; within a subplot E (wide)
and I (narrow) rates are overlaid, and — crucially for the QC purpose — every CV
fold is drawn as its own thin line with the all-trials mean bold on top. A fold
that wanders off the others is the thing we're hunting for, so we show the folds
rather than hiding them inside the mean.

The files hold raw counts, so the reductions are chosen here (constants below):
rebin to PLOT_BIN_MS, optional Hamming smoothing, baseline normalisation
(baseline ~= 1.0), and a seeded split into N_FOLDS folds (the defaults reproduce
the legacy files' folds).  y=1 marks baseline and t=0 marks first-pulse onset.

Usage:
    python plot_population_rates.py                       # default mouse
    python plot_population_rates.py M150609_ICTP2         # another mouse
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import trial_analysis as ta

BIN_SAMPLES = 30      # which saved file to read (30 = 1 ms)
PLOT_BIN_MS = 10      # rebin to this before averaging; must be a multiple of the file's bin
HAMMING_MS = 0.0      # 40.0 for the paper's smoothing (smears onset earlier)
N_FOLDS = 3
FOLD_SEED = 0

# counts axis-1 index -> (label, colour). 0 = E (wide), 1 = I (narrow); see DATA.md.
_POPS = [("E", "tab:blue"), ("I", "tab:red")]

_TYPES = ["single_E", "single_I", "paired_EE", "paired_II", "paired_EI", "paired_IE"]


def _cond_label(d, k, n):
    d1, d2 = d["cond_dur1_ms"][k], d["cond_dur2_ms"][k]
    dur = f"{d1:g}" if np.isnan(d2) or d1 == d2 else f"{d1:g}/{d2:g}"
    ipi = d["cond_ipi_ms"][k]
    stim = "single" if ipi == 0 else f"intT {ipi}ms"
    screen = "" if d["cond_contrast"][k] == 0 else f", screen c{d['cond_contrast'][k]} Tp{d['cond_tp_ms'][k]:g}"
    return f"{stim}, dur {dur}ms{screen}  (n={n})"


def _stim_spans(d, k):
    """(start_s, end_s) grey strips for each laser pulse of condition k.

    Pulse 1 onset at t=0; pulse 2 (paired only) at onset_ipi = intT + dur1, since
    Protocol intT is end-of-pulse-1 to start-of-pulse-2 (README).
    """
    spans = [(0.0, d["cond_dur1_ms"][k] / 1000.0)]
    if d["cond_ipi_ms"][k] > 0:
        t2 = d["cond_onset_ipi_ms"][k] / 1000.0
        spans.append((t2, t2 + d["cond_dur2_ms"][k] / 1000.0))
    return spans


def _plot_type(d, counts, t, bin_width_s, fold, type_name, animal_id):
    conds = np.flatnonzero(d["cond_exp_type"] == type_name)
    n_cond = conds.size

    ncols = min(4, n_cond)
    nrows = int(np.ceil(n_cond / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True, squeeze=False
    )
    axes = axes.ravel()

    for ax, k in zip(axes, conds):
        in_cond = d["cond_idx"] == k
        for pop, (plabel, colour) in enumerate(_POPS):
            for f in range(N_FOLDS):
                sel = in_cond & (fold == f)
                if sel.any():
                    p = ta.psth(counts[sel], t, bin_width_s, hamming_ms=HAMMING_MS)
                    ax.plot(t, p[pop], color=colour, lw=0.6, alpha=0.35)
            p = ta.psth(counts[in_cond], t, bin_width_s, hamming_ms=HAMMING_MS)
            ax.plot(t, p[pop], color=colour, lw=1.8, label=plabel)
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        for s0, s1 in _stim_spans(d, k):
            ax.axvspan(s0, s1, color="0.55", alpha=0.7, lw=0)
        ax.set_title(_cond_label(d, k, int(in_cond.sum())), fontsize=8)
        # Match the paper's peristimulus window (Fig. panel A): -0.1 to 0.4 s.
        ax.set_xlim(-0.1, 0.4)
    for ax in axes[n_cond:]:
        ax.axis("off")

    axes[0].legend(fontsize=8, loc="upper right")
    smooth = f", Hamming {HAMMING_MS:g} ms" if HAMMING_MS > 0 else ""
    fig.suptitle(f"{animal_id} — {type_name}  (thin = folds, bold = all trials; "
                 f"{PLOT_BIN_MS} ms bins{smooth})")
    fig.supxlabel("time from first pulse (s)")
    fig.supylabel("baseline-normalised rate")
    fig.tight_layout()
    # One dir per mouse; filename keeps the full animal_id so the two M150609
    # protocols (ICTP1/ICTP2) stay distinct within the same folder.
    out_dir = f"results/population_rates_plots/{animal_id.split('_')[0]}_TRIALS"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/pop_rates_{animal_id}_{type_name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    animal_id = sys.argv[1] if len(sys.argv) > 1 else "M150605_ICTP1"
    d = ta.load(animal_id, bin_samples=BIN_SAMPLES)
    factor = round(PLOT_BIN_MS / 1000 / ta.bin_s(d))
    counts, t = ta.rebin(d["counts"], d["time_axis"], factor)
    bin_width_s = factor * ta.bin_s(d)
    fold = ta.fold_labels(d["cond_idx"], N_FOLDS, FOLD_SEED)
    for type_name in _TYPES:
        if (d["cond_exp_type"] == type_name).any():
            print("wrote", _plot_type(d, counts, t, bin_width_s, fold, type_name, animal_id))


if __name__ == "__main__":
    main()
