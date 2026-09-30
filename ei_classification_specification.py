"""
ei_classification_specification.py — per-mouse E/I two-box classification spec.

The E-vs-I waveform cut is tuned per recording (Lin & Harris 2020, S1.6 picks the
duration / FW3M / gradient boundaries per session), so a single hard-coded box does
not fit every animal.  This module holds the *full* four-metric box for each mouse —
both the wide/excitatory (E) box and the narrow/inhibitory (I) box — so the loader
applies the right cut per mouse instead of overriding one metric on a shared default.

Each box is a dict keyed by the four classification features, each an inclusive
`(lo, hi)` range in the feature's units.  Use `-inf` / `+inf` for an open side (the
gradient cuts are naturally one-sided).

    duration : ms      (trough -> following peak)
    fw3m     : ms       (full width at 2/3 trough depth)
    early    : uV/ms    (gradient 0.07 ms after trough)
    late     : uV/ms    (gradient 0.50 ms after trough)

FW3M values below are the hand-picked ones carried over from the previous per-mouse
tuning.  The other three metrics are PLACEHOLDERS (shape only) pending Dabin's
per-mouse values — see the `# TODO` markers.
"""

from __future__ import annotations

INF = float("inf")

# Per-mouse full spec: filesystem mouse prefix -> {"E": box, "I": box}.
# Each box maps the four feature keys to inclusive (lo, hi) ranges.
#
# fw3m: real, hand-picked per mouse.
# duration / early / late: PLACEHOLDER — fill in the real per-mouse cuts.  # TODO(dabin)
_MOUSE_SPEC: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "M150605_ICTP1": {
        "E": dict(duration=(0.5, 0.8),
                  fw3m=(0.15, 0.30), # no clear bimodality 
                  early=(-INF, 300.0),
                  late=(0.0, INF)),
        "I": dict(duration=(0.22, 0.42),
                  fw3m=(0.08, 0.18), # no clear bimodality 
                  early=(300.0, INF),
                  late=(-INF, 0.0)),
    },
    "M150609_ICTP1": {
        "E": dict(duration=(0.5, 0.8),  
                  fw3m=(0.28, 0.45),
                  early=(-INF, 400.0),
                  late=(0.0, INF)), # no clear bimodality
        "I": dict(duration=(0.15, 0.35),
                  fw3m=(0.08, 0.2),
                  early=(400.0, INF),
                  late=(-INF, 0.0)), # no clear bimodality
    },
    "M150609_ICTP2": {
        "E": dict(duration=(0.5, 0.8),
                  fw3m=(0.2, 0.3), # intentionally overlapping
                  early=(-INF, 400.0), # intentionally overlapping
                  late=(0.0, INF)), # no clear bimodality
        "I": dict(duration=(0.18, 0.35),
                  fw3m=(0.08, 0.25),
                  early=(200.0, INF), # intentionally overlapping
                  late=(-INF, 0.0)), # no clear bimodality
    },
    "M151020_ICTP1": {
        "E": dict(duration=(0.5, 0.80), 
                  fw3m=(0.15, 0.30),
                  early=(-INF, 400.0), # no clear bimodality
                  late=(0.0, INF)),
        "I": dict(duration=(0.18, 0.4),
                  fw3m=(0.08, 0.18),
                  early=(350.0, INF), # no clear bimodality
                  late=(-INF, 0.0)),
    },
    # "M150909_ICTP3": {
    #     "E": dict(duration=(0.5, 0.80), 
    #               fw3m=(0.15, 0.30),
    #               early=(-INF, 400.0), # no clear bimodality
    #               late=(0.0, INF)),
    #     "I": dict(duration=(0.18, 0.4),
    #               fw3m=(0.08, 0.18),
    #               early=(350.0, INF), # no clear bimodality
    #               late=(-INF, 0.0)),
    # },
    "M150823_ICTP2": {
        "E": dict(duration=(0.5, 0.80), 
                  fw3m=(0.15, 0.30),
                  early=(-INF, 400.0), # no clear bimodality
                  late=(0.0, INF)),
        "I": dict(duration=(0.18, 0.4),
                  fw3m=(0.08, 0.18),
                  early=(350.0, INF), # no clear bimodality
                  late=(-INF, 0.0)),
    },

}


def mouse_ei_boxes(mouse_id: str) -> tuple[dict, dict]:
    """
    (box_e, box_i) full four-metric E/I boxes for an animal.

    Accepts a filesystem animal_id ("M150605_ICTP1") or a bare mouse ("M150605").
    Returns fresh dict copies so callers can mutate them safely.  Raises KeyError for
    a mouse without a hand-picked spec — callers decide whether to fall back to a
    default (see `data_loader._mouse_ei_boxes`).
    """
    key = mouse_id if mouse_id in _MOUSE_SPEC else mouse_id.split("_")[0]
    try:
        spec = _MOUSE_SPEC[key]
    except KeyError:
        known = ", ".join(sorted(_MOUSE_SPEC))
        raise KeyError(
            f"No E/I classification spec for mouse {mouse_id!r}. Known: {known}"
        ) from None
    return dict(spec["E"]), dict(spec["I"])
