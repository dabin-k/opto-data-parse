"""
fw3m.py — per-mouse FW3M (full-width at one-third max) E/I classification ranges.

The FW3M boundaries that separate wide (E) from narrow (I) spikes are tuned per
recording (Lin & Harris 2020, S1.6 tunes the duration/FW3M cut per session), so a
single hard-coded range does not fit every animal.  This module holds the
hand-picked ranges and resolves them from an animal ID, so the loader applies the
right cut per mouse instead of the M150605 default everywhere.

Ranges are (min_ms, max_ms).  A mouse maps to (wide_E_range, narrow_I_range).
"""

from __future__ import annotations

# Candidate FW3M ranges (ms), named so the per-mouse picks below read clearly.
_WIDE = {
    "18_30": (0.18, 0.30),
    "18_33": (0.18, 0.33),
    "15_30": (0.15, 0.30),
}
_NARROW = {
    "08_18": (0.08, 0.18),
    "07_20": (0.07, 0.20),
    "10_18": (0.10, 0.18),
    "10_22": (0.10, 0.22),
}

# Filesystem mouse prefix -> (wide/E FW3M range, narrow/I FW3M range).
_MOUSE_FW3M: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "M150605": (_WIDE["18_30"], _NARROW["07_20"]),
    "M150609": (_WIDE["15_30"], _NARROW["07_20"]),  # wide fit poor here; widened low end
}


def mouse_fw3m_ranges(
    mouse_id: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    ((wide_E_lo, wide_E_hi), (narrow_I_lo, narrow_I_hi)) in ms for an animal.

    Accepts a filesystem animal_id ("M150605_ICTP1") or a bare mouse ("M150605").
    Raises KeyError for a mouse without hand-picked ranges — callers decide whether
    to fall back to a default (see `data_loader._mouse_ei_boxes`).
    """
    key = mouse_id if mouse_id in _MOUSE_FW3M else mouse_id.split("_")[0]
    try:
        return _MOUSE_FW3M[key]
    except KeyError:
        known = ", ".join(sorted(_MOUSE_FW3M))
        raise KeyError(
            f"No hand-picked FW3M ranges for mouse {mouse_id!r}. Known: {known}"
        ) from None
