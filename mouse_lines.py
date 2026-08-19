"""
mouse_lines.py — mouse ID → mouse line → opsin stimulation wavelengths.

Encodes Tables S1 and S2 of Lin, Okun, Carandini & Harris (2020) so the
wavelength that drives each population can be resolved *per animal* instead of
assuming "blue = excitatory".  Which wavelength targets E vs I is a property of
the mouse line (which opsin sits in which cell type), not a fixed constant — see
CLAUDE.md.  Kept separate from data_loader.py while the loader is still
single-session; this is the seed of the mouse/experiment configurability aim.

Two public functions
---------------------
mouse_line(mouse_id) -> str
    Map an animal ID to its mouse line (a key of _LINE_WAVELENGTHS).

line_wavelengths(line) -> (e_nm, i_nm)
    Map a mouse line to (excitatory-targeting nm, inhibitory-targeting nm).
    Either entry is None if that population carries no opsin in that line.
"""

from __future__ import annotations

# --- Table S1: opsin configuration per mouse line --------------------------
# (excitatory-targeting wavelength [nm], inhibitory-targeting wavelength [nm]).
# None => that population has no opsin and cannot be optogenetically stimulated.
_LINE_WAVELENGTHS: dict[str, tuple[int | None, int | None]] = {
    "C57BL/6J":         (None, None),   # wild type, no opsin
    "Thy18":            (445, None),    # ChR2 in pyramidal (E) only
    "PVcre;Ai32":       (None, 445),    # ChR2 in PV (I) only — 445 nm drives I here
    "PVcre;Thy18+C1V1": (445, 561),     # ChR2 in E @445 nm, C1V1 in PV (I) @561 nm
}

# --- Table S2: mouse ID → mouse line ---------------------------------------
# PVcre;Thy18 animals were injected with C1V1 (→ "PVcre;Thy18+C1V1"), giving the
# two-wavelength E/I paradigm.  The three footnote-a animals showed no virus
# expression, so functionally only their Thy18 ChR2 (E) is drivable → "Thy18".
_MOUSE_LINE: dict[str, str] = {
    # PVcre;Thy18 + C1V1 (445 nm → E, 561 nm → I)
    "M150605A": "PVcre;Thy18+C1V1",
    "M150909C": "PVcre;Thy18+C1V1",
    "M150823B": "PVcre;Thy18+C1V1",
    "M150303B": "PVcre;Thy18+C1V1",
    "M150609A": "PVcre;Thy18+C1V1",
    "M150609B": "PVcre;Thy18+C1V1",
    "M161122":  "PVcre;Thy18+C1V1",
    "M151020A": "PVcre;Thy18+C1V1",
    "M160817B": "PVcre;Thy18+C1V1",
    "M141020A": "PVcre;Thy18+C1V1",
    "M160817A": "PVcre;Thy18+C1V1",
    # PVcre;Thy18 genotype but no C1V1 expression (Table S2 footnote a) → E only
    "M150909A": "Thy18",
    "M150909B": "Thy18",
    "M150309A": "Thy18",
    # Thy18 (445 nm → E only)
    "M150224":  "Thy18",
    "M160802":  "Thy18",
    "M160908":  "Thy18",
    "M160804":  "Thy18",
    "M151120B": "Thy18",
    "M170308":  "Thy18",
    "M170522":  "Thy18",
    "M170602":  "Thy18",
    "M170614":  "Thy18",
    # PVcre;Ai32 (445 nm → I only)
    "M160906":  "PVcre;Ai32",
    "M160921":  "PVcre;Ai32",
    "M160226":  "PVcre;Ai32",
    "M151105":  "PVcre;Ai32",
    # C57BL/6J (no opsin; visual flash only)
    "M160210A": "C57BL/6J",
    "M160210B": "C57BL/6J",
    "M181130B": "C57BL/6J",
    "M181130C": "C57BL/6J",
    "M181204B": "C57BL/6J",
}

# Filesystem / internal IDs that name the same animal as a Table S2 entry.
# M150605_ICTP1 is the session dir for Table S2's M150605A (only M150605 in S2;
# ran the two-wavelength 2waves paired-pulse protocol, consistent with +C1V1).
_MOUSE_ID_ALIASES: dict[str, str] = {
    "M150605_ICTP1": "M150605A",
    "M150609_ICTP1": "M150609A",
    # Add more session-dir -> Table S2 ID as they are verified.  The ICTP1 -> "A"
    # correspondence is confirmed for these two; do not assume it for others.
}


def mouse_line(mouse_id: str) -> str:
    """
    Return the mouse line for an animal ID (a key of _LINE_WAVELENGTHS).

    Accepts either a Table S2 mouse ID (e.g. "M150605A") or a known filesystem
    alias (e.g. "M150605_ICTP1").  Raises KeyError for unknown IDs.
    """
    key = _MOUSE_ID_ALIASES.get(mouse_id, mouse_id)
    try:
        return _MOUSE_LINE[key]
    except KeyError:
        known = ", ".join(sorted(_MOUSE_LINE))
        raise KeyError(
            f"Unknown mouse_id {mouse_id!r}; not in Table S2. Known IDs: {known}"
        ) from None


def line_wavelengths(line: str) -> tuple[int | None, int | None]:
    """
    Return (excitatory-targeting nm, inhibitory-targeting nm) for a mouse line.

    None in a slot means that population carries no opsin in this line and cannot
    be optogenetically stimulated, e.g. line_wavelengths("PVcre;Ai32") == (None, 445).
    Raises KeyError for an unknown line.
    """
    try:
        return _LINE_WAVELENGTHS[line]
    except KeyError:
        known = ", ".join(_LINE_WAVELENGTHS)
        raise KeyError(
            f"Unknown mouse line {line!r}. Known lines: {known}"
        ) from None


def mouse_wavelengths(mouse_id: str) -> tuple[int | None, int | None]:
    """Convenience: (E-targeting nm, I-targeting nm) straight from a mouse ID."""
    return line_wavelengths(mouse_line(mouse_id))
