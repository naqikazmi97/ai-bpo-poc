"""
providers.py — Supported electric utility providers.

This is the single place to maintain the allowed-provider list. It is NOT in
the LLM system prompt: llm.py exposes a `check_electric_provider` tool that the
model calls during STEP 6, and that tool uses match_provider() below.

To add/remove a provider, edit ELECTRIC_PROVIDERS (and _ALIASES for common
spoken/ASR shorthands). No prompt or pipeline changes needed.

Company list sourced from US_Electric_Companies_Expanded.xlsx.
"""
from difflib import get_close_matches
import re

# Canonical, display-ready names. Edit this list to maintain coverage.
ELECTRIC_PROVIDERS = [
    "NextEra Energy",
    "Florida Power & Light",
    "Duke Energy",
    "Southern Company",
    "Georgia Power",
    "Alabama Power",
    "Mississippi Power",
    "Dominion Energy",
    "American Electric Power (AEP)",
    "Appalachian Power",
    "AEP Ohio",
    "Public Service Company of Oklahoma",
    "Exelon",
    "ComEd",
    "PECO",
    "BGE",
    "Pepco",
    "Xcel Energy",
    "Entergy",
    "Edison International",
    "Southern California Edison",
    "PG&E Corporation",
    "Pacific Gas & Electric",
    "PPL Corporation",
    "FirstEnergy",
    "Ohio Edison",
    "Toledo Edison",
    "PSEG",
    "CenterPoint Energy",
    "Evergy",
    "AES Corporation",
    "AES Indiana",
    "Alliant Energy",
    "Avangrid",
    "Central Maine Power",
    "New York State Electric & Gas",
    "Rochester Gas & Electric",
    "OGE Energy",
    "Portland General Electric",
    "Hawaiian Electric",
    "IDACORP",
    "Idaho Power",
    "Black Hills Energy",
    "NorthWestern Energy",
    "PNM Resources",
    "Public Service Company of New Mexico",
    "Ameren",
    "Eversource Energy",
    "Consolidated Edison (Con Edison)",
    "National Grid USA",
    "Sempra",
    "San Diego Gas & Electric",
    "WEC Energy Group",
    "We Energies",
    "DTE Energy",
    "Consumers Energy",
    "Avista Utilities",
    "Duquesne Light",
    "Cleco",
]

# Common spoken / ASR shorthands -> canonical name.
# Keys must be in normalized form (see _normalize); values must match a name
# in ELECTRIC_PROVIDERS exactly. "pge" is intentionally omitted because it is
# ambiguous between Pacific Gas & Electric (CA) and Portland General Electric (OR).
_ALIASES = {
    "pg and e": "Pacific Gas & Electric",
    "pg e": "Pacific Gas & Electric",
    "sce": "Southern California Edison",
    "sdge": "San Diego Gas & Electric",
    "sdg and e": "San Diego Gas & Electric",
    "fpl": "Florida Power & Light",
    "con ed": "Consolidated Edison (Con Edison)",
    "coned": "Consolidated Edison (Con Edison)",
    "con edison": "Consolidated Edison (Con Edison)",
    "aep": "American Electric Power (AEP)",
    "nyseg": "New York State Electric & Gas",
    "rge": "Rochester Gas & Electric",
    "cmp": "Central Maine Power",
    "dte": "DTE Energy",
    "oge": "OGE Energy",
    "pnm": "PNM Resources",
}


def _normalize(s: str) -> str:
    """Lowercase, expand &, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NORM_TO_CANONICAL = {_normalize(p): p for p in ELECTRIC_PROVIDERS}


def match_provider(spoken: str):
    """
    Match a customer-spoken provider name against the supported list.

    Returns the canonical provider name (str) if supported, else None.
    Designed to tolerate ASR noise: handles aliases, normalized exact
    matches, the list entry appearing inside the spoken phrase, and a
    fuzzy fallback for close misspellings.
    """
    if not spoken or not spoken.strip():
        return None

    norm = _normalize(spoken)

    # 1. Known shorthand / alias.
    if norm in _ALIASES:
        return _ALIASES[norm]

    # 2. Exact normalized match.
    if norm in _NORM_TO_CANONICAL:
        return _NORM_TO_CANONICAL[norm]

    # 3. A supported provider's name appears inside what they said
    #    (e.g. "i'm with duke energy right now" -> "Duke Energy").
    #    Prefer the longest match so "southern california edison" wins over
    #    a shorter incidental substring.
    candidates = [(n, c) for n, c in _NORM_TO_CANONICAL.items() if n in norm]
    if candidates:
        return max(candidates, key=lambda nc: len(nc[0]))[1]

    # 4. Fuzzy fallback for ASR errors / slight misspellings.
    close = get_close_matches(norm, list(_NORM_TO_CANONICAL.keys()), n=1, cutoff=0.82)
    if close:
        return _NORM_TO_CANONICAL[close[0]]

    return None