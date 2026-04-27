"""Backend i18n helpers.

Localized strings live in ``server/locales/<lang>.json`` and are served to
the frontend via ``GET /api/i18n/{lang}``. Both files must stay in sync —
every key present in one must be present in the other (enforced by
test_i18n.py).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import settings

LOCALES_DIR = Path(__file__).parent.parent / "locales"


@lru_cache(maxsize=8)
def load_locale(language: str) -> dict[str, str]:
    """Return the localized strings for ``language``.

    Falls back to the configured default language if the requested file
    doesn't exist. Cached so repeated calls don't re-read the disk.
    """
    candidate = LOCALES_DIR / f"{language}.json"
    if not candidate.exists():
        candidate = LOCALES_DIR / f"{settings.default_language}.json"
    if not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8"))


def supported_languages() -> list[str]:
    return list(settings.supported_languages)
