"""Backend i18n helpers for Steno."""

import json
from pathlib import Path

from steno.config import Config

_cache: dict[str, dict] = {}


def load_locale(lang: str) -> dict:
    """Load and return the JSON for the given language code.

    Falls back to 'en' if the requested language is not found.
    """
    if lang in _cache:
        return _cache[lang]

    locales_dir = Config.locales_path()
    locale_file = locales_dir / f"{lang}.json"

    if not locale_file.exists():
        lang = Config.DEFAULT_LANGUAGE
        locale_file = locales_dir / f"{lang}.json"

    with open(locale_file, encoding="utf-8") as f:
        data = json.load(f)

    _cache[lang] = data
    return data


def get_supported_languages() -> list[dict]:
    """Return list of supported languages with code and display name."""
    language_names = {
        "en": "English",
        "es": "Español",
    }
    return [
        {"code": lang, "name": language_names.get(lang, lang)}
        for lang in Config.SUPPORTED_LANGUAGES
    ]
