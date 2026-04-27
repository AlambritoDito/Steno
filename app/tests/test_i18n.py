"""Tests for steno.i18n module."""

import pytest
from httpx import ASGITransport, AsyncClient

from steno.i18n import load_locale, get_supported_languages, _cache
from steno.server import app


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear i18n cache between tests."""
    _cache.clear()
    yield
    _cache.clear()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def test_load_locale_english():
    """Loads en.json and returns a dict."""
    data = load_locale("en")
    assert isinstance(data, dict)
    assert "app_name" in data
    assert data["app_name"] == "Steno"


def test_load_locale_spanish():
    """Loads es.json and returns a dict."""
    data = load_locale("es")
    assert isinstance(data, dict)
    assert "app_name" in data
    assert data["tagline"] == "Transcripción en tiempo real para clases y juntas"


def test_load_locale_fallback():
    """Unknown language code falls back to English."""
    data = load_locale("xx")
    assert isinstance(data, dict)
    assert data["app_name"] == "Steno"
    # Should match English
    en = load_locale("en")
    assert data == en


def test_locales_have_same_keys():
    """en.json and es.json have identical key sets."""
    en = load_locale("en")
    es = load_locale("es")
    assert set(en.keys()) == set(es.keys())


def test_get_supported_languages():
    """Returns list with en and es."""
    langs = get_supported_languages()
    assert isinstance(langs, list)
    codes = [l["code"] for l in langs]
    assert "en" in codes
    assert "es" in codes


@pytest.mark.asyncio
async def test_api_i18n_english(client):
    """GET /api/i18n/en returns 200 with JSON."""
    resp = await client.get("/api/i18n/en")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] == "Steno"


@pytest.mark.asyncio
async def test_api_i18n_spanish(client):
    """GET /api/i18n/es returns 200 with JSON."""
    resp = await client.get("/api/i18n/es")
    assert resp.status_code == 200
    data = resp.json()
    assert "app_name" in data


@pytest.mark.asyncio
async def test_api_i18n_invalid_falls_back(client):
    """GET /api/i18n/xx returns English strings."""
    resp = await client.get("/api/i18n/xx")
    assert resp.status_code == 200
    data = resp.json()
    en = load_locale("en")
    assert data == en


@pytest.mark.asyncio
async def test_api_languages_list(client):
    """GET /api/languages returns list with code and name."""
    resp = await client.get("/api/languages")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    codes = [l["code"] for l in data]
    names = [l["name"] for l in data]
    assert "en" in codes
    assert "es" in codes
    assert "English" in names
    assert "Español" in names


# --- v0.2.0: new locale keys ---


def test_v020_locale_keys_present():
    """All v0.2.0 locale keys exist in English locale."""
    en = load_locale("en")
    v020_keys = [
        "portaudio_missing",
        "error_loading_models",
        "error_generic_retry",
        "update_available",
        "update_download",
        "update_dismiss",
        "debug_hint",
    ]
    for key in v020_keys:
        assert key in en, f"Missing locale key: {key}"
