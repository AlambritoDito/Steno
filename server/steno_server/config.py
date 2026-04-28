"""Steno Server runtime configuration.

All settings are loaded from environment variables (via pydantic-settings)
with safe defaults that work for local development. Production deploys
override the load-bearing ones (HF_TOKEN, AUTH_PASSWORD, HOST, PORT)
through the LaunchAgent .plist or shell exports.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server-wide configuration. Reads STENO_SERVER_<NAME> env vars."""

    model_config = SettingsConfigDict(
        env_prefix="STENO_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Whisper model & language ------------------------------------------------

    model_name: str = "mlx-community/whisper-large-v3-turbo"
    default_language: str = "es"
    supported_languages: list[str] = ["es", "en"]

    # -- Pipeline / VAD ----------------------------------------------------------

    enable_vad: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 400

    # -- Anti-hallucination Whisper params --------------------------------------

    temperature: float = 0.0
    condition_on_previous_text: bool = False
    no_repeat_ngram_size: int = 5  # not exposed by mlx-whisper; tracked via delooping postprocess
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    hallucination_silence_threshold: float = 2.0  # seconds

    # -- Phase 2 toggles ---------------------------------------------------------

    enable_denoise: bool = True
    enable_diarization: bool = True
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    # -- Storage -----------------------------------------------------------------

    storage_dir: Path = Path("/tmp/steno-server")
    job_retention_hours: int = 24
    max_upload_size_mb: int = 500

    # -- Server ------------------------------------------------------------------

    host: str = "0.0.0.0"  # noqa: S104 — intentional, needs to bind LAN + Tailscale
    port: int = 8090
    workers: int = 1  # mlx-whisper is not thread-safe; keep single-worker

    # -- Auth --------------------------------------------------------------------

    auth_password: str | None = Field(default=None, validation_alias="AUTH_PASSWORD")
    session_cookie_name: str = "steno_session"
    session_cookie_secure: bool = True  # production default; flipped off in tests
    session_duration_hours: int = 12

    # -- Logging -----------------------------------------------------------------

    log_level: str = "INFO"
    log_dir: Path = Path.home() / "Library" / "Logs" / "steno-server"
    log_retention_days: int = 30


settings = Settings()
