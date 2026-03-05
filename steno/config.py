"""Global configuration for Steno."""

from pathlib import Path


class Config:
    MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
    SAMPLE_RATE = 16000  # Hz required by Whisper
    CHUNK_DURATION = 5  # seconds per audio chunk
    OVERLAP_DURATION = 0.5  # overlap between chunks to avoid cutting words
    SESSIONS_DIR = "sessions"
    STATIC_DIR = "static"
    LOCALES_DIR = "locales"
    HOST = "127.0.0.1"
    PORT = 8080
    DEFAULT_LANGUAGE = "en"
    SUPPORTED_LANGUAGES = ["en", "es"]
    SILENCE_THRESHOLD = 0.01  # RMS below this = silence, skip transcription

    @classmethod
    def project_root(cls) -> Path:
        """Return the project root directory."""
        return Path(__file__).parent.parent

    @classmethod
    def sessions_path(cls) -> Path:
        return cls.project_root() / cls.SESSIONS_DIR

    @classmethod
    def static_path(cls) -> Path:
        return cls.project_root() / cls.STATIC_DIR

    @classmethod
    def locales_path(cls) -> Path:
        return cls.project_root() / cls.LOCALES_DIR
