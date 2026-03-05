"""Global configuration for Steno."""

import json
import shutil
from pathlib import Path


# Available MLX-Whisper models sorted by size/quality
WHISPER_MODELS = {
    "tiny": {
        "repo": "mlx-community/whisper-tiny",
        "size_mb": 75,
        "min_ram_gb": 4,
        "quality": "Basic",
        "speed": "Fastest",
    },
    "base": {
        "repo": "mlx-community/whisper-base",
        "size_mb": 145,
        "min_ram_gb": 4,
        "quality": "Fair",
        "speed": "Very fast",
    },
    "small": {
        "repo": "mlx-community/whisper-small",
        "size_mb": 490,
        "min_ram_gb": 8,
        "quality": "Good",
        "speed": "Fast",
    },
    "large-v3-turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "size_mb": 1600,
        "min_ram_gb": 16,
        "quality": "Excellent",
        "speed": "Moderate",
    },
    "large-v3": {
        "repo": "mlx-community/whisper-large-v3",
        "size_mb": 3100,
        "min_ram_gb": 32,
        "quality": "Best",
        "speed": "Slow",
    },
}


def _detect_hardware() -> dict:
    """Detect Apple Silicon chip and RAM."""
    import platform
    import subprocess

    ram_bytes = 0
    chip = "Unknown"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        ram_bytes = int(result.stdout.strip())
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        chip = result.stdout.strip()
    except Exception:
        chip = platform.processor() or "Unknown"

    ram_gb = ram_bytes / (1024 ** 3) if ram_bytes else 8  # default 8 GB
    return {"chip": chip, "ram_gb": round(ram_gb, 1)}


def recommend_model(ram_gb: float) -> str:
    """Return the best model key for the given RAM."""
    if ram_gb >= 32:
        return "large-v3"
    elif ram_gb >= 16:
        return "large-v3-turbo"
    elif ram_gb >= 8:
        return "small"
    else:
        return "base"


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}


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
    SETTINGS_FILE = ".steno_settings.json"

    @classmethod
    def is_frozen(cls) -> bool:
        """Return True when running as a PyInstaller bundle."""
        import sys
        return getattr(sys, "frozen", False)

    @classmethod
    def project_root(cls) -> Path:
        """Return the project root (or PyInstaller bundle root)."""
        if cls.is_frozen():
            import sys
            return Path(sys._MEIPASS)
        return Path(__file__).parent.parent

    @classmethod
    def data_dir(cls) -> Path:
        """Writable directory for sessions, settings, and user data.

        When running as a packaged app, this points to
        ``~/Documents/Steno/`` so data persists between launches.
        In development, it falls back to the project root.
        """
        if cls.is_frozen():
            data = Path.home() / "Documents" / "Steno"
            data.mkdir(parents=True, exist_ok=True)
            return data
        return cls.project_root()

    @classmethod
    def sessions_path(cls) -> Path:
        path = cls.data_dir() / cls.SESSIONS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def static_path(cls) -> Path:
        return cls.project_root() / cls.STATIC_DIR

    @classmethod
    def locales_path(cls) -> Path:
        return cls.project_root() / cls.LOCALES_DIR

    @classmethod
    def settings_path(cls) -> Path:
        return cls.data_dir() / cls.SETTINGS_FILE

    @classmethod
    def load_settings(cls) -> dict:
        """Load saved settings from disk."""
        path = cls.settings_path()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    @classmethod
    def save_settings(cls, settings: dict) -> None:
        """Save settings to disk."""
        path = cls.settings_path()
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    @staticmethod
    def model_cache_path(repo: str) -> Path | None:
        """Return the HuggingFace cache directory for a model repo, or None."""
        # HF stores models in ~/.cache/huggingface/hub/models--{org}--{name}
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        folder_name = "models--" + repo.replace("/", "--")
        path = cache_dir / folder_name
        return path if path.exists() else None

    @staticmethod
    def delete_model_cache(repo: str) -> bool:
        """Delete a model's HuggingFace cache. Returns True if deleted."""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        folder_name = "models--" + repo.replace("/", "--")
        path = cache_dir / folder_name
        if path.exists():
            shutil.rmtree(path)
            return True
        return False

    @staticmethod
    def model_cache_size_mb(repo: str) -> float:
        """Return the disk size of a model cache in MB."""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        folder_name = "models--" + repo.replace("/", "--")
        path = cache_dir / folder_name
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 1)
