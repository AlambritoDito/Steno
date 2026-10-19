"""Platform configuration and durable local settings for Steno."""

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("steno.config")


def default_backend() -> str:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mlx"
    if platform.system() == "Windows" and platform.machine().lower() in {"amd64", "x86_64"}:
        return "faster-whisper"
    return "unsupported"


# Stable product keys; model formats are specific to the inference engine.
_MODEL_SPECS = {
    "tiny": (75, 4, "Basic", "Fastest"),
    "base": (145, 4, "Fair", "Very fast"),
    "small": (490, 8, "Good", "Fast"),
    "large-v3-turbo": (1600, 16, "Excellent", "Moderate"),
    "large-v3": (3100, 32, "Best", "Slow"),
}
WHISPER_MODELS = {}
for _key, (_size, _ram, _quality, _speed) in _MODEL_SPECS.items():
    _repos = {
        "mlx": f"mlx-community/whisper-{_key}",
        "faster-whisper": (
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
            if _key == "large-v3-turbo" else f"Systran/faster-whisper-{_key}"
        ),
    }
    WHISPER_MODELS[_key] = {
        "repo": _repos.get(default_backend(), _repos["faster-whisper"]),
        "repos": _repos, "size_mb": _size, "min_ram_gb": _ram,
        "quality": _quality, "speed": _speed,
    }


def resolve_model_repo(name: str, backend: str | None = None) -> str:
    """Translate saved product keys or another platform's repository name."""
    backend = backend or default_backend()
    for key, info in WHISPER_MODELS.items():
        if name == key or name in info["repos"].values():
            return info["repos"].get(backend, info["repo"])
    return name


def _detect_hardware() -> dict:
    """Report measured RAM; unknown hardware is never represented as guessed RAM."""
    chip = platform.processor() or platform.machine() or "Unknown"
    ram_gb = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except (ImportError, OSError, RuntimeError):
        logger.warning("Could not determine system memory")
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            chip = result.stdout.strip() or chip
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "chip": chip, "ram_gb": ram_gb, "platform": platform.system(),
        "architecture": platform.machine(), "backend": default_backend(),
    }


def recommend_model(ram_gb: float | None) -> str:
    if ram_gb is None:
        return "base"
    if ram_gb >= 32:
        return "large-v3"
    if ram_gb >= 16:
        return "large-v3-turbo"
    if ram_gb >= 8:
        return "small"
    return "base"


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}


class Config:
    MODEL_NAME = WHISPER_MODELS["large-v3-turbo"]["repo"]
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 5
    OVERLAP_DURATION = 0.5
    SESSIONS_DIR = "sessions"
    STATIC_DIR = "static"
    LOCALES_DIR = "locales"
    HOST = "127.0.0.1"
    PORT = 8080
    DEFAULT_LANGUAGE = "en"
    SUPPORTED_LANGUAGES = ["en", "es"]
    SILENCE_THRESHOLD = 0.01
    SETTINGS_FILE = ".steno_settings.json"

    @classmethod
    def is_frozen(cls) -> bool:
        import sys
        return getattr(sys, "frozen", False)

    @classmethod
    def project_root(cls) -> Path:
        if cls.is_frozen():
            import sys
            return Path(sys._MEIPASS)
        return Path(__file__).parent.parent

    @classmethod
    def data_dir(cls) -> Path:
        override = os.environ.get("STENO_DATA_DIR")
        if override:
            data = Path(override).expanduser()
        elif platform.system() == "Darwin":
            data = Path.home() / "Library" / "Application Support" / "Steno"
        elif platform.system() == "Windows":
            data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Steno"
        else:
            data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "Steno"
        data.mkdir(parents=True, exist_ok=True)
        return data

    @classmethod
    def legacy_data_dirs(cls) -> list[Path]:
        """Read-only migration sources; callers must preserve the originals."""
        paths = [Path.home() / "Documents" / "Steno"]
        if not cls.is_frozen():
            paths.append(cls.project_root())
        current = cls.data_dir().resolve()
        return [p for p in paths if p.resolve() != current]

    @classmethod
    def sessions_path(cls) -> Path:
        path = cls.data_dir() / cls.SESSIONS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def images_path(cls, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("Invalid session identifier")
        path = cls.sessions_path() / "images" / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def logs_path(cls) -> Path:
        path = Path(os.environ["STENO_LOG_DIR"]) if os.environ.get("STENO_LOG_DIR") else cls.data_dir() / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def models_dir(cls) -> Path:
        path = cls.data_dir() / "models"
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
        path = cls.settings_path()
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError("Settings must be an object")
            return settings
        except FileNotFoundError:
            return {}
        except (ValueError, UnicodeError):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            backup = path.with_name(f"{path.name}.corrupt-{stamp}")
            path.replace(backup)
            logger.warning("Invalid settings preserved at %s", backup)
            return {}

    @classmethod
    def save_settings(cls, settings: dict) -> None:
        if not isinstance(settings, dict):
            raise TypeError("Settings must be an object")
        path = cls.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(settings, indent=2, ensure_ascii=False, allow_nan=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _cache_folder(repo: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or ".." in repo.split("/"):
            raise ValueError("Invalid model repository")
        return "models--" + repo.replace("/", "--")

    @classmethod
    def _cache_roots(cls) -> list[Path]:
        roots = [cls.models_dir()]
        roots += [p / "models" for p in cls.legacy_data_dirs()]
        if os.environ.get("HF_HUB_CACHE"):
            roots.append(Path(os.environ["HF_HUB_CACHE"]))
        roots.append(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
        return list(dict.fromkeys(roots))

    @classmethod
    def model_cache_path(cls, repo: str) -> Path | None:
        folder = cls._cache_folder(repo)
        return next((root / folder for root in cls._cache_roots() if (root / folder).is_dir()), None)

    @classmethod
    def model_snapshot_path(cls, repo: str) -> Path | None:
        from steno.backends.base import validate_model_files
        folder = cls._cache_folder(repo)
        backend = "mlx" if repo.startswith("mlx-community/") else "faster-whisper"
        for root in cls._cache_roots():
            cache = root / folder
            ref = cache / "refs" / "main"
            candidates = []
            if ref.is_file():
                revision = ref.read_text(encoding="utf-8").strip()
                if re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
                    candidates.append(cache / "snapshots" / revision)
            if not candidates and (cache / "snapshots").is_dir():
                candidates = sorted((cache / "snapshots").iterdir(), reverse=True)
            for snapshot in candidates:
                try:
                    validate_model_files(snapshot, backend)
                    return snapshot
                except (OSError, ValueError):
                    continue
        return None

    @classmethod
    def model_is_downloaded(cls, repo: str) -> bool:
        return cls.model_snapshot_path(repo) is not None

    @classmethod
    def delete_model_cache(cls, repo: str) -> bool:
        # Shared/legacy caches are migration sources, never deleted by Steno.
        path = cls.models_dir() / cls._cache_folder(repo)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            return True
        return False

    @classmethod
    def model_cache_size_mb(cls, repo: str) -> float:
        path = cls.model_cache_path(repo)
        if path is None:
            return 0.0
        seen = set()
        total = 0
        for file in path.rglob("*"):
            if file.is_file():
                resolved = file.resolve()
                if resolved not in seen:
                    total += file.stat().st_size
                    seen.add(resolved)
        return round(total / (1024 * 1024), 1)
