"""Lazy native inference adapters; importing Steno never loads a GPU runtime."""

from steno.backends.base import BackendUnavailableError


def create_backend(name: str, device: str = "auto"):
    if name == "mlx":
        from steno.backends.mlx import MLXBackend
        return MLXBackend(device=device)
    if name == "faster-whisper":
        from steno.backends.faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend(device=device)
    return UnsupportedBackend()


class UnsupportedBackend:
    name = "unsupported"

    def capabilities(self):
        return {
            "backend": self.name, "available": False, "device": None,
            "compute_type": None, "cuda_available": False,
            "fallback_reason": "unsupported_platform", "validated": False,
        }

    def load(self, model_path):
        raise BackendUnavailableError("Steno requires Apple Silicon macOS or Windows x64")

    def unload(self):
        pass
