"""Apple Silicon MLX-Whisper adapter."""

import gc
import importlib
import importlib.util
from pathlib import Path

import numpy as np

from steno.backends.base import BackendInferenceError, BackendUnavailableError


class MLXBackend:
    name = "mlx"

    def __init__(self, device: str = "auto"):
        if device not in {"auto", "metal"}:
            raise ValueError("MLX requires the Metal device")
        self.model_path = None
        self._module = None
        self._validated = False

    def capabilities(self) -> dict:
        return {
            "backend": self.name, "device": "metal", "compute_type": "float16",
            "available": importlib.util.find_spec("mlx_whisper") is not None,
            "cuda_available": False, "fallback_reason": None,
            "validated": self._validated,
        }

    def load(self, model_path: Path) -> None:
        try:
            self._module = importlib.import_module("mlx_whisper")
            self.model_path = str(model_path)
            # Exercise native weights and kernels before reporting ready.
            self._module.transcribe(
                np.zeros(16000, dtype=np.float32), path_or_hf_repo=self.model_path,
                language="en", verbose=None, condition_on_previous_text=False,
            )
            self._validated = True
        except Exception as exc:
            self.unload()
            raise BackendUnavailableError(f"MLX initialization failed: {exc}") from exc

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[dict]:
        try:
            result = self._module.transcribe(
                audio, path_or_hf_repo=self.model_path, language=language,
                verbose=None, condition_on_previous_text=False,
            )
            segments = [
                {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
                for s in result.get("segments", []) if s.get("text", "").strip()
            ]
            if not segments and result.get("text", "").strip():
                segments = [{"start": 0.0, "end": len(audio) / 16000, "text": result["text"].strip()}]
            return segments
        except Exception as exc:
            raise BackendInferenceError(f"MLX transcription failed: {exc}") from exc

    def unload(self) -> None:
        if self._module is not None:
            # mlx-whisper owns the model in ModelHolder, not on this facade.
            transcribe_module = importlib.import_module("mlx_whisper.transcribe")
            holder = transcribe_module.ModelHolder
            if holder.model_path == self.model_path:
                holder.model = None
                holder.model_path = None
            gc.collect()
            mx = importlib.import_module("mlx.core")
            mx.clear_cache()
        self._module = None
        self.model_path = None
        self._validated = False
