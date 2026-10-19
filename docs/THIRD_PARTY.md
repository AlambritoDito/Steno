# Third-party components

The application's MIT license does not replace dependency licenses.

- marked 12: MIT; vendored `static/vendor/marked.min.js` retains its notice.
- DOMPurify 3.4.17: Apache-2.0 or MPL-2.0; see `static/vendor/DOMPurify-LICENSE`. The distribution file is pinned and vendored so rendering works offline.
- Electron, electron-builder, FastAPI, sounddevice/PortAudio, NumPy, MLX/MLX-Whisper and faster-whisper/CTranslate2 retain their upstream notices in the distribution.
- PyAV includes FFmpeg libraries. Include the wheel's license/notices and preserve its redistribution terms; the application does not invoke an external FFmpeg executable.
- NVIDIA CUDA runtime, cuBLAS, NVRTC and cuDNN: downloaded from NVIDIA's official redistributable archives, verified against `packaging/cuda-runtime.json`. The packaging helper preserves bundled license files. The NVIDIA driver is never redistributed by Steno.
- Whisper models download separately from their model repositories. Review the selected model's model card and license before redistribution; models are excluded from installers.

Dependency versions are fixed in `uv.lock`, `package-lock.json` and the CUDA manifest. Generated build artifacts and model weights are not source-controlled.
