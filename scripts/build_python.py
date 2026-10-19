"""Build the backend natively. Run using `uv run --frozen --extra build` on macOS/Windows."""
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def collect_msvc_runtime():
    """Bundle app-local redistributable CRT/OpenMP, without an elevated installer."""
    configured = os.environ.get("STENO_VC_REDIST_DIR")
    if configured:
        root = Path(configured)
    else:
        locator = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        installation = subprocess.check_output([str(locator), "-latest", "-property", "installationPath"], text=True).strip()
        candidates = sorted((Path(installation) / "VC" / "Redist" / "MSVC").glob("*/x64"))
        if not candidates:
            raise RuntimeError("Install Visual Studio C++ build tools or set STENO_VC_REDIST_DIR")
        root = candidates[-1]
    dlls = {file.name.lower(): file for file in root.rglob("*.dll") if "debug" not in str(file).lower()}
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll"):
        if name not in dlls:
            raise RuntimeError(f"MSVC redistributable missing {name} in {root}")
    return list(dlls.values())


def build_arguments(target, cuda_directory=None, msvc_dlls=()):
    args = [
        "--name", "steno-server", "--onedir", "--noconfirm", "--clean",
        "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "python"),
        "--specpath", str(ROOT / "build"),
        "--add-data", f"{ROOT / 'static'}:static", "--add-data", f"{ROOT / 'locales'}:locales",
        "--add-data", f"{ROOT / 'LICENSE'}:licenses", "--collect-data", "certifi",
        "--collect-all", "av", "--collect-all", "sounddevice", "--collect-all", "_sounddevice_data",
        "--hidden-import", "psutil",
        "--hidden-import", "uvicorn.logging", "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols.http.auto", "--hidden-import", "uvicorn.protocols.http.h11_impl",
        "--hidden-import", "uvicorn.protocols.websockets.auto", "--hidden-import", "uvicorn.protocols.websockets.websockets_impl",
        "--hidden-import", "uvicorn.lifespan.on", "--hidden-import", "anyio._backends._asyncio",
    ]
    for distribution in ("fastapi", "numpy", "av", "sounddevice", "huggingface-hub", "psutil"):
        args += ["--copy-metadata", distribution]
    if target == "darwin":
        args += ["--target-arch", "arm64", "--collect-all", "mlx", "--hidden-import", "mlx_whisper",
                 "--collect-all", "tiktoken", "--hidden-import", "tiktoken_ext.openai_public",
                 "--collect-data", "mlx_whisper", "--hidden-import", "scipy.signal",
                 "--hidden-import", "numba", "--copy-metadata", "mlx-metal"]
        for distribution in ("mlx", "mlx-whisper"):
            args += ["--copy-metadata", distribution]
    elif target == "win32":
        if not cuda_directory:
            raise ValueError("Windows builds must include the pinned CUDA runtime")
        args += ["--collect-all", "faster_whisper", "--collect-all", "ctranslate2",
                 "--collect-all", "onnxruntime", "--add-data", f"{cuda_directory}:cuda"]
        for distribution in ("faster-whisper", "ctranslate2", "onnxruntime"):
            args += ["--copy-metadata", distribution]
        for dll in msvc_dlls:
            args += ["--add-binary", f"{dll}:."]
    else:
        raise ValueError("Only native macOS arm64 and Windows x64 builds are supported")
    args += [str(ROOT / "main.py")]
    return args


def main():
    if sys.platform == "darwin" and platform.machine() != "arm64":
        raise RuntimeError("Build macOS on Apple Silicon without Rosetta")
    cuda = None
    msvc = []
    if sys.platform == "win32":
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            raise RuntimeError("Build Windows with x64 Python")
        from fetch_cuda_runtime import fetch
        cuda = fetch()
        msvc = collect_msvc_runtime()
    from collect_licenses import collect
    licenses = collect(ROOT / "build" / "third-party-licenses")
    import PyInstaller.__main__
    args = build_arguments(sys.platform, cuda, msvc)
    args[-1:-1] = ["--add-data", f"{licenses}:licenses/python"]
    PyInstaller.__main__.run(args)


if __name__ == "__main__":
    main()
