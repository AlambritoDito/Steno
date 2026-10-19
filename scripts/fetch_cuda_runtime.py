"""Fetch pinned NVIDIA redistributables for the Windows build, with SHA-256 checks."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_runtime(archive, destination, component):
    """Extract only app-local DLLs and licenses; never trust archive paths."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/")).name
            if not name or member.is_dir():
                continue
            if name.lower().endswith(".dll"):
                target = destination / name
            elif "license" in name.lower() or name.lower().startswith("eula"):
                target = destination / "licenses" / f"{component}-{name}"
                target.parent.mkdir(exist_ok=True)
            else:
                continue
            with source.open(member) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def fetch(destination=ROOT / "build" / "cuda-runtime"):
    manifest = json.loads((ROOT / "packaging" / "cuda-runtime.json").read_text())
    cache = ROOT / ".cache" / "redistributables"
    cache.mkdir(parents=True, exist_ok=True)
    for artifact in manifest["artifacts"]:
        archive = cache / artifact["url"].rsplit("/", 1)[-1]
        if not archive.exists() or digest(archive) != artifact["sha256"]:
            partial = archive.with_suffix(".partial")
            print(f"Downloading {artifact['name']}...", flush=True)
            with urllib.request.urlopen(artifact["url"], timeout=120) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target)
            if digest(partial) != artifact["sha256"]:
                partial.unlink()
                raise RuntimeError(f"Checksum mismatch: {artifact['name']}")
            partial.replace(archive)
        extract_runtime(archive, destination, artifact["name"])
    missing = [name for name in REQUIRED if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing runtime DLLs: {missing}")
    shutil.copy2(ROOT / "packaging" / "cuda-runtime.json", destination / "manifest.json")
    return destination


if __name__ == "__main__":
    print(fetch())
