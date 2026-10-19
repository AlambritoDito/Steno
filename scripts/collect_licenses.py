"""Preserve installed Python dependency license texts in the desktop bundle."""
import importlib.metadata
import json
from pathlib import Path
import shutil


def collect(destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "unknown")
        manifest.append({"name": name, "version": distribution.version,
                         "license": distribution.metadata.get("License-Expression") or distribution.metadata.get("License", "")})
        for file in distribution.files or []:
            filename = Path(file).name.lower()
            if not any(marker in filename for marker in ("license", "copying", "notice")):
                continue
            source = Path(distribution.locate_file(file))
            if source.is_file():
                target = destination / name / Path(file)
                # Distribution file lists can reference locations outside site-packages.
                if not target.resolve().is_relative_to(destination.resolve()):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    (destination / "python-dependencies.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination
