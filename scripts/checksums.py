"""Write checksums only for completed installer artifacts."""
import hashlib
from pathlib import Path

release = Path(__file__).resolve().parents[1] / "release"
artifacts = sorted([*release.glob("*.dmg"), *release.glob("*.exe")])
if not artifacts:
    raise SystemExit("No installer artifacts found")
lines = []
for artifact in artifacts:
    with artifact.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    lines.append(f"{checksum}  {artifact.name}")
(release / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
