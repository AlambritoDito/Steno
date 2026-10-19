"""Generate native application icons from the tracked Steno brand asset."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "packaging" / "resources"
    target.mkdir(parents=True, exist_ok=True)
    image = Image.open(ROOT / "assets" / "steno-icon.png").convert("RGBA")
    image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    image.save(target / "icon.icns", format="ICNS")
    image.save(target / "icon.ico", format="ICO", sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])


if __name__ == "__main__":
    main()
