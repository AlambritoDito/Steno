"""Create a Windows portable ZIP without changing the NSIS input directory."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = """Steno portable — Windows 11

1. Extract this entire ZIP into a folder where you can save files, such as Documents.
2. Open Steno.exe. The first model download needs an internet connection.
3. Keep all files together. Your notes, audio, models and logs live in Steno Data.
4. To report a problem, open Settings and download the diagnostics ZIP.
5. Before replacing this app with a newer version, close Steno and back up Steno Data.

Steno portátil — Windows 11

1. Extrae TODO el ZIP en una carpeta donde puedas guardar archivos, como Documentos.
2. Abre Steno.exe. La primera descarga del modelo necesita conexión a internet.
3. Mantén todos los archivos juntos. Tus notas, audio, modelos y registros están en Steno Data.
4. Para reportar un problema, abre Configuración y descarga el ZIP de diagnóstico.
5. Antes de actualizar la aplicación, cierra Steno y respalda Steno Data.
"""


def build_portable(source, destination, version):
    source, destination = Path(source), Path(destination)
    if not (source / "Steno.exe").is_file():
        raise RuntimeError("Build the Windows application before creating its portable ZIP")
    if (source / "portable.json").exists() or (source / "Steno Data").exists():
        raise RuntimeError("Refusing to package a used or modified NSIS input directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(source.rglob("*")):
                if file.is_symlink():
                    raise RuntimeError(f"Unexpected symlink in Windows application: {file.name}")
                if file.is_file():
                    archive.write(file, "Steno/" + file.relative_to(source).as_posix())
            archive.writestr("Steno/portable.json", json.dumps({
                "format": "steno-portable", "version": 1, "application_version": version,
            }, indent=2))
            archive.writestr("Steno/START-HERE.txt", INSTRUCTIONS)
            archive.write(ROOT / "docs" / "PRUEBA-WINDOWS.md", "Steno/LEEME.txt")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main():
    version = json.loads((ROOT / "package.json").read_text())["version"]
    artifact = ROOT / "release" / f"Steno-{version}-win-x64-portable.zip"
    print(build_portable(ROOT / "release" / "win-unpacked", artifact, version))


if __name__ == "__main__":
    main()
