"""Retain the neural forecast CSV exports as verified, lossless gzip files."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def pack(folder: Path):
    manifest = folder / "csv_compression_manifest.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else {}
    for source in sorted(folder.rglob("*.csv")):
        target = source.with_suffix(".csv.gz")
        temp = target.with_suffix(".tmp")
        with source.open("rb") as original:
            digest = hashlib.file_digest(original, "sha256").hexdigest()
            original.seek(0)
            with temp.open("wb") as raw, gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6,
            ) as out:
                shutil.copyfileobj(original, out)
        with gzip.open(temp, "rb") as saved:
            if hashlib.file_digest(saved, "sha256").hexdigest() != digest:
                raise ValueError(f"Compression did not preserve {source}")
        entries[str(target.relative_to(folder))] = {
            "original_bytes": source.stat().st_size, "original_sha256": digest,
        }
        temp.replace(target)
        source.unlink()
    manifest.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"Verified {len(entries)} forecast CSVs in {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast_directory", type=Path)
    pack(parser.parse_args().forecast_directory)
