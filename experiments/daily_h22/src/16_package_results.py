"""Cria manifesto reproduzível com hashes dos outputs disponíveis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

import config as C
from utils import ensure_directories, save_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> dict:
    ensure_directories()
    manifest_path = C.RESULTS_DIR / "artifact_manifest.json"
    files = []
    for path in sorted(C.RESULTS_DIR.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith(".db-journal"):
            continue
        files.append({
            "path": str(path.relative_to(C.RESULTS_DIR)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "statsmodels", "arch",
                    "xgboost", "optuna", "torch", "captum", "pyarrow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "experiment": "daily_h22_v2",
        "target": C.TARGET,
        "horizon": C.HORIZON,
        "window": {"train_years": C.TRAIN_YEARS, "validation_years": C.VAL_YEARS,
                   "test_years": C.TEST_YEARS, "step_years": C.STEP_YEARS},
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }
    save_json(payload, manifest_path)
    print(f"Manifesto: {manifest_path}; arquivos={len(files)}")
    return payload


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

