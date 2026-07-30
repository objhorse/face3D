"""Run Stage A balanced semantic nasal fitting from a hash-locked baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from run_multiview_nasal_shape_experiment import (
    run_multiview_nasal_shape_experiment,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline_glb(path: str | Path, expected_sha256: str) -> str:
    baseline = Path(path).resolve()
    if not baseline.is_file() or baseline.stat().st_size <= 0:
        raise FileNotFoundError(f"baseline GLB does not exist: {baseline}")
    expected = str(expected_sha256).lower()
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("expected_sha256 must be 64 lowercase hex digits")
    actual = _sha256_file(baseline)
    if actual.lower() != expected:
        raise RuntimeError(
            f"baseline GLB hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def _copy_new(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite balanced nasal artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = _sha256_file(source)
    destination_hash = _sha256_file(destination)
    if destination_hash != source_hash:
        raise RuntimeError(
            f"copied artifact hash mismatch: {destination}"
        )
    return destination_hash


def run_balanced_nasal_shape_experiment(
    capture_dir: str | Path,
    source_output: str | Path,
    output: str | Path,
    *,
    rig_calibration: str | Path,
    expected_baseline_glb_sha256: str,
    viewer_template: str | Path | None = None,
) -> Path:
    """Run balanced v4 and publish stable, explicitly named A/B artifacts."""
    source = Path(source_output).resolve()
    target = Path(output).resolve()
    baseline_source = source / "meshes" / "face_same_texture.glb"
    baseline_hash = verify_baseline_glb(
        baseline_source,
        expected_baseline_glb_sha256,
    )
    legacy_report = run_multiview_nasal_shape_experiment(
        capture_dir,
        source,
        target,
        rig_calibration=rig_calibration,
        viewer_template=viewer_template,
        optimization_strategy="balanced_semantic_v4",
        expected_baseline_glb_sha256=baseline_hash,
    )
    aliases = {
        "baseline_glb": (
            baseline_source,
            target / "meshes" / "baseline.glb",
        ),
        "candidate_geometry_glb": (
            target / "meshes" / "face_mesh.glb",
            target / "meshes" / "candidate.glb",
        ),
        "candidate_textured_glb": (
            target / "meshes" / "face_same_texture.glb",
            target / "meshes" / "candidate_textured.glb",
        ),
        "viewer": (
            target / "nasal_shape_compare.html",
            target / "balanced_nasal_compare.html",
        ),
    }
    artifact_hashes = {
        name: _copy_new(source_path, destination)
        for name, (source_path, destination) in aliases.items()
    }
    payload = json.loads(legacy_report.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema": "balanced-semantic-nasal-experiment-v4",
            "optimization_strategy": "balanced_semantic_v4",
            "canonical_baseline": {
                "source_path": str(baseline_source),
                "copied_path": str(aliases["baseline_glb"][1]),
                "sha256": baseline_hash,
                "copied_unchanged": (
                    artifact_hashes["baseline_glb"] == baseline_hash
                ),
            },
            "balanced_artifacts": {
                name: {
                    "path": str(destination),
                    "sha256": artifact_hashes[name],
                }
                for name, (_source, destination) in aliases.items()
            },
        }
    )
    report = target / "fit_report.json"
    if report.exists():
        raise FileExistsError(
            f"refusing to overwrite balanced nasal report: {report}"
        )
    temporary_report = target / ".fit_report.json.tmp"
    temporary_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_report, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage A balanced semantic nasal fitting."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rig-calibration", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_balanced_nasal_shape_experiment(
        args.capture_dir,
        args.source_output,
        args.output,
        rig_calibration=args.rig_calibration,
        expected_baseline_glb_sha256=args.expected_baseline_sha256,
        viewer_template=args.viewer_template,
    )
    print(f"Balanced nasal fit report: {report}")


if __name__ == "__main__":
    main()


__all__ = [
    "run_balanced_nasal_shape_experiment",
    "verify_baseline_glb",
]
