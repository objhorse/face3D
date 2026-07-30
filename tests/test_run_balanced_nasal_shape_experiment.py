from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import run_balanced_nasal_shape_experiment as runner


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_verify_baseline_hash_rejects_mismatch(tmp_path: Path):
    baseline = tmp_path / "face_same_texture.glb"
    baseline.write_bytes(b"canonical-baseline")

    with pytest.raises(RuntimeError, match="baseline GLB hash mismatch"):
        runner.verify_baseline_glb(
            baseline,
            "0" * 64,
        )


def test_balanced_runner_locks_baseline_and_writes_named_aliases(
    tmp_path: Path,
    monkeypatch,
):
    captures = tmp_path / "captures_20260612_135253"
    source = tmp_path / "source"
    output = tmp_path / "balanced"
    rig = tmp_path / "rig.json"
    captures.mkdir()
    (source / "meshes").mkdir(parents=True)
    baseline_payload = b"canonical-baseline"
    baseline = source / "meshes" / "face_same_texture.glb"
    baseline.write_bytes(baseline_payload)
    rig.write_text("{}", encoding="utf-8")
    expected_hash = _sha256(baseline_payload)
    calls = []

    def run_existing(
        capture_dir,
        source_output,
        output_dir,
        *,
        rig_calibration,
        viewer_template=None,
        optimization_strategy,
        expected_baseline_glb_sha256,
    ):
        calls.append(
            {
                "capture_dir": Path(capture_dir),
                "source_output": Path(source_output),
                "output": Path(output_dir),
                "rig": Path(rig_calibration),
                "viewer_template": viewer_template,
                "strategy": optimization_strategy,
                "hash": expected_baseline_glb_sha256,
            }
        )
        target = Path(output_dir)
        (target / "meshes").mkdir(parents=True)
        (target / "meshes" / "face_mesh.glb").write_bytes(
            b"candidate-geometry"
        )
        (target / "meshes" / "face_same_texture.glb").write_bytes(
            b"candidate-textured"
        )
        (target / "nasal_shape_compare.html").write_text(
            "<html>viewer</html>",
            encoding="utf-8",
        )
        report = target / "nasal_fit_report.json"
        report.write_text(
            json.dumps({"status": "success"}),
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        runner,
        "run_multiview_nasal_shape_experiment",
        run_existing,
    )

    report = runner.run_balanced_nasal_shape_experiment(
        captures,
        source,
        output,
        rig_calibration=rig,
        expected_baseline_glb_sha256=expected_hash,
    )

    assert calls[0]["strategy"] == "balanced_semantic_v4"
    assert calls[0]["hash"] == expected_hash
    assert report == output / "fit_report.json"
    assert (output / "meshes" / "baseline.glb").read_bytes() == baseline_payload
    assert (output / "meshes" / "candidate.glb").read_bytes() == (
        b"candidate-geometry"
    )
    assert (output / "meshes" / "candidate_textured.glb").read_bytes() == (
        b"candidate-textured"
    )
    assert (output / "balanced_nasal_compare.html").is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["optimization_strategy"] == "balanced_semantic_v4"
    assert payload["canonical_baseline"]["sha256"] == expected_hash
    assert payload["canonical_baseline"]["copied_unchanged"] is True
