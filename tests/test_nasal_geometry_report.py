from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.reports.nasal_geometry_report import (
    validate_minimal_nasal_candidate,
    write_nasal_geometry_report,
)


def _triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv = vertices[:, :2].copy()
    return vertices, faces, uv, faces.copy()


def _validity(candidate: np.ndarray, **overrides):
    vertices, faces, uv, uv_faces = _triangle()
    arguments = {
        "parameters": np.array([0.0, 0.1]),
        "baseline_vertices": vertices,
        "candidate_vertices": candidate,
        "baseline_faces": faces,
        "candidate_faces": faces,
        "baseline_uv_vertices": uv,
        "candidate_uv_vertices": uv,
        "baseline_uv_faces": uv_faces,
        "candidate_uv_faces": uv_faces,
    }
    arguments.update(overrides)
    return validate_minimal_nasal_candidate(**arguments)


def test_minimal_validity_accepts_finite_same_topology_candidate() -> None:
    vertices, _faces, _uv, _uv_faces = _triangle()
    candidate = vertices.copy()
    candidate[2, 2] = 0.1

    report = _validity(candidate)

    assert report["passed"]
    assert report["face_flips"]["count"] == 0


def test_minimal_validity_detects_nan() -> None:
    vertices, _faces, _uv, _uv_faces = _triangle()
    vertices[0, 0] = np.nan

    report = _validity(vertices)

    assert not report["passed"]
    assert "vertices_are_nonfinite_or_have_changed_count" in report["issues"]


def test_minimal_validity_detects_topology_change() -> None:
    vertices, _faces, _uv, _uv_faces = _triangle()

    report = _validity(
        vertices,
        candidate_faces=np.array([[0, 2, 1]], dtype=np.int64),
    )

    assert not report["passed"]
    assert "face_topology_changed" in report["issues"]


def test_minimal_validity_detects_new_degenerate_face() -> None:
    vertices, _faces, _uv, _uv_faces = _triangle()
    candidate = vertices.copy()
    candidate[2] = np.array([0.5, 0.0, 0.0])

    report = _validity(candidate)

    assert not report["passed"]
    assert "degenerate_faces_increased" in report["issues"]


def test_minimal_validity_detects_relative_face_flip() -> None:
    vertices, _faces, _uv, _uv_faces = _triangle()
    candidate = vertices.copy()
    candidate[2, 1] = -1.0

    report = _validity(candidate)

    assert not report["passed"]
    assert report["face_flips"]["indices"] == [0]
    assert "new_face_flips" in report["issues"]


def test_geometry_report_writes_six_images_and_offline_html(
    tmp_path: Path,
) -> None:
    screenshots = {"baseline": {}, "candidate": {}}
    for role in screenshots:
        for view in ("front", "subject-left", "subject-right"):
            path = tmp_path / f"{role}_{view}.png"
            Image.new("RGB", (8, 8), (40, 50, 60)).save(path)
            screenshots[role][view] = path
    objective = {
        "raw_costs": {"front": 2.0},
        "robust_costs": {"front": 1.5},
    }
    validity = {
        "passed": True,
        "quality": {
            "comparison": {
                "degenerate_delta": 0,
                "nonmanifold_edge_delta": 0,
                "boundary_edge_delta": 0,
            }
        },
        "face_flips": {"count": 0},
    }

    report = write_nasal_geometry_report(
        tmp_path,
        dataset_label="captures_fixture",
        screenshots=screenshots,
        baseline_objective=objective,
        candidate_objective=objective,
        evidence={
            "per_view_effective_confidence_sums": {
                "front": 1.0,
                "subject-left": 0.0,
                "subject-right": 1.0,
            }
        },
        validity=validity,
    )

    text = report.read_text(encoding="utf-8")
    assert "captures_fixture" in text
    assert "Low-confidence evidence: subject-left" in text
    assert "Texture scoring is not part" in text
    assert "baseline_front.png" in text
    assert "candidate_subject-right.png" in text


def test_geometry_report_rejects_missing_screenshot(tmp_path: Path) -> None:
    screenshots = {
        role: {
            view: tmp_path / f"{role}_{view}.png"
            for view in ("front", "subject-left", "subject-right")
        }
        for role in ("baseline", "candidate")
    }
    with pytest.raises(FileNotFoundError, match="screenshot"):
        write_nasal_geometry_report(
            tmp_path,
            dataset_label="fixture",
            screenshots=screenshots,
            baseline_objective={},
            candidate_objective={},
            evidence={},
            validity={},
        )
