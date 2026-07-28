from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.reports.nasal_geometry_report as geometry_report
from src.reports.nasal_geometry_report import (
    validate_minimal_nasal_candidate,
    write_nasal_evidence_overlays,
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


@pytest.mark.parametrize(
    ("override_name", "override_value"),
    [
        ("candidate_vertices", "not-an-array"),
        ("candidate_vertices", np.zeros((2, 2))),
        ("candidate_faces", np.array([[0, 1, 99]], dtype=np.int64)),
        ("candidate_faces", np.array([0, 1, 2], dtype=np.int64)),
    ],
)
def test_minimal_validity_returns_failure_for_malformed_mesh_inputs(
    override_name: str,
    override_value,
) -> None:
    vertices, _faces, _uv, _uv_faces = _triangle()

    report = _validity(vertices, **{override_name: override_value})

    assert not report["passed"]
    assert report["issues"]


def test_minimal_validity_rejects_empty_mesh_and_uv_topology() -> None:
    report = validate_minimal_nasal_candidate(
        parameters=np.array([0.0]),
        baseline_vertices=np.empty((0, 3)),
        candidate_vertices=np.empty((0, 3)),
        baseline_faces=np.empty((0, 3), dtype=np.int64),
        candidate_faces=np.empty((0, 3), dtype=np.int64),
        baseline_uv_vertices=np.empty((0, 2)),
        candidate_uv_vertices=np.empty((0, 2)),
        baseline_uv_faces=np.empty((0, 3), dtype=np.int64),
        candidate_uv_faces=np.empty((0, 3), dtype=np.int64),
    )

    assert not report["passed"]
    assert not report["checks"]["nonempty_vertices"]
    assert not report["checks"]["nonempty_faces"]
    assert not report["checks"]["nonempty_uv_vertices"]
    assert not report["checks"]["nonempty_uv_faces"]


def test_minimal_validity_rejects_out_of_range_uv_indices_without_crashing() -> None:
    vertices, faces, uv, _uv_faces = _triangle()
    invalid_uv_faces = np.array([[0, 1, 9]], dtype=np.int64)

    report = _validity(
        vertices,
        baseline_uv_vertices=uv,
        candidate_uv_vertices=uv,
        baseline_uv_faces=invalid_uv_faces,
        candidate_uv_faces=invalid_uv_faces.copy(),
        baseline_faces=faces,
        candidate_faces=faces.copy(),
    )

    assert not report["passed"]
    assert not report["checks"]["valid_uv_face_indices"]
    assert "uv_topology_or_coordinates_changed" in report["issues"]


def test_geometry_renderer_replaces_source_material() -> None:
    source_mesh = type("SourceMesh", (), {"visual": object()})()

    class Graph:
        nodes_geometry = ("face",)

        def __getitem__(self, _name):
            return np.eye(4), "face_geometry"

    source_scene = type(
        "SourceScene",
        (),
        {"graph": Graph(), "geometry": {"face_geometry": source_mesh}},
    )()
    calls = []

    class FakeScene:
        def __init__(self, **_kwargs):
            self.nodes = []

        def add(self, mesh, pose):
            self.nodes.append((mesh, pose))

    class FakeMaterial:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMesh:
        @staticmethod
        def from_trimesh(mesh, **kwargs):
            calls.append((mesh, kwargs))
            return "uniform-mesh"

    fake_pyrender = type(
        "FakePyrender",
        (),
        {
            "Scene": FakeScene,
            "MetallicRoughnessMaterial": FakeMaterial,
            "Mesh": FakeMesh,
        },
    )

    rendered = geometry_report._uniform_geometry_scene(
        source_scene,
        fake_pyrender,
    )

    assert len(rendered.nodes) == 1
    assert calls[0][0] is source_mesh
    assert isinstance(calls[0][1]["material"], FakeMaterial)
    assert calls[0][1]["material"] is not source_mesh.visual
    assert calls[0][1]["smooth"] is True


def test_geometry_report_writes_six_images_and_offline_html(
    tmp_path: Path,
) -> None:
    screenshots = {"baseline": {}, "candidate": {}}
    for role in screenshots:
        for view in ("front", "subject-left", "subject-right"):
            path = tmp_path / f"{role}_{view}.png"
            Image.new("RGB", (8, 8), (40, 50, 60)).save(path)
            screenshots[role][view] = path
    evidence_overlays = write_nasal_evidence_overlays(
        tmp_path,
        work_images_by_view={
            view: np.full((40, 60, 3), 30, dtype=np.uint8)
            for view in ("front", "subject-left", "subject-right")
        },
        observation_curves_by_view={
            view: {"target": np.array([[10.0, 10.0], [20.0, 20.0]])}
            for view in ("front", "subject-left", "subject-right")
        },
        baseline_projection_by_view={
            view: np.array([[12.0, 10.0], [18.0, 18.0]])
            for view in ("front", "subject-left", "subject-right")
        },
        candidate_projection_by_view={
            view: np.array([[13.0, 10.0], [19.0, 18.0]])
            for view in ("front", "subject-left", "subject-right")
        },
    )
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
        evidence_overlays=evidence_overlays,
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
    assert "texture scoring is not part" in text
    assert "Uniform-material geometry renders" in text
    assert "Original-image projection evidence" in text
    assert "Observation" in text
    assert "Baseline projection" in text
    assert "Candidate projection" in text
    assert "baseline_front.png" in text
    assert "candidate_subject-right.png" in text
    assert "evidence_subject-left.png" in text


def test_geometry_report_rejects_missing_screenshot(tmp_path: Path) -> None:
    screenshots = {
        role: {
            view: tmp_path / f"{role}_{view}.png"
            for view in ("front", "subject-left", "subject-right")
        }
        for role in ("baseline", "candidate")
    }
    evidence_overlays = {
        view: tmp_path / f"evidence_{view}.png"
        for view in ("front", "subject-left", "subject-right")
    }
    with pytest.raises(FileNotFoundError, match="evidence image"):
        write_nasal_geometry_report(
            tmp_path,
            dataset_label="fixture",
            screenshots=screenshots,
            evidence_overlays=evidence_overlays,
            baseline_objective={},
            candidate_objective={},
            evidence={},
            validity={},
        )
