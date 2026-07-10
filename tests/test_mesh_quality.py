import numpy as np
import pytest

from src.geometry.mesh_quality import (
    MeshQualityThresholds,
    assert_quality_gate,
    compare_mesh_quality,
    compute_mesh_quality,
    make_quality_gate,
)


def _square_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def test_detects_nan_vertices():
    vertices, faces = _square_mesh()
    vertices[0, 0] = np.nan

    report = compute_mesh_quality(vertices, faces)

    assert report["finite_vertices"] is False


def test_detects_face_count_drop():
    vertices, faces = _square_mesh()
    baseline = compute_mesh_quality(vertices, faces)
    candidate = compute_mesh_quality(vertices, faces[:1])

    comparison = compare_mesh_quality(
        baseline,
        candidate,
        thresholds=MeshQualityThresholds(min_face_ratio=0.98),
    )

    assert comparison["passed"] is False
    assert "face_count_dropped_below_threshold" in comparison["issues"]


def test_detects_degenerate_face_increase():
    vertices, faces = _square_mesh()
    bad_faces = np.array([[0, 1, 2], [0, 0, 0]], dtype=np.int64)
    baseline = compute_mesh_quality(vertices, faces)
    candidate = compute_mesh_quality(vertices, bad_faces)

    gate = make_quality_gate(baseline=baseline, candidate=candidate)

    assert gate["passed"] is False
    assert "degenerate_faces_increased" in gate["comparison"]["issues"]
    with pytest.raises(RuntimeError):
        assert_quality_gate(gate)


def test_detects_boundary_edge_increase():
    vertices, faces = _square_mesh()
    closed_like_baseline = compute_mesh_quality(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.int64),
    )
    open_candidate = compute_mesh_quality(vertices, faces)

    comparison = compare_mesh_quality(closed_like_baseline, open_candidate)

    assert comparison["passed"] is False
    assert "boundary_edges_increased" in comparison["issues"]

