from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from run_nasal_observation_audit import _load_capture_images
from run_nasal_texture_observation_audit import (
    _relative_artifact_paths,
    evaluate_release_a_gate,
    load_locked_v10_surface,
    point_to_supported_surface_distances,
    subdivide_low_resolution_vertex_masks,
    verify_hash_locked_input,
)


def test_relative_artifact_paths_handles_nested_render_mapping(tmp_path: Path) -> None:
    renders = {
        "baseline": {"front": tmp_path / "debug" / "baseline_front.png"},
        "candidate": {"front": tmp_path / "debug" / "candidate_front.png"},
    }
    assert _relative_artifact_paths(renders, tmp_path) == {
        "baseline": {"front": "debug/baseline_front.png"},
        "candidate": {"front": "debug/candidate_front.png"},
    }
from src.cross_view_geometry import Camera
from src.geometry.nasal_texture_observations import (
    NasalCoordinateProvenance,
    NasalEpipolarMatchResult,
    NasalPairMatch,
    NasalTextureObservationBundle,
    TrustedNasalObservation,
)
from src.geometry.nasal_view_registration import (
    FixedNasalViewRegistration,
    NasalViewRegistrationConfig,
)
from src.geometry.profile_triangulation import ProfileRig
from src.reports.nasal_texture_observation_report import (
    write_nasal_texture_observation_report,
)


def _provenance(view: str) -> NasalCoordinateProvenance:
    return NasalCoordinateProvenance(
        semantic_view=view,
        source_size=(200, 120),
        work_size=(200, 120),
        source_to_work=np.eye(3),
        undistorted=True,
    )


def _trusted(side: str, index: int) -> TrustedNasalObservation:
    region = ("soft_triangle", "alar_dome", "alar_groove")[index % 3]
    pair = NasalPairMatch(
        side_view=side,
        front_pixel=np.asarray([20.0 + 28.0 * index, 35.0 if side == "subject-left" else 78.0]),
        side_pixel=np.asarray([18.0 + 28.0 * index, 36.0 if side == "subject-left" else 79.0]),
        confidence=0.9,
        semantic_region=region,
        source_matcher="synthetic",
        provenance_by_view={"front": _provenance("front"), side: _provenance(side)},
        diagnostics={"uniqueness_margin": 0.2},
    )
    return TrustedNasalObservation(
        pair_match=pair,
        point_reference_m=np.asarray([0.001 * index, 0.0, 0.50]),
        covariance_proxy=np.eye(3) * 1e-6,
        depths_m={"front": 0.50, side: 0.51},
        reprojection_errors_px={"front": 0.2, side: 0.3},
        ray_angle_deg=10.0,
        weight=0.8,
    )


def _bundle(count_per_side: int) -> NasalTextureObservationBundle:
    return NasalTextureObservationBundle(
        trusted=tuple(
            _trusted(side, index)
            for side in ("subject-left", "subject-right")
            for index in range(count_per_side)
        ),
        rejected=(),
        metadata={"test": True},
    )


def _registration() -> FixedNasalViewRegistration:
    views = ("front", "subject-left", "subject-right")
    return FixedNasalViewRegistration(
        offsets_by_view={view: np.zeros(2) for view in views},
        confidence_by_view={view: 0.0 for view in views},
        sample_counts_by_view={view: 0 for view in views},
        inlier_counts_by_view={view: 0 for view in views},
        semantic_region_counts_by_view={view: 0 for view in views},
        residual_p90_px_by_view={view: 0.0 for view in views},
        residual_quantiles_px_by_view={view: np.zeros(3) for view in views},
        offset_confidence_interval95_px_by_view={view: np.zeros((2, 2)) for view in views},
        accepted_by_view={view: False for view in views},
        config=NasalViewRegistrationConfig(),
    )


def test_hash_lock_rejects_mismatch(tmp_path) -> None:
    source = tmp_path / "source.obj"
    source.write_bytes(b"immutable geometry")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_hash_locked_input(source, "0" * 64, "geometry")


def test_semantic_masks_follow_exact_subdivision_topology() -> None:
    import trimesh

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    propagated, propagated_faces = subdivide_low_resolution_vertex_masks(
        faces,
        {"nose": np.asarray([True, False, False])},
        iterations=1,
    )
    _vertices_sub, expected_faces = trimesh.remesh.subdivide_loop(
        vertices,
        faces,
        iterations=1,
    )
    assert np.array_equal(propagated_faces, expected_faces)
    assert np.array_equal(propagated["nose"][:3], [True, False, False])
    assert np.count_nonzero(propagated["nose"]) == 3


def test_locked_v10_loader_uses_obj_vertices_and_verifies_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    expected_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    expected_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    obj = tmp_path / "locked.obj"
    obj.write_text(
        "v 0.000000 0.000000 0.000000\n"
        "v 1.000000 0.000000 0.000000\n"
        "v 0.000000 1.000000 0.000000\n"
        "vt 0.000000 0.000000\n"
        "vt 1.000000 0.000000\n"
        "vt 0.000000 1.000000\n"
        "f 1/1 2/2 3/3\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        "run_nasal_texture_observation_audit._subdivide_candidate",
        lambda _baseline, _vertices: (
            expected_vertices,
            expected_faces,
            np.zeros((3, 2)),
            expected_faces,
        ),
    )
    loaded_vertices, loaded_faces, metadata = load_locked_v10_surface(
        obj,
        object(),
        expected_vertices,
    )
    assert np.array_equal(loaded_vertices, expected_vertices)
    assert np.array_equal(loaded_faces, expected_faces)
    assert metadata["vertex_count"] == 3


def test_model_distance_is_measured_to_triangle_surface() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    point = np.asarray([[0.25, 0.25, 0.1]], dtype=np.float64)
    distance = point_to_supported_surface_distances(
        point,
        vertices,
        faces,
        np.ones(3, dtype=bool),
    )
    assert distance == pytest.approx([0.1])
    nearest_vertex_distance = np.min(np.linalg.norm(vertices - point[0], axis=1))
    assert distance[0] < nearest_vertex_distance


def test_release_a_gate_reports_insufficient_and_success() -> None:
    failed = evaluate_release_a_gate(_bundle(2))
    assert failed["status"] == "insufficient_texture_evidence"
    assert not failed["gates"]["minimum_trusted_per_side"]

    passed = evaluate_release_a_gate(_bundle(6))
    assert passed["passed"]
    assert passed["status"] == "ready_for_geometry"
    assert all(passed["gates"].values())
    assert passed["spatial_diameter_px_by_side"]["subject-left"] >= 18.0
    assert passed["unique_anchor_count_by_side"]["subject-right"] == 6


def test_report_generation_does_not_create_or_modify_geometry(tmp_path) -> None:
    source_geometry = tmp_path / "locked.obj"
    source_geometry.write_bytes(b"v10 geometry remains unchanged")
    before = source_geometry.read_bytes()
    bundle = _bundle(6)
    results = {
        side: NasalEpipolarMatchResult(
            matches=tuple(value.pair_match for value in bundle.by_side[side]),
            rejected=(),
        )
        for side in ("subject-left", "subject-right")
    }
    images = {
        "front": np.full((120, 200, 3), 128, dtype=np.uint8),
        "subject-left": np.full((120, 200, 3), 128, dtype=np.uint8),
        "subject-right": np.full((120, 200, 3), 128, dtype=np.uint8),
    }
    output = tmp_path / "report"
    report = write_nasal_texture_observation_report(
        output,
        images_by_view=images,
        matches_by_side=results,
        observations=bundle,
        registration=_registration(),
        gate=evaluate_release_a_gate(bundle),
        model_distances_m=np.full(len(bundle.trusted), 0.001),
        metadata={"geometry_modified": False},
    )
    assert report.is_file()
    assert (output / "metrics.json").is_file()
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["geometry_modified"] is False
    assert source_geometry.read_bytes() == before
    assert not list(output.rglob("*.obj"))
    assert not list(output.rglob("*.glb"))


def test_capture_loader_rejects_missing_camera_view(tmp_path) -> None:
    def camera(name: str, view: str) -> Camera:
        return Camera(
            name=name,
            view=view,
            image_size=(20, 16),
            K=np.eye(3),
            dist=np.zeros(5),
            R_rig_to_camera=np.eye(3),
            t_rig_to_camera=np.zeros(3),
        )

    rig = ProfileRig(
        cameras_by_view={
            "left": camera("camera1", "left"),
            "front": camera("camera2", "front"),
            "right": camera("camera3", "right"),
        },
        reference_view="front",
        units="meters",
        calibration_path="synthetic",
        stereo_rms_px={"left": 0.1, "front": 0.0, "right": 0.1},
    )
    cv2.imwrite(str(tmp_path / "camera2_only.jpg"), np.zeros((16, 20, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected one RGB capture"):
        _load_capture_images(tmp_path, rig)
