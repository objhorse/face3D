from __future__ import annotations

import numpy as np

from src.geometry.absolute_eyelid_fit import (
    ABSOLUTE_EYELID_PARAMETER_NAMES,
    FACE_FRAME_INDICES,
    AbsoluteEyelidFitConfig,
    apply_absolute_eyelid_parameters,
    compute_absolute_eye_metrics,
    fit_absolute_semantic_eyelids,
)
from src.geometry.eyelid_fit import project_landmarks
from src.geometry.harmonic_semantic_deformer import HarmonicSemanticDeformer
from src.geometry.local_mesh_quality import evaluate_local_mesh_quality
from src.geometry.semantic_eyelid_rig import build_semantic_eyelid_rig


def _face_landmarks() -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.float64)
    points[27:31] = np.array(
        [[0.0, -18.0], [0.0, -12.0], [0.0, -6.0], [0.0, 0.0]]
    )
    points[48:68] = np.array(
        [
            [-14.0, 18.0], [-10.0, 16.0], [-5.0, 15.0], [0.0, 15.0],
            [5.0, 15.0], [10.0, 16.0], [14.0, 18.0], [10.0, 21.0],
            [5.0, 23.0], [0.0, 24.0], [-5.0, 23.0], [-10.0, 21.0],
            [-8.0, 18.0], [-4.0, 17.0], [0.0, 17.0], [4.0, 17.0],
            [8.0, 18.0], [4.0, 20.0], [0.0, 21.0], [-4.0, 20.0],
        ]
    )
    points[36:42] = np.array(
        [
            [-34.0, -5.0], [-28.0, -7.0], [-20.0, -7.0],
            [-14.0, -5.0], [-20.0, -4.0], [-28.0, -4.0],
        ]
    )
    points[42:48] = np.array(
        [
            [14.0, -5.0], [20.0, -7.0], [28.0, -7.0],
            [34.0, -5.0], [28.0, -4.0], [20.0, -4.0],
        ]
    )
    return points


def test_eye_only_translation_is_not_removed_by_face_frame() -> None:
    projected = _face_landmarks()
    observed = projected.copy()
    observed[36:48, 1] += 6.0

    metrics = compute_absolute_eye_metrics(
        projected,
        observed,
        eye_states={"subject_right": "closed", "subject_left": "closed"},
    )

    assert metrics["mean_px"] > 4.0
    assert metrics["eyes"]["subject_right"]["center_delta_y_px"] < -5.5
    assert metrics["eyes"]["subject_left"]["center_delta_y_px"] < -5.5


def test_whole_image_translation_is_removed_by_shared_face_frame() -> None:
    projected = _face_landmarks()
    observed = projected + np.array([17.0, -11.0])

    metrics = compute_absolute_eye_metrics(
        projected,
        observed,
        eye_states={"subject_right": "closed", "subject_left": "closed"},
    )

    assert metrics["mean_px"] < 1e-6
    assert metrics["face_frame_anchor_mean_px"] < 1e-6
    assert len(FACE_FRAME_INDICES) >= 20


def test_eye_width_and_roll_are_reported_in_shared_frame() -> None:
    projected = _face_landmarks()
    observed = projected.copy()
    observed[36, 0] -= 3.0
    observed[39, 0] += 3.0
    observed[42, 1] += 2.0
    observed[45, 1] -= 2.0

    metrics = compute_absolute_eye_metrics(
        projected,
        observed,
        eye_states={"subject_right": "closed", "subject_left": "closed"},
    )

    assert metrics["eyes"]["subject_right"]["width_delta_px"] < -5.5
    assert abs(metrics["eyes"]["subject_left"]["roll_delta_degrees"]) > 5.0


def _patch_mesh(
    offset_x: float,
    offset_y: float,
    vertices: list[list[float]],
    faces: list[list[int]],
) -> int:
    start = len(vertices)
    for y in range(5):
        for x in range(5):
            vertices.append(
                [
                    offset_x + float(x),
                    offset_y + float(y),
                    1.0 + 0.04 * (2 - abs(x - 2)),
                ]
            )
    for y in range(4):
        for x in range(4):
            a = start + y * 5 + x
            faces.extend(([a, a + 1, a + 5], [a + 1, a + 6, a + 5]))
    return start


def _semantic_eye_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    right_start = _patch_mesh(0.0, 0.0, vertices, faces)
    left_start = _patch_mesh(10.0, 0.0, vertices, faces)
    stable_start = _patch_mesh(5.0, 10.0, vertices, faces)
    vertices.append([100.0, 100.0, 100.0])

    mapping = np.zeros((68, 3), dtype=np.int64)
    right = {
        36: [5, 10, 6],
        37: [10, 11, 6],
        38: [11, 12, 7],
        39: [8, 13, 12],
        40: [12, 17, 13],
        41: [11, 16, 12],
    }
    for index, triangle in right.items():
        mapping[index] = np.asarray(triangle) + right_start
        mapping[index + 6] = np.asarray(triangle) + left_start

    stable_triangles = (
        [0, 1, 5],
        [1, 6, 5],
        [6, 7, 11],
        [7, 12, 11],
        [12, 13, 17],
        [13, 18, 17],
    )
    for offset, index in enumerate(FACE_FRAME_INDICES):
        mapping[index] = (
            np.asarray(stable_triangles[offset % len(stable_triangles)])
            + stable_start
        )
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        mapping,
    )


def _camera() -> dict[str, np.ndarray]:
    return {
        "K": np.array(
            [[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
        ),
        "R": np.eye(3),
        "t": np.array([0.0, 0.0, 12.0]),
    }


def test_absolute_optimizer_recovers_eye_relative_translation_and_width() -> None:
    vertices, faces, mapping = _semantic_eye_mesh()
    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        mapping,
        support_rings=3,
        sigma_rings=2.0,
        smoothing_iterations=5,
    )
    target_coefficients = np.zeros(
        len(ABSOLUTE_EYELID_PARAMETER_NAMES),
        dtype=np.float64,
    )
    for eye_name, center_y, width in (
        ("subject_right", 0.18, 0.12),
        ("subject_left", 0.15, 0.10),
    ):
        target_coefficients[
            ABSOLUTE_EYELID_PARAMETER_NAMES.index(f"{eye_name}_center_y")
        ] = center_y
        target_coefficients[
            ABSOLUTE_EYELID_PARAMETER_NAMES.index(f"{eye_name}_width")
        ] = width
    target_vertices, _target_offsets = apply_absolute_eyelid_parameters(
        vertices,
        rig,
        target_coefficients,
    )
    barycentric = np.full((68, 3), 1.0 / 3.0, dtype=np.float64)
    target_projection = project_landmarks(
        target_vertices,
        mapping,
        barycentric,
        _camera(),
    )
    cameras = {
        name: _camera()
        for name in ("front", "subject-left", "subject-right")
    }
    observed = {name: target_projection.copy() for name in cameras}
    view_weights = {
        name: {"subject_right": 1.0, "subject_left": 1.0}
        for name in cameras
    }

    result = fit_absolute_semantic_eyelids(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=mapping,
        barycentric=barycentric,
        cameras=cameras,
        observed_landmarks=observed,
        eye_states={"subject_right": "closed", "subject_left": "closed"},
        view_eye_weights=view_weights,
        cfg=AbsoluteEyelidFitConfig(
            parameter_prior_weight=0.001,
            symmetry_weight=0.0,
            deformation_smoothness_weight=0.001,
            minimum_improvement_ratio=0.01,
            max_nfev=240,
        ),
    )

    assert result.report["accepted"]
    assert result.report["after"]["mean_px"] < 0.35 * result.report["before"]["mean_px"]
    outside = np.setdiff1d(np.arange(len(vertices)), rig.active_vertices)
    np.testing.assert_array_equal(result.vertices[outside], vertices[outside])


def test_absolute_center_parameter_moves_corner_seeds() -> None:
    vertices, faces, mapping = _semantic_eye_mesh()
    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        mapping,
        support_rings=3,
        smoothing_iterations=0,
    )
    coefficients = np.zeros(
        len(ABSOLUTE_EYELID_PARAMETER_NAMES),
        dtype=np.float64,
    )
    coefficients[
        ABSOLUTE_EYELID_PARAMETER_NAMES.index("subject_right_center_y")
    ] = 0.2

    candidate, _offsets = apply_absolute_eyelid_parameters(
        vertices,
        rig,
        coefficients,
    )

    corner_vertices = np.unique(
        np.concatenate(
            (
                rig.control_seeds["subject_right_outer_corner"],
                rig.control_seeds["subject_right_inner_corner"],
            )
        )
    )
    assert np.linalg.norm(
        candidate[corner_vertices] - vertices[corner_vertices],
        axis=1,
    ).max() > 0.0


def test_harmonic_deformer_moves_eye_as_one_smooth_region() -> None:
    vertices, faces, mapping = _semantic_eye_mesh()
    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        mapping,
        support_rings=4,
        smoothing_iterations=0,
    )
    control_basis = np.zeros(
        (1, len(rig.control_names), 3),
        dtype=np.float64,
    )
    for index, name in enumerate(rig.control_names):
        if name.startswith("subject_right"):
            control_basis[0, index, 1] = 0.35
    deformer = HarmonicSemanticDeformer.build(
        vertices,
        faces,
        rig,
        control_basis,
    )

    candidate = deformer.apply(np.array([1.0]))
    quality = evaluate_local_mesh_quality(
        vertices,
        candidate,
        faces,
        active_vertices=rig.active_vertices,
    )

    assert quality["accepted"], quality
    outside = np.setdiff1d(np.arange(len(vertices)), rig.active_vertices)
    np.testing.assert_array_equal(candidate[outside], vertices[outside])
    assert deformer.diagnostics["free_vertex_count"] > 0
